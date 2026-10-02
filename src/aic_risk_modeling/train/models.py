"""Defines PyTorch models used in training.

All models take channels-last inputs, exactly as produced by the tf.data
pipeline (images are (batch, H, W, C), time series are (batch, T, H, W, C) or
(batch, T, features)), and return channels-last outputs. Layouts are permuted
to channels-first internally where torch layers require it.

Branch models expose `input_name` (the key of the input dict they consume)
and `out_channels` (feature channels of their output) so `decoder_fusion`
can route inputs and size its first convolution.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

PATCH_SIZE = 128  # Spatial size that non-image branches are broadcast to


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SeparableConv2d(nn.Module):
    """Depthwise + pointwise convolution (Keras SeparableConv2D equivalent)."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size // 2, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super().__init__()
        self.conv1 = SeparableConv2d(in_channels, num_filters, 3)
        self.conv2 = SeparableConv2d(num_filters, num_filters, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super().__init__()
        self.conv = ConvBlock(in_channels, num_filters)

    def forward(self, x):
        skip = self.conv(x)
        return skip, F.max_pool2d(skip, 2)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, num_filters):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, num_filters, 2, stride=2)
        self.conv = ConvBlock(num_filters + skip_channels, num_filters)

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))


class ConvLSTM2d(nn.Module):
    """Single-layer ConvLSTM (Keras ConvLSTM2D equivalent), batch-first.

    Input is (batch, T, C, H, W). Returns the full hidden sequence
    (batch, T, hidden, H, W) when `return_sequences`, else the final hidden
    state (batch, hidden, H, W).
    """

    def __init__(self, in_channels, hidden_channels, kernel_size,
                 return_sequences=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.return_sequences = return_sequences
        self.gates = nn.Conv2d(in_channels + hidden_channels,
                               4 * hidden_channels, kernel_size,
                               padding=kernel_size // 2)
        # Forget-gate bias starts at 1 (Keras unit_forget_bias)
        with torch.no_grad():
            self.gates.bias.zero_()
            self.gates.bias[hidden_channels:2 * hidden_channels].fill_(1.0)

    def forward(self, x):
        batch, steps, _, height, width = x.shape
        hidden = x.new_zeros(batch, self.hidden_channels, height, width)
        cell = x.new_zeros(batch, self.hidden_channels, height, width)
        outputs = []
        for t in range(steps):
            i, f, g, o = self.gates(
                torch.cat([x[:, t], hidden], dim=1)).chunk(4, dim=1)
            cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
            hidden = torch.sigmoid(o) * torch.tanh(cell)
            if self.return_sequences:
                outputs.append(hidden)
        return torch.stack(outputs, dim=1) if self.return_sequences else hidden


def _time_distributed(module, x):
    """Apply a module to each step of a (batch, T, ...) tensor."""
    batch, steps = x.shape[:2]
    return module(x.flatten(0, 1)).unflatten(0, (batch, steps))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, input_shape, input_name=None, base_filters=64):
        super().__init__()
        self.input_name = input_name
        in_channels = input_shape[-1]
        # Width-parametric: channels double each level from `base_filters`. The
        # bottleneck ConvTranspose scales ~quadratically with width, so lowering
        # base_filters (e.g. 48) shrinks the model substantially with the same
        # 4-level depth. Default 64 preserves the original UNet exactly.
        b = base_filters
        c1, c2, c3, c4, cb = b, 2 * b, 4 * b, 8 * b, 16 * b
        self.e1 = EncoderBlock(in_channels, c1)
        self.e2 = EncoderBlock(c1, c2)
        self.e3 = EncoderBlock(c2, c3)
        self.e4 = EncoderBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, cb)
        self.d1 = DecoderBlock(cb, c4, c4)
        self.d2 = DecoderBlock(c4, c3, c3)
        self.d3 = DecoderBlock(c3, c2, c2)
        self.d4 = DecoderBlock(c2, c1, c1)
        self.out_channels = c1

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.bottleneck(p4)
        d = self.d1(b, s4)
        d = self.d2(d, s3)
        d = self.d3(d, s2)
        d = self.d4(d, s1)
        return d.permute(0, 2, 3, 1)


class PixelMLP(nn.Module):
    """Per-pixel (pointwise) MLP over a channels-last spatial input.

    Applies the same small MLP independently at every pixel:
    ``(B, H, W, C_in) -> (B, H, W, out_channels)`` with no spatial mixing --
    ``nn.Linear`` acts on the last axis, so this is equivalent to a stack of 1x1
    convolutions. The neural analogue of the tabular random-forest baseline (each
    pixel classified from its own stacked band/timestep values), exposed as a
    fusion branch so it shares the decoder head with the other baselines.
    """

    def __init__(self, input_shape, input_name=None, hidden=(128, 64),
                 out_channels=32, dropout=0.3):
        super().__init__()
        self.input_name = input_name
        dims = [input_shape[-1], *hidden, out_channels]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, H, W, C_in); Linear over the last axis == per-pixel MLP.
        return self.net(x)


class CoordFourierForFusion(nn.Module):
    """Encode a per-tile coordinate (e.g. lon/lat) into broadcast fusion features.

    Front-ends the broadcast MLP with random Fourier features (Tancik et al. 2020)
    so the network can represent high-frequency spatial structure -- raw normalized
    coordinates through a small Linear cannot. Intended for static, low-dimensional
    metadata such as `md_single`'s (md_x, md_y); not for absolute year, which does
    not generalize to unseen years and is dropped from `feature_names`.

    The per-tile vector is broadcast across the spatial grid.
    """

    def __init__(self, input_shape, input_name=None, num_freqs=16, sigma=1.0,
                 out_channels=16):
        super().__init__()
        self.input_name = input_name
        in_features = input_shape[-1]
        # Fixed random projection (seeded by the trainer's torch.manual_seed) saved
        # with the model so encoding is identical across save/load.
        self.register_buffer("freq_proj",
                             torch.randn(in_features, num_freqs) * sigma)
        feat_dim = in_features + 2 * num_freqs  # raw coords + sin/cos
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, out_channels), nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, 1, in_features)
        proj = 2 * math.pi * (x @ self.freq_proj)  # (B, 1, num_freqs)
        feats = torch.cat([x, proj.sin(), proj.cos()], dim=-1)
        h = self.net(feats)
        h = h.reshape(h.shape[0], 1, 1, self.out_channels)
        return h.expand(-1, PATCH_SIZE, PATCH_SIZE, -1)


class ConvLSTMModel(nn.Module):
    def __init__(self, input_shape, input_name=None, for_fusion=True):
        super().__init__()
        self.input_name = input_name
        self.for_fusion = for_fusion
        in_channels = input_shape[-1]
        self.lstm1 = ConvLSTM2d(in_channels, 128, 5, return_sequences=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.lstm2 = ConvLSTM2d(128, 128, 3, return_sequences=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.lstm3 = ConvLSTM2d(128, 128, 1)
        self.bn3 = nn.BatchNorm2d(128)
        if for_fusion:
            self.out_channels = 128
        else:
            self.out_conv = nn.Conv2d(128, 1, 3, padding=1)
            self.out_channels = 1

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        x = _time_distributed(self.bn1, self.lstm1(x))
        x = _time_distributed(self.bn2, self.lstm2(x))
        h = self.bn3(self.lstm3(x))
        if not self.for_fusion:
            h = torch.sigmoid(self.out_conv(h))
        return h.permute(0, 2, 3, 1)


class ConvLSTMBottleneck(nn.Module):
    def __init__(self, input_shape, input_name=None, for_fusion=True):
        super().__init__()
        self.input_name = input_name
        self.for_fusion = for_fusion
        in_channels = input_shape[-1]
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.convlstm = ConvLSTM2d(64, 128, 3)
        self.bn = nn.BatchNorm2d(128)
        self.up = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1)
        if for_fusion:
            self.out_channels = 64
        else:
            self.out_conv = nn.Conv2d(64, 1, 3, padding=1)
            self.out_channels = 1

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        x = F.relu(_time_distributed(self.conv1, x))
        x = F.relu(_time_distributed(self.conv2, x))
        h = self.bn(self.convlstm(x))
        h = F.relu(self.up(h))
        if not self.for_fusion:
            h = torch.sigmoid(self.out_conv(h))
        return h.permute(0, 2, 3, 1)


class PixelLSTM(nn.Module):
    """Per-pixel temporal LSTM over a spatio-temporal input.

    Runs an LSTM over time independently at every pixel:
    ``(B, T, H, W, C) -> (B, H, W, hidden)``, taking the final hidden state. No
    spatial mixing -- the sequence counterpart of ``PixelMLP``. Exposed as a
    fusion branch so it shares the decoder head with the other baselines.
    """

    def __init__(self, input_shape, input_name=None, hidden=32, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.input_name = input_name
        in_features = input_shape[-1]
        self.hidden = hidden
        self.lstm = nn.LSTM(in_features, hidden, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.out_channels = hidden

    def forward(self, x):
        # x: (B, T, H, W, C) -> per-pixel sequences (B*H*W, T, C)
        b, t, h, w, c = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(b * h * w, t, c)
        seq, _ = self.lstm(x)
        last = seq[:, -1]                          # (B*H*W, hidden)
        return last.reshape(b, h, w, self.hidden)


class IdentityModel(nn.Module):
    def __init__(self, input_shape, input_name=None):
        super().__init__()
        self.input_name = input_name
        self.input_shape = list(input_shape)
        self.out_channels = input_shape[-1]

    def forward(self, x):
        return x


class FusionDecoder(nn.Module):
    """Runs each branch on its named input, concatenates the channels-last
    outputs, and applies a conv head. Returns (batch, H, W) sigmoid probabilities.
    """

    def __init__(self, branch_models, head_kernel=3):
        """Receptive field of the shared head = 1 + 4 * (head_kernel - 1)."""
        super().__init__()
        if head_kernel < 1 or head_kernel % 2 == 0:
            raise ValueError(f"head_kernel must be a positive odd int, got {head_kernel}")
        self.head_kernel = head_kernel
        self.branches = nn.ModuleList(branch_models)
        in_channels = sum(m.out_channels for m in branch_models)
        k, pad = head_kernel, head_kernel // 2
        self.conv1 = nn.Conv2d(in_channels, 128, k, padding=pad)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, k, padding=pad)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, k, padding=pad)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, k, padding=pad)
        self.out_conv = nn.Conv2d(16, 1, 1)

    @property
    def receptive_field(self):
        """Effective receptive field of the head, in pixels."""
        return 1 + 4 * (self.head_kernel - 1)

    def forward(self, inputs):
        feats = [branch(inputs[branch.input_name]).permute(0, 3, 1, 2)
                 for branch in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        # Head runs in float32 even under autocast (Keras dtype="float32" layer)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.sigmoid(self.out_conv(x.float())).squeeze(1)


class SimpleReadout(nn.Module):
    """Bare per-pixel linear readout over concatenated branch outputs.

    Unlike FusionDecoder, this has no hidden head and no spatial mixing: it
    concatenates the channels-last branch outputs and applies a single 1x1 conv
    to the class logits. All model capacity is meant to live in the branch
    encoder(s) -- e.g. a single all-bands PixelMLP -- so the whole model is a
    naive per-pixel MLP with a linear output layer, and this decoder adds nothing
    but the readout (and the sigmoid). Returns (B, H, W) sigmoid probabilities.
    """

    def __init__(self, branch_models):
        super().__init__()
        self.branches = nn.ModuleList(branch_models)
        in_channels = sum(m.out_channels for m in branch_models)
        self.out_conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, inputs):
        feats = [branch(inputs[branch.input_name]).permute(0, 3, 1, 2)
                 for branch in self.branches]
        x = torch.cat(feats, dim=1)
        # Match FusionDecoder: run the float32 readout even under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.sigmoid(self.out_conv(x.float())).squeeze(1)


class TransformerLayer(nn.Module):
    """Pre-norm transformer encoder layer (self-attention + MLP)."""

    def __init__(self, dim, num_heads, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class VanillaViT(nn.Module):
    """Textbook Vision Transformer segmentation baseline (no domain structure).

    Deliberately the naive-practitioner reference point for the architecture
    comparison: every branch must be a spatial ``identity`` branch (rank-3
    ``[H, W, C]`` input_shape; time is folded into the channel axis via
    ``stack_timesteps: false``), so there are NO per-modality encoders.

    The stacked channel image is patch-embedded with a single strided conv, a
    learned position embedding is added, ``depth`` standard (joint) self-
    attention layers mix all patches, and each token is linearly decoded back to
    its ``patch_size x patch_size`` output block (a transposed patch embed).
    There is no convolutional segmentation head -- that absence is the point of
    the vanilla baseline, and a known confound vs the other decoders (whose
    shared conv head supplies most of their fine-scale spatial context) when the
    result is reported. Returns (batch, H, W) sigmoid probabilities.
    """

    def __init__(self, branch_models, embed_dim=128, patch_size=8, depth=4,
                 num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.branches = nn.ModuleList(branch_models)

        shapes = set()
        for branch in branch_models:
            shape = getattr(branch, "input_shape", None)
            if shape is None or len(shape) != 3:
                raise ValueError(
                    f"decoder_vit expects spatial identity branches only; branch "
                    f"'{branch.input_name}' is a {type(branch).__name__} with "
                    f"input_shape {shape}")
            shapes.add(tuple(shape[:2]))
        if len(shapes) != 1:
            raise ValueError(f"All spatial inputs must share H, W; got {sorted(shapes)}")
        height, width = shapes.pop()
        if height % patch_size or width % patch_size:
            raise ValueError(f"patch_size {patch_size} must divide H, W "
                             f"({height}, {width})")
        self.grid = (height // patch_size, width // patch_size)
        num_patches = self.grid[0] * self.grid[1]

        in_channels = sum(m.out_channels for m in branch_models)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size,
                                     stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(num_patches, embed_dim))
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)])
        self.decode_norm = nn.LayerNorm(embed_dim)
        self.decode = nn.Linear(embed_dim, patch_size * patch_size)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, inputs):
        x = torch.cat([branch(inputs[branch.input_name]).permute(0, 3, 1, 2)
                       for branch in self.branches], dim=1)

        # Patchify -> tokens, add position embedding, joint self-attention.
        batch = x.shape[0]
        x = self.patch_embed(x)                       # (B, D, gh, gw)
        x = x.flatten(2).permute(0, 2, 1) + self.pos  # (B, N, D)
        for layer in self.layers:
            x = layer(x)

        # Linear patch-decode: each token -> its p x p block, reassembled into a
        # full-resolution map (a transposed patch embed).
        gh, gw = self.grid
        p = self.patch_size
        x = self.decode(self.decode_norm(x))          # (B, N, p*p)
        x = x.reshape(batch, gh, gw, p, p)
        x = x.permute(0, 1, 3, 2, 4).reshape(batch, gh * p, gw * p)
        # Head output runs in float32 even under autocast (matches other decoders).
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.sigmoid(x.float())


# ---------------------------------------------------------------------------
# Factories (looked up dynamically by trainer.build_model / build_decoder)
# ---------------------------------------------------------------------------

def get_unet(input_shape, input_name=None, base_filters=64):
    return UNet(input_shape, input_name, base_filters=base_filters)


def get_coord_fourier(input_shape, input_name=None):
    return CoordFourierForFusion(input_shape, input_name)


def get_pixel_mlp(input_shape, input_name=None, hidden=(128, 64), out_channels=32,
                  dropout=0.3):
    return PixelMLP(input_shape, input_name, hidden=hidden,
                    out_channels=out_channels, dropout=dropout)


def get_pixel_lstm(input_shape, input_name=None, hidden=32):
    return PixelLSTM(input_shape, input_name, hidden=hidden)

def get_convlstm(input_shape, input_name=None, for_fusion=True):
    return ConvLSTMModel(input_shape, input_name, for_fusion=for_fusion)


def get_convlstm_bottleneck(input_shape, input_name=None, for_fusion=True):
    return ConvLSTMBottleneck(input_shape, input_name, for_fusion=for_fusion)



def get_identity(input_shape, input_name=None):
    return IdentityModel(input_shape, input_name)


def decoder_fusion(branch_models, **kwargs):
    """Conv-head fusion of the branches; kwargs from config['decoder_config']."""
    return FusionDecoder(branch_models, **kwargs)


def decoder_simple(branch_models):
    """Bare 1x1 linear readout over the branches (see SimpleReadout)."""
    return SimpleReadout(branch_models)


def decoder_vit(branch_models, **kwargs):
    """Vanilla Vision Transformer baseline; kwargs from config['decoder_config']."""
    return VanillaViT(branch_models, **kwargs)


# --- Factored two-scale model (src/aic_risk_modeling/train/factored.py) --------
# Function-local imports: factored.py imports TransformerLayer from this module.

def get_pixel_temporal(input_shape, input_name=None, **kwargs):
    """Per-pixel temporal transformer; full resolution, no patch tokenization."""
    from .factored import PixelTemporalEncoder
    return PixelTemporalEncoder(input_shape, input_name=input_name, **kwargs)


def get_coarse_temporal(input_shape, input_name=None, **kwargs):
    """Temporal transformer on a pooled grid, for natively-coarse (>=4 km) bands."""
    from .factored import CoarseTemporalEncoder
    return CoarseTemporalEncoder(input_shape, input_name=input_name, **kwargs)


def decoder_factored(branch_models, **kwargs):
    """logit = gamma(t) + m(x,t) + s(x,t) + c(x,t); kwargs from config['decoder_config']."""
    from .factored import FactoredFireModel
    return FactoredFireModel(branch_models, **kwargs)
