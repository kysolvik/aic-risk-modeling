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
    def __init__(self, input_shape, input_name=None, for_fusion=True):
        super().__init__()
        self.input_name = input_name
        self.for_fusion = for_fusion
        in_channels = input_shape[-1]
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)
        self.d1 = DecoderBlock(1024, 512, 512)
        self.d2 = DecoderBlock(512, 256, 256)
        self.d3 = DecoderBlock(256, 128, 128)
        self.d4 = DecoderBlock(128, 64, 64)
        if for_fusion:
            # Expose the 64-channel decoder feature map as fusion features
            # instead of collapsing to a single channel.
            self.out_channels = 64
        else:
            self.out_conv = nn.Conv2d(64, 1, 1)
            self.out_channels = 1

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
        if not self.for_fusion:
            d = F.relu(self.out_conv(d))
        return d.permute(0, 2, 3, 1)


class UNetLite(nn.Module):
    def __init__(self, input_shape, input_name=None, for_fusion=True):
        super().__init__()
        self.input_name = input_name
        self.for_fusion = for_fusion
        in_channels = input_shape[-1]
        self.e1 = EncoderBlock(in_channels, 16)
        self.e2 = EncoderBlock(16, 32)
        self.bottleneck = ConvBlock(32, 64)
        self.d1 = DecoderBlock(64, 32, 32)
        self.d2 = DecoderBlock(32, 16, 16)
        if for_fusion:
            # Expose the 16-channel decoder feature map as fusion features
            # instead of collapsing to a single channel.
            self.out_channels = 16
        else:
            self.out_conv = nn.Conv2d(16, 1, 1)
            self.out_channels = 1

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        b = self.bottleneck(p2)
        d = self.d1(b, s2)
        d = self.d2(d, s1)
        if not self.for_fusion:
            d = F.relu(self.out_conv(d))
        return d.permute(0, 2, 3, 1)


class MLP(nn.Module):
    def __init__(self, input_shape, input_name=None):
        super().__init__()
        self.input_name = input_name
        self.net = nn.Sequential(
            nn.Linear(input_shape[-1], 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
        self.out_channels = 1

    def forward(self, x):
        return self.net(x)


class MLPForFusion(nn.Module):
    def __init__(self, input_shape, input_name=None):
        super().__init__()
        self.input_name = input_name
        self.net = nn.Sequential(
            nn.Linear(input_shape[-1], 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.out_channels = 16

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.shape[0], 1, 1, self.out_channels)
        return x.expand(-1, PATCH_SIZE, PATCH_SIZE, -1)


class CoordFourierForFusion(nn.Module):
    """Encode a per-tile coordinate (e.g. lon/lat) into broadcast fusion features.

    Front-ends the broadcast MLP with random Fourier features (Tancik et al. 2020)
    so the network can represent high-frequency spatial structure -- raw normalized
    coordinates through a small Linear cannot. Intended for static, low-dimensional
    metadata such as `md_single`'s (md_x, md_y); not for absolute year, which does
    not generalize to unseen years and is dropped from `feature_names`.

    Like `MLPForFusion`, the per-tile vector is broadcast across the spatial grid.
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


class MultiScaleMLPHead(nn.Module):
    def __init__(self, input_shape, input_name=None, hidden=128):
        super().__init__()
        self.input_name = input_name
        in_features = input_shape[-1]
        self.dense1 = nn.Linear(in_features, hidden)
        self.dense2 = nn.Linear(in_features, hidden)
        self.dense3 = nn.Linear(in_features, hidden)
        self.norm = nn.LayerNorm(3 * hidden)
        self.fuse = nn.Linear(3 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.out_channels = 1

    def _pooled_scale(self, x, dense, factor):
        x = F.avg_pool2d(x.permute(0, 3, 1, 2), factor).permute(0, 2, 3, 1)
        x = F.gelu(dense(x)).permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=factor, mode="bilinear")
        return x.permute(0, 2, 3, 1)

    def forward(self, x):
        s1 = F.gelu(self.dense1(x))
        s2 = self._pooled_scale(x, self.dense2, 2)
        s3 = self._pooled_scale(x, self.dense3, 4)
        fused = self.norm(torch.cat([s1, s2, s3], dim=-1))
        fused = F.gelu(self.fuse(fused))
        return torch.sigmoid(self.out(fused))


class SimpleConvLSTM(nn.Module):
    def __init__(self, input_shape, input_name=None):
        super().__init__()
        self.input_name = input_name
        in_channels = input_shape[-1]
        self.conv = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.convlstm = ConvLSTM2d(32, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.out_conv = nn.Conv2d(64, 1, 3, padding=1)
        self.out_channels = 1

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        x = F.relu(_time_distributed(self.conv, x))
        h = self.bn(self.convlstm(x))
        return torch.sigmoid(self.out_conv(h)).permute(0, 2, 3, 1)


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


class LSTMModel(nn.Module):
    def __init__(self, input_shape, input_name=None):
        super().__init__()
        self.input_name = input_name
        self.lstm1 = nn.LSTM(input_shape[-1], 32, batch_first=True)
        self.lstm2 = nn.LSTM(32, 32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.out_channels = 32

    def forward(self, x):
        seq, _ = self.lstm1(x)
        seq, _ = self.lstm2(self.dropout(seq))
        h = self.dropout(seq[:, -1])
        h = h.reshape(h.shape[0], 1, 1, self.out_channels)
        return h.expand(-1, PATCH_SIZE, PATCH_SIZE, -1)


class TransformerModel(nn.Module):
    def __init__(self, input_shape, input_name=None, embed_dim=32,
                 num_heads=4, ff_dim=64, dropout=0.1):
        super().__init__()
        self.input_name = input_name
        seq_len, in_features = input_shape
        self.proj = nn.Linear(in_features, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.register_buffer("positions", torch.arange(seq_len),
                             persistent=False)
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, 16)
        self.out_channels = 16

    def forward(self, x):
        x = self.proj(x) + self.pos_embedding(self.positions)
        # Attention block
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.dropout(y)
        # Feed-forward block
        y = self.ln2(x)
        y = self.ff2(self.dropout(F.relu(self.ff1(y))))
        x = x + y
        # Pool over time, project, and broadcast across the spatial grid
        h = self.dropout(x.mean(dim=1))
        h = F.relu(self.out_proj(h))
        h = h.reshape(h.shape[0], 1, 1, self.out_channels)
        return h.expand(-1, PATCH_SIZE, PATCH_SIZE, -1)


class IdentityModel(nn.Module):
    def __init__(self, input_shape, input_name=None):
        super().__init__()
        self.input_name = input_name
        self.input_shape = list(input_shape)
        self.out_channels = input_shape[-1]

    def forward(self, x):
        return x


class ProjectionModel(nn.Module):
    """Linear (1x1) projection of a channels-last input to `out_channels`.

    Adjust width of passthrough branches (e.g. the 64-channel AlphaEarth
    embedding) so a single branch doesn't dominate the fusion head's channel
    budget. Intended for rank-3 [H, W, C] inputs, which route to the spatial
    branches by rank exactly like IdentityModel.
    """

    def __init__(self, input_shape, input_name=None, out_channels=16):
        super().__init__()
        self.input_name = input_name
        self.input_shape = list(input_shape)
        self.out_channels = out_channels
        self.proj = nn.Linear(input_shape[-1], out_channels)

    def forward(self, x):
        return self.proj(x)


class _ScaleGain(nn.Module):
    """Per-source learnable scalar gain (magnitude rescale, distribution kept)."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return x * self.gain


class BranchNorm(nn.Module):
    """Per-source normalization applied before a fusion concat.

    Stops one loud or wide branch from dominating the concat and drowning out a
    quieter branch (measure the imbalance with ``scripts/probe_branch_scales.py``).
    Operates on a list of channels-first ``(B, C, H, W)`` source tensors, in the
    order they enter the concat, and returns the normalized list.

    Modes:
      - ``None`` (default): disabled. ``forward`` returns the list unchanged and
        the module holds no parameters.
      - ``"groupnorm"``: per-source ``GroupNorm(1, C)`` + affine (per-sample
        LayerNorm over ``(C, H, W)``). Robust to any channel count and to the
        spatially-constant broadcast branches (16-ch -> nonzero cross-channel
        variance). CAVEAT: GroupNorm over the spatial dims zeroes a
        spatially-constant *single-channel* source (variance 0), so any source
        with ``C == 1`` auto-falls back to a scale gain instead.
      - ``"scale"``: per-source learnable scalar gain (``x * g``, ``g`` init 1.0).
        Fixes only loudness and preserves each branch's distribution, so it is
        safe for the raw-standardized identity branch.

    ``exclude`` is a set of source labels (branch ``input_name``s, plus the
    reserved ``"<transformer>"`` for MTSViT's upsampled features) that pass
    through untouched -- for branches whose absolute level is a real predictor.
    """

    def __init__(self, channels, labels, mode=None, exclude=None):
        super().__init__()
        self.mode = mode
        if mode is None:
            self.norms = None
            return
        exclude = set(exclude or ())
        norms = []
        for c, label in zip(channels, labels):
            if label in exclude:
                norms.append(nn.Identity())
            elif mode == "scale" or (mode == "groupnorm" and c == 1):
                norms.append(_ScaleGain())
            elif mode == "groupnorm":
                norms.append(nn.GroupNorm(1, c))
            else:
                raise ValueError(
                    f"unknown branch_norm mode {mode!r} "
                    "(use 'groupnorm', 'scale', or null)")
        self.norms = nn.ModuleList(norms)

    def forward(self, feats):
        if self.norms is None:
            return feats
        return [norm(f) for norm, f in zip(self.norms, feats)]


class FusionDecoder(nn.Module):
    """Runs each branch on its named input, concatenates the channels-last
    outputs, and applies a conv head.

    With num_classes == 1 (binary) returns (batch, H, W) sigmoid probabilities;
    with num_classes > 1 returns (batch, H, W, num_classes) softmax
    probabilities.
    """

    def __init__(self, branch_models, num_classes=1, branch_norm=None,
                 branch_norm_exclude=None):
        super().__init__()
        self.num_classes = num_classes
        self.branches = nn.ModuleList(branch_models)
        self.branch_norm = BranchNorm(
            [m.out_channels for m in branch_models],
            [m.input_name for m in branch_models],
            mode=branch_norm, exclude=branch_norm_exclude)
        in_channels = sum(m.out_channels for m in branch_models)
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.out_conv = nn.Conv2d(16, num_classes, 1)

    def forward(self, inputs):
        # Channels-first per branch, normalize, then concat (equivalent to the
        # channels-last concat + permute when branch_norm is disabled).
        feats = [branch(inputs[branch.input_name]).permute(0, 3, 1, 2)
                 for branch in self.branches]
        x = torch.cat(self.branch_norm(feats), dim=1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        # Head runs in float32 even under autocast (Keras dtype="float32" layer)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.out_conv(x.float())
            if self.num_classes == 1:
                return torch.sigmoid(logits).squeeze(1)
            return torch.softmax(logits, dim=1).permute(0, 2, 3, 1)


class FiLM(nn.Module):
    """Feature-wise linear modulation (Perez et al. 2018).

    Produces a per-channel scale/shift from a conditioning vector and applies it
    to a (batch, C, H, W) feature map (spatially uniform). The projection is
    zero-initialized so the layer starts as the identity transform
    (gamma = beta = 0), giving a stable warmup.
    """

    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * num_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class FiLMFusion(nn.Module):
    """Climate-conditioned spatial fusion decoder.

    A lightweight alternative to `MTSViTFusion`: spatial-feature branches are
    concatenated and passed through a convolutional head whose features are
    FiLM-modulated by an encoding of the temporal climate-index branches (e.g.
    monthly oceanic indices). The climate state thus globally reweights spatial
    features rather than being fused as extra channels.

    Because ENSO teleconnections are spatially uneven, the conditioning is a
    joint function of climate and tile location: a per-tile coordinate (named by
    `location_input`) is encoded with random Fourier features and *gates* the
    climate encoding before it generates the per-channel gamma/beta, so the same
    climate state modulates different tiles differently. `location_input` is
    optional; without it the decoder reduces to pure-climate FiLM.

    Branch routing: the branch named `location_input` is the location modulator;
    identity branches with (T, F) inputs are the climate indices; every other
    branch provides a (batch, H, W, C) spatial feature map (so spatio-temporal
    inputs must use a temporal-reducing branch model such as `convlstm`, not a
    raw `identity`).

    With num_classes == 1 (binary) returns (batch, H, W) sigmoid probabilities;
    with num_classes > 1 returns (batch, H, W, num_classes) softmax probabilities.
    """

    def __init__(self, branch_models, num_classes=1, cond_dim=128,
                 location_input=None, num_freqs=16, sigma=1.0,
                 branch_norm=None, branch_norm_exclude=None):
        super().__init__()
        self.num_classes = num_classes
        self.location_input = location_input

        spatial_branches = []
        context_branches = []
        location_branch = None
        for branch in branch_models:
            if location_input is not None and branch.input_name == location_input:
                location_branch = branch
                continue
            shape = getattr(branch, "input_shape", None)
            if shape is not None and len(shape) == 2:
                context_branches.append(branch)
            else:
                spatial_branches.append(branch)
        if location_input is not None and location_branch is None:
            raise ValueError(
                f"location_input '{location_input}' matches no input branch")
        if not context_branches:
            raise ValueError(
                "decoder_film needs at least one climate-index input: an "
                "identity branch with shape [T, F]")
        if not spatial_branches:
            raise ValueError("decoder_film needs at least one spatial branch")
        self.spatial_branches = nn.ModuleList(spatial_branches)
        self.context_names = [b.input_name for b in context_branches]
        self.branch_norm = BranchNorm(
            [b.out_channels for b in spatial_branches],
            [b.input_name for b in spatial_branches],
            mode=branch_norm, exclude=branch_norm_exclude)

        # Climate encoder: flatten each (T, F) index series and project.
        cond_in = sum(s[0] * s[1] for s in
                      (b.input_shape for b in context_branches))
        self.conditioner = nn.Sequential(
            nn.Linear(cond_in, cond_dim), nn.ReLU(),
            nn.Linear(cond_dim, cond_dim), nn.ReLU(),
        )

        # Location encoder + gate: tile coordinate -> random Fourier features ->
        # per-channel gate on the climate code. Zero-initialized gate so the
        # model starts as pure-climate FiLM and learns the gating on top.
        if location_input is not None:
            in_features = location_branch.input_shape[-1]
            self.register_buffer("freq_proj",
                                 torch.randn(in_features, num_freqs) * sigma)
            self.loc_encoder = nn.Sequential(
                nn.Linear(in_features + 2 * num_freqs, cond_dim), nn.ReLU(),
                nn.Linear(cond_dim, cond_dim), nn.ReLU(),
            )
            self.loc_gate = nn.Linear(cond_dim, 2 * cond_dim)
            nn.init.zeros_(self.loc_gate.weight)
            nn.init.zeros_(self.loc_gate.bias)

        in_channels = sum(b.out_channels for b in spatial_branches)
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.film1 = FiLM(cond_dim, 128)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.film2 = FiLM(cond_dim, 64)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.film3 = FiLM(cond_dim, 32)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.film4 = FiLM(cond_dim, 16)
        self.out_conv = nn.Conv2d(16, num_classes, 1)

    def _condition(self, inputs):
        flat = [inputs[name].flatten(1) for name in self.context_names]
        cond = self.conditioner(torch.cat(flat, dim=1))
        if self.location_input is not None:
            coord = inputs[self.location_input].flatten(1)  # (B, in_features)
            proj = 2 * math.pi * (coord @ self.freq_proj)
            feats = torch.cat([coord, proj.sin(), proj.cos()], dim=-1)
            gamma, beta = self.loc_gate(self.loc_encoder(feats)).chunk(2, dim=1)
            cond = cond * (1 + gamma) + beta  # location gates climate
        return cond

    def forward(self, inputs):
        cond = self._condition(inputs)
        feats = [branch(inputs[branch.input_name]).permute(0, 3, 1, 2)
                 for branch in self.spatial_branches]
        x = torch.cat(self.branch_norm(feats), dim=1)
        x = F.relu(self.film1(self.bn1(self.conv1(x)), cond))
        x = F.relu(self.film2(self.bn2(self.conv2(x)), cond))
        x = F.relu(self.film3(self.bn3(self.conv3(x)), cond))
        x = F.relu(self.film4(self.conv4(x), cond))
        # Head output runs in float32 even under autocast
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.out_conv(x.float())
            if self.num_classes == 1:
                return torch.sigmoid(logits).squeeze(1)
            return torch.softmax(logits, dim=1).permute(0, 2, 3, 1)


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


class CrossAttnTemporalLayer(nn.Module):
    """Temporal transformer layer with cross-attention to shared context tokens.

    Operates on per-patch time sequences (batch * patches, T, dim) and lets each
    token additionally attend to context tokens (batch, T_ctx, dim) that are
    shared across patch locations (e.g. monthly climate indices).
    """

    def __init__(self, dim, num_heads, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                               batch_first=True)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                                batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, context):
        # x: (B*N, T, D); context: (B, T_ctx, D)
        y = self.ln1(x)
        x = x + self.self_attn(y, y, y, need_weights=False)[0]
        # Every patch location attends to the same context, so fold the patch
        # axis into the query token axis instead of expanding key/values.
        bn, t, d = x.shape
        b = context.shape[0]
        q = self.ln_q(x).reshape(b, (bn // b) * t, d)
        kv = self.ln_kv(context)
        attended = self.cross_attn(q, kv, kv, need_weights=False)[0]
        x = x + attended.reshape(bn, t, d)
        return x + self.mlp(self.ln2(x))


class MTSViTFusion(nn.Module):
    """Multi-step temporo-spatial fusion (TSViT/MTSViT-inspired).

    Stage 1 (temporal): each spatio-temporal modality is patch-embedded per
    frame; a temporal transformer encoder (shared across modalities) runs
    self-attention over each patch location's time series with cross-attention
    to temporal context tokens (e.g. monthly oceanic indices), and a per-
    modality cls token summarizes the series.
    Stage 2 (spatial): modality tokens (optionally plus the patch-embedded
    single-timestep spatial branches, when `spatial_in_encoder`) are fused per
    patch location and mixed by a spatial transformer encoder.
    Stage 3 (decoder): tokens are progressively upsampled to full resolution,
    concatenated with the (full-resolution) single-timestep spatial branches,
    and passed through a convolutional segmentation head. With num_classes == 1
    (binary) returns (batch, H, W) sigmoid probabilities; with num_classes > 1
    returns (batch, H, W, num_classes) softmax probabilities.

    Branch routing: identity branches with (T, H, W, C) inputs are the
    spatio-temporal modalities; identity branches with (T, F) inputs are the
    temporal context; all other branches provide spatial features to the
    segmentation head (at full resolution) and, when `spatial_in_encoder` is set,
    also to the spatial encoder (patch-embedded).
    """

    def __init__(self, branch_models, num_classes=1, embed_dim=128,
                 patch_size=8, temporal_depth=2, spatial_depth=2, num_heads=4,
                 mlp_ratio=2, dropout=0.1, spatial_in_encoder=False,
                 branch_norm=None, branch_norm_exclude=None,
                 transformer_out_channels=16, climate_film=False,
                 film_location=None, film_cond_dim=128, film_num_freqs=16,
                 film_sigma=1.0, film_location_features=2):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.spatial_in_encoder = spatial_in_encoder

        spatiotemporal_branches = []
        temporal_branches = []
        spatial_branches = []
        for branch in branch_models:
            # Only identity branches expose `input_shape` and are routed by rank:
            # (T, H, W, C) -> spatio-temporal modality, (T, F) -> temporal
            # context, (H, W, C) -> spatial features. Every other branch (e.g. a
            # unet_lite CNN) is a spatial feature branch whose output is
            # concatenated into the segmentation head.
            shape = getattr(branch, "input_shape", None)
            if shape is None:
                spatial_branches.append(branch)
            elif len(shape) == 4:
                spatiotemporal_branches.append(branch)
            elif len(shape) == 2:
                temporal_branches.append(branch)
            elif len(shape) == 3:
                spatial_branches.append(branch)
            else:
                raise ValueError(f'Cannot determine type of branch model {branch}')
        if not spatiotemporal_branches:
            raise ValueError(
                "decoder_mtsvit needs at least one spatio-temporal input: an "
                "identity branch with timesteps and shape [H, W]")
        self.spatial_branches = nn.ModuleList(spatial_branches)
        self.temporal_names = [b.input_name for b in spatiotemporal_branches]
        self.context_names = [b.input_name for b in temporal_branches]

        # Per-modality patch embedding, temporal position embedding, cls token
        self.patch_embeds = nn.ModuleDict()
        self.temporal_pos = nn.ParameterDict()
        self.cls_tokens = nn.ParameterDict()
        height, width = spatiotemporal_branches[0].input_shape[1:3]
        if height % patch_size or width % patch_size:
            raise ValueError(f"patch_size {patch_size} must divide H, W "
                             f"({height}, {width})")
        self.grid = (height // patch_size, width // patch_size)
        num_patches = self.grid[0] * self.grid[1]
        for branch in spatiotemporal_branches:
            steps, h, w, channels = branch.input_shape
            if (h, w) != (height, width):
                raise ValueError("All spatio-temporal inputs must share H, W")
            name = branch.input_name
            self.patch_embeds[name] = nn.Conv2d(channels, embed_dim,
                                                patch_size, stride=patch_size)
            self.temporal_pos[name] = nn.Parameter(
                torch.zeros(steps, embed_dim))
            self.cls_tokens[name] = nn.Parameter(
                torch.zeros(1, 1, 1, embed_dim))

        # Temporal context (climate indices): project + position embedding
        self.context_projs = nn.ModuleDict()
        self.context_pos = nn.ParameterDict()
        for branch in temporal_branches:
            steps, features = branch.input_shape
            name = branch.input_name
            self.context_projs[name] = nn.Linear(features, embed_dim)
            self.context_pos[name] = nn.Parameter(
                torch.zeros(steps, embed_dim))

        # Stage 1: temporal encoder (cross-attends to context when present)
        if temporal_branches:
            self.temporal_layers = nn.ModuleList([
                CrossAttnTemporalLayer(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(temporal_depth)])
        else:
            self.temporal_layers = nn.ModuleList([
                TransformerLayer(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(temporal_depth)])

        # Stage 2: fuse the temporal modality summaries per patch, then spatial
        # encoder. When `spatial_in_encoder`, single-timestep spatial branches
        # are also patch-embedded down to the token grid so they join cross-patch
        # attention (they feed the head at full resolution either way).
        num_token_sources = len(spatiotemporal_branches)
        if spatial_in_encoder:
            self.spatial_patch_embeds = nn.ModuleDict()
            for branch in spatial_branches:
                self.spatial_patch_embeds[branch.input_name] = nn.Conv2d(
                    branch.out_channels, embed_dim, patch_size, stride=patch_size)
            num_token_sources += len(spatial_branches)
        self.modality_fuse = nn.Linear(num_token_sources * embed_dim, embed_dim)
        self.spatial_pos = nn.Parameter(torch.zeros(num_patches, embed_dim))
        self.spatial_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(spatial_depth)])

        # Stage 3: upsample tokens back to full resolution.
        # `transformer_out_channels` floors the halving, setting the width the
        # transformer contributes to the fusion concat (default 16). This
        # controls how much of the head's channel budget the temporal/
        # weather/climate pathway gets vs the full-res spatial branches.
        upsample = []
        in_ch = embed_dim
        scale = patch_size
        while scale > 1:
            out_ch = max(in_ch // 2, transformer_out_channels)
            upsample.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            upsample.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            upsample.append(nn.ReLU())
            in_ch = out_ch
            scale //= 2
        self.upsample = nn.Sequential(*upsample)

        # Segmentation head over upsampled tokens + spatial branch features.
        # The concat order is [upsampled transformer features] + spatial branches;
        # branch_norm matches that order (label "<transformer>" for the former).
        self.branch_norm = BranchNorm(
            [in_ch] + [b.out_channels for b in spatial_branches],
            ["<transformer>"] + [b.input_name for b in spatial_branches],
            mode=branch_norm, exclude=branch_norm_exclude)
        head_in = in_ch + sum(b.out_channels for b in spatial_branches)
        self.head = nn.Sequential(
            nn.Conv2d(head_in, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
        )
        self.out_conv = nn.Conv2d(16, num_classes, 1)

        # Optional climate x location FiLM over the head (the MTSViT port of
        # the standalone FiLMFusion decoder's mechanism. The rank-2 context
        # branches (monthly climate indices) are encoded to a conditioning
        # vector, optionally gated per tile by the `film_location` coordinates
        # (random Fourier features, zero-init gate -- ENSO teleconnections are
        # spatially uneven), and zero-init FiLM layers modulate each head
        # stage. Identity at init, and `head.*` parameter names are unchanged.
        self.climate_film = climate_film
        self.film_location = film_location
        if climate_film:
            if not temporal_branches:
                raise ValueError(
                    "climate_film needs at least one temporal-context input: "
                    "an identity branch with shape [T, F]")
            cond_in = sum(b.input_shape[0] * b.input_shape[1]
                          for b in temporal_branches)
            self.film_conditioner = nn.Sequential(
                nn.Linear(cond_in, film_cond_dim), nn.ReLU(),
                nn.Linear(film_cond_dim, film_cond_dim), nn.ReLU(),
            )
            if film_location is not None:
                matches = [b for b in branch_models
                           if b.input_name == film_location]
                if not matches:
                    raise ValueError(f"film_location '{film_location}' "
                                     f"matches no input branch")
                # Non-identity location branches (e.g. coord_fourier) don't
                # expose input_shape; film_location_features covers them
                # (md_single's (md_x, md_y) -> 2).
                loc_shape = getattr(matches[0], "input_shape", None)
                in_features = (loc_shape[-1] if loc_shape is not None
                               else film_location_features)
                self.register_buffer(
                    "film_freq_proj",
                    torch.randn(in_features, film_num_freqs) * film_sigma)
                self.film_loc_encoder = nn.Sequential(
                    nn.Linear(in_features + 2 * film_num_freqs,
                              film_cond_dim), nn.ReLU(),
                    nn.Linear(film_cond_dim, film_cond_dim), nn.ReLU(),
                )
                self.film_loc_gate = nn.Linear(film_cond_dim,
                                               2 * film_cond_dim)
                nn.init.zeros_(self.film_loc_gate.weight)
                nn.init.zeros_(self.film_loc_gate.bias)
            # One FiLM per head stage (the head's conv widths are fixed).
            self.films = nn.ModuleList(
                [FiLM(film_cond_dim, c) for c in (128, 64, 32, 16)])

        for pos in list(self.temporal_pos.values()) + list(self.context_pos.values()):
            nn.init.trunc_normal_(pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        for cls in self.cls_tokens.values():
            nn.init.trunc_normal_(cls, std=0.02)

    def _film_condition(self, inputs):
        """Climate conditioning vector, optionally gated by tile location
        (mirrors FiLMFusion._condition)."""
        flat = [inputs[name].flatten(1) for name in self.context_names]
        cond = self.film_conditioner(torch.cat(flat, dim=1))
        if self.film_location is not None:
            coord = inputs[self.film_location].flatten(1)
            proj = 2 * math.pi * (coord @ self.film_freq_proj)
            feats = torch.cat([coord, proj.sin(), proj.cos()], dim=-1)
            gamma, beta = self.film_loc_gate(
                self.film_loc_encoder(feats)).chunk(2, dim=1)
            cond = cond * (1 + gamma) + beta  # location gates climate
        return cond

    def _encode_context(self, inputs):
        if not self.context_names:
            return None
        tokens = [self.context_projs[name](inputs[name]) + self.context_pos[name]
                  for name in self.context_names]
        return torch.cat(tokens, dim=1)

    def _encode_temporal(self, x, name, context):
        # x: (B, T, H, W, C) -> patch tokens (B, N, T, D)
        batch, steps = x.shape[:2]
        x = _time_distributed(self.patch_embeds[name],
                              x.permute(0, 1, 4, 2, 3))
        x = x.flatten(3).permute(0, 3, 1, 2)  # (B, N, T, D)
        x = x + self.temporal_pos[name]
        cls = self.cls_tokens[name].expand(batch, x.shape[1], 1, -1)
        x = torch.cat([cls, x], dim=2).flatten(0, 1)  # (B*N, T+1, D)
        for layer in self.temporal_layers:
            x = layer(x, context) if context is not None else layer(x)
        return x[:, 0].unflatten(0, (batch, -1))  # cls token: (B, N, D)

    def forward(self, inputs):
        context = self._encode_context(inputs)

        # Stage 1: temporal encoding per modality
        tokens = [self._encode_temporal(inputs[name], name, context)
                  for name in self.temporal_names]

        # Spatial branches: full-res feature maps (channels-first), computed once
        # and reused by the (optional) spatial-encoder tokens and the decoder head.
        spatial_feats = [branch(inputs[branch.input_name]).permute(0, 3, 1, 2)
                         for branch in self.spatial_branches]

        # Optionally patch-embed each spatial map to the patch grid and add it as
        # another token source for the spatial encoder.
        if self.spatial_in_encoder:
            for branch, feat in zip(self.spatial_branches, spatial_feats):
                emb = self.spatial_patch_embeds[branch.input_name](feat)  # (B,D,gh,gw)
                tokens.append(emb.flatten(2).permute(0, 2, 1))           # (B, N, D)

        # Stage 2: fuse all token sources, spatial encoding
        x = self.modality_fuse(torch.cat(tokens, dim=-1)) + self.spatial_pos
        for layer in self.spatial_layers:
            x = layer(x)

        # Stage 3: upsample and fuse with spatial branches
        batch = x.shape[0]
        x = x.permute(0, 2, 1).reshape(batch, self.embed_dim, *self.grid)
        x = self.upsample(x)
        x = torch.cat(self.branch_norm([x] + spatial_feats), dim=1)
        if self.climate_film:
            cond = self._film_condition(inputs)
            # Head layout is [Conv, ReLU, BN] x3 + [Conv, ReLU]; apply one
            # FiLM at each stage boundary (after indices 2, 5, 8, 10).
            film_at = {2: 0, 5: 1, 8: 2, 10: 3}
            for i, module in enumerate(self.head):
                x = module(x)
                if i in film_at:
                    x = self.films[film_at[i]](x, cond)
        else:
            x = self.head(x)
        # Head output runs in float32 even under autocast
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.out_conv(x.float())
            if self.num_classes == 1:
                return torch.sigmoid(logits).squeeze(1)
            return torch.softmax(logits, dim=1).permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# Factories (looked up dynamically by trainer.build_model / build_decoder)
# ---------------------------------------------------------------------------

def get_unet(input_shape, input_name=None, for_fusion=True):
    return UNet(input_shape, input_name, for_fusion=for_fusion)


def get_unet_lite(input_shape, input_name=None, for_fusion=True):
    return UNetLite(input_shape, input_name, for_fusion=for_fusion)


def get_projection(input_shape, input_name=None, out_channels=16):
    return ProjectionModel(input_shape, input_name, out_channels=out_channels)


def get_mlp(input_shape, input_name=None):
    return MLP(input_shape, input_name)


def get_mlp_for_fusion(input_shape, input_name=None):
    return MLPForFusion(input_shape, input_name)


def get_coord_fourier(input_shape, input_name=None):
    return CoordFourierForFusion(input_shape, input_name)


def get_multi_scale_mlp_head(input_shape, input_name=None, hidden=128):
    return MultiScaleMLPHead(input_shape, input_name, hidden=hidden)


def get_simple_convlstm(input_shape, input_name=None):
    return SimpleConvLSTM(input_shape, input_name)


def get_convlstm(input_shape, input_name=None, for_fusion=True):
    return ConvLSTMModel(input_shape, input_name, for_fusion=for_fusion)


def get_convlstm_bottleneck(input_shape, input_name=None, for_fusion=True):
    return ConvLSTMBottleneck(input_shape, input_name, for_fusion=for_fusion)


def get_lstm(input_shape, input_name=None):
    return LSTMModel(input_shape, input_name)


def get_transformer(input_shape, input_name=None):
    return TransformerModel(input_shape, input_name)


def get_identity(input_shape, input_name=None):
    return IdentityModel(input_shape, input_name)


def decoder_fusion(branch_models, num_classes=1, **kwargs):
    """branch_models: list of branch modules (e.g. [lstm_branch, cnn_branch]).

    kwargs come from config['decoder_config'] (e.g. branch_norm)."""
    return FusionDecoder(branch_models, num_classes=num_classes, **kwargs)


def decoder_mtsvit(branch_models, num_classes=1, **kwargs):
    """Multi-step temporo-spatial fusion; kwargs come from config['decoder_config']."""
    return MTSViTFusion(branch_models, num_classes=num_classes, **kwargs)


def decoder_film(branch_models, num_classes=1, **kwargs):
    """Climate(×location)-conditioned spatial fusion; kwargs from config['decoder_config']."""
    return FiLMFusion(branch_models, num_classes=num_classes, **kwargs)
