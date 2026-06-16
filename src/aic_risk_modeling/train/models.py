"""Defines PyTorch models used in training.

All models take channels-last inputs, exactly as produced by the tf.data
pipeline (images are (batch, H, W, C), time series are (batch, T, H, W, C) or
(batch, T, features)), and return channels-last outputs. Layouts are permuted
to channels-first internally where torch layers require it.

Branch models expose `input_name` (the key of the input dict they consume)
and `out_channels` (feature channels of their output) so `decoder_fusion`
can route inputs and size its first convolution.
"""

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
    def __init__(self, input_shape, input_name=None):
        super().__init__()
        self.input_name = input_name
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
        return F.relu(self.out_conv(d)).permute(0, 2, 3, 1)


class UNetLite(nn.Module):
    def __init__(self, input_shape, input_name=None):
        super().__init__()
        self.input_name = input_name
        in_channels = input_shape[-1]
        self.e1 = EncoderBlock(in_channels, 16)
        self.e2 = EncoderBlock(16, 32)
        self.bottleneck = ConvBlock(32, 64)
        self.d1 = DecoderBlock(64, 32, 32)
        self.d2 = DecoderBlock(32, 16, 16)
        self.out_conv = nn.Conv2d(16, 1, 1)
        self.out_channels = 1

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        b = self.bottleneck(p2)
        d = self.d1(b, s2)
        d = self.d2(d, s1)
        return F.relu(self.out_conv(d)).permute(0, 2, 3, 1)


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


class FusionDecoder(nn.Module):
    """Runs each branch on its named input, concatenates the channels-last
    outputs, and applies a conv head. Returns (batch, H, W) probabilities.
    """

    def __init__(self, branch_models):
        super().__init__()
        self.branches = nn.ModuleList(branch_models)
        in_channels = sum(m.out_channels for m in branch_models)
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, inputs):
        feats = [branch(inputs[branch.input_name]) for branch in self.branches]
        x = torch.cat(feats, dim=-1).permute(0, 3, 1, 2)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        # Head runs in float32 even under autocast (Keras dtype="float32" layer)
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = torch.sigmoid(self.out_conv(x.float()))
        return out.squeeze(1)


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
    Stage 2 (spatial): modality tokens are fused per patch location and mixed
    by a spatial transformer encoder.
    Stage 3 (decoder): tokens are progressively upsampled to full resolution,
    concatenated with the single-timestep spatial branches, and passed through
    a convolutional segmentation head. Returns (batch, H, W) probabilities.

    Branch routing: identity branches with (T, H, W, C) inputs are the
    spatio-temporal modalities; identity branches with (T, F) inputs are the
    temporal context; all other branches provide spatial features to the head.
    """

    def __init__(self, branch_models, embed_dim=128, patch_size=8,
                 temporal_depth=2, spatial_depth=2, num_heads=4,
                 mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        spatiotemporal_branches = []
        temporal_branches = []
        spatial_branches = []
        for branch in branch_models:
            shape = getattr(branch, "input_shape", None)
            # Spatial-temporal
            if len(shape) == 4:
                spatiotemporal_branches.append(branch)
            # Temporal context only
            elif len(shape) == 2:
                temporal_branches.append(branch)
            # Spatial only
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

        # Stage 2: fuse modalities per patch location, then spatial encoder
        self.modality_fuse = nn.Linear(len(spatiotemporal_branches) * embed_dim,
                                       embed_dim)
        self.spatial_pos = nn.Parameter(torch.zeros(num_patches, embed_dim))
        self.spatial_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(spatial_depth)])

        # Stage 3: upsample tokens back to full resolution
        upsample = []
        in_ch = embed_dim
        scale = patch_size
        while scale > 1:
            out_ch = max(in_ch // 2, 16)
            upsample.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            upsample.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            upsample.append(nn.ReLU())
            in_ch = out_ch
            scale //= 2
        self.upsample = nn.Sequential(*upsample)

        # Segmentation head over upsampled tokens + spatial branch features
        head_in = in_ch + sum(b.out_channels for b in spatial_branches)
        self.head = nn.Sequential(
            nn.Conv2d(head_in, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
        )
        self.out_conv = nn.Conv2d(16, 1, 1)

        for pos in list(self.temporal_pos.values()) + list(self.context_pos.values()):
            nn.init.trunc_normal_(pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        for cls in self.cls_tokens.values():
            nn.init.trunc_normal_(cls, std=0.02)

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

        # Stage 2: fuse modalities, spatial encoding
        x = self.modality_fuse(torch.cat(tokens, dim=-1)) + self.spatial_pos
        for layer in self.spatial_layers:
            x = layer(x)

        # Stage 3: upsample and fuse with spatial branches
        batch = x.shape[0]
        x = x.permute(0, 2, 1).reshape(batch, self.embed_dim, *self.grid)
        x = self.upsample(x)
        spatial_feats = [branch(inputs[branch.input_name]).permute(0, 3, 1, 2)
                         for branch in self.spatial_branches]
        x = self.head(torch.cat([x] + spatial_feats, dim=1))
        # Head output runs in float32 even under autocast
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = torch.sigmoid(self.out_conv(x.float()))
        return out.squeeze(1)


# ---------------------------------------------------------------------------
# Factories (looked up dynamically by trainer.build_model / build_decoder)
# ---------------------------------------------------------------------------

def get_unet(input_shape, input_name=None):
    return UNet(input_shape, input_name)


def get_unet_lite(input_shape, input_name=None):
    return UNetLite(input_shape, input_name)


def get_mlp(input_shape, input_name=None):
    return MLP(input_shape, input_name)


def get_mlp_for_fusion(input_shape, input_name=None):
    return MLPForFusion(input_shape, input_name)


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


def decoder_fusion(branch_models):
    """branch_models: list of branch modules (e.g. [lstm_branch, cnn_branch])"""
    return FusionDecoder(branch_models)


def decoder_mtsvit(branch_models, **kwargs):
    """Multi-step temporo-spatial fusion; kwargs come from config['decoder_config']."""
    return MTSViTFusion(branch_models, **kwargs)
