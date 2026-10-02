"""Factored fire model: logit = gamma(t) + year_gain + m (coarse) + s (pixel) + c (local).

`c` masks its centre tap, so s and c are disjoint (ignition vs spread); gamma is a
frozen offline-fit year offset (eval/year_offset.py)."""

import json
import math

import torch
import torch.nn.functional as F
from torch import nn

from .models import TransformerLayer


def _read_json(path):
    if str(path).startswith("gs://"):
        from tensorflow.io import gfile  # noqa: PLC0415 - lazy, keeps TF off the import path
        with gfile.GFile(path, "r") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


class PixelTemporalEncoder(nn.Module):
    """Per-pixel temporal transformer: (B, T, H, W, C) -> (B, H, W, D); cost ~ H*W*T^2*D."""

    def __init__(self, input_shape, input_name=None, dim=32, depth=2, num_heads=4,
                 mlp_ratio=2, dropout=0.1):
        super().__init__()
        if len(input_shape) != 4:
            raise ValueError(f"expected (T, H, W, C) input_shape, got {input_shape}")
        steps, _, _, channels = input_shape
        self.input_name = input_name
        self.out_channels = dim
        self.proj = nn.Linear(channels, dim)
        self.temporal_pos = nn.Parameter(torch.zeros(steps, dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.layers = nn.ModuleList(
            [TransformerLayer(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch, steps, height, width, _ = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(batch * height * width, steps, -1)
        x = self.proj(x) + self.temporal_pos
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=1)
        return x.reshape(batch, height, width, self.out_channels)


class CoarseTemporalEncoder(nn.Module):
    """PixelTemporalEncoder on a `grid` x `grid` pooled input, upsampled back; for >=4 km inputs."""

    def __init__(self, input_shape, input_name=None, grid=16, dim=32, depth=2,
                 num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        if len(input_shape) != 4:
            raise ValueError(f"expected (T, H, W, C) input_shape, got {input_shape}")
        self.input_name = input_name
        self.grid = grid
        self.out_channels = dim
        steps, _, _, channels = input_shape
        self.encoder = PixelTemporalEncoder(
            [steps, grid, grid, channels], input_name=input_name, dim=dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        batch, steps, height, width, channels = x.shape
        flat = x.reshape(batch * steps, height, width, channels).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(flat, self.grid)
        pooled = pooled.permute(0, 2, 3, 1).reshape(batch, steps, self.grid, self.grid, channels)
        feats = self.encoder(pooled).permute(0, 3, 1, 2)
        up = F.interpolate(feats, size=(height, width), mode="bilinear", align_corners=False)
        return up.permute(0, 2, 3, 1)


class YearOffset(nn.Module):
    """Frozen per-year logit offset gamma(t), looked up by the raw md_year input.

    md_year is a lookup key only; its group must have `normalize: false`."""

    def __init__(self, offsets, input_name="md_year", trainable=False, strict=True):
        super().__init__()
        offsets = {int(y): float(v) for y, v in offsets.items()}
        if not offsets:
            raise ValueError("YearOffset needs a non-empty year -> offset mapping")
        self.input_name = input_name
        self.strict = strict
        self.min_year = min(offsets)
        self.max_year = max(offsets)
        table = torch.tensor([offsets.get(y, 0.0)
                              for y in range(self.min_year, self.max_year + 1)],
                             dtype=torch.float32)
        known = torch.tensor([y in offsets for y in range(self.min_year, self.max_year + 1)])
        if trainable:
            self.table = nn.Parameter(table)
        else:
            self.register_buffer("table", table)
        self.register_buffer("known", known)

    @classmethod
    def from_json(cls, path, **kw):
        doc = _read_json(path)
        return cls(doc["per_year_offset"], **kw)

    def forward(self, year):
        year = torch.as_tensor(year).reshape(year.shape[0], -1)[:, 0]
        idx = torch.round(year).long() - self.min_year
        if self.strict:
            oob = (idx < 0) | (idx >= self.table.numel())
            if bool(oob.any()):
                bad = (year[oob] if oob.any() else year).flatten()[:5].tolist()
                raise KeyError(
                    f"year(s) {bad} outside gamma table [{self.min_year}, {self.max_year}]. "
                    "Re-run scripts/analysis/fit_year_offset.py to extend it -- do NOT "
                    "silently fall back to 0, which would mean 'average year'.")
            if not bool(self.known[idx].all()):
                raise KeyError("year(s) missing from the gamma table")
        idx = idx.clamp(0, self.table.numel() - 1)
        return self.table[idx].reshape(-1, 1, 1, 1)


def build_year_offset(spec, year_group):
    """YearOffset from a decoder_config `year_offset` block (coeffs_path or offsets), or None."""
    if not spec:
        return None
    if not year_group:
        raise ValueError("year_offset given but year_group is unset")
    spec = dict(spec)
    path = spec.pop("coeffs_path", None)
    spec.pop("terms", None)
    offsets = spec.pop("offsets", None)
    kw = {k: spec.pop(k) for k in ("trainable", "strict") if k in spec}
    if spec:
        raise ValueError(f"unknown year_offset keys: {sorted(spec)}")
    if (path is None) == (offsets is None):
        raise ValueError("year_offset needs exactly one of coeffs_path / offsets")
    if path is not None:
        return YearOffset.from_json(path, input_name=year_group, **kw)
    return YearOffset(offsets, input_name=year_group, **kw)


class SpatialYearGain(nn.Module):
    """Per-chip loading g_res(location) so the year term becomes gamma * (1 + g_res).

    Zero-init (exact no-op at start) and mean-centred (batch mean in train, running mean in
    eval), so the basin-mean year amplitude stays gamma."""

    def __init__(self, input_name="md_single", loc_features=2, num_freqs=16, sigma=1.0,
                 hidden=64, momentum=0.1):
        super().__init__()
        self.input_name = input_name
        self.loc_features = loc_features
        self.momentum = momentum
        # Persistent buffer: the random projection must survive save/load.
        self.register_buffer("freq_proj", torch.randn(loc_features, num_freqs) * sigma)
        self.register_buffer("running_mean", torch.zeros(1))
        feat_dim = loc_features + 2 * num_freqs
        self.body = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, coords):
        coords = coords.reshape(coords.shape[0], -1)[:, :self.loc_features]
        proj = 2 * math.pi * (coords @ self.freq_proj)
        feats = torch.cat([coords, proj.sin(), proj.cos()], dim=-1)
        raw = self.out(self.body(feats))
        if self.training:
            batch_mean = raw.mean()
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
            centre = batch_mean
        else:
            centre = self.running_mean
        return (raw - centre).reshape(-1, 1, 1, 1)


def build_year_gain(spec, year_gain_group):
    """SpatialYearGain from a decoder_config `year_gain` block ({} = defaults), or None."""
    if spec is None:
        return None
    if not year_gain_group:
        raise ValueError("year_gain given but year_gain_group is unset")
    spec = dict(spec)
    kw = {k: spec.pop(k)
          for k in ("loc_features", "num_freqs", "sigma", "hidden", "momentum") if k in spec}
    if spec:
        raise ValueError(f"unknown year_gain keys: {sorted(spec)}")
    return SpatialYearGain(input_name=year_gain_group, **kw)


class PixelSusceptibility(nn.Module):
    """`s`: pointwise (1x1 conv) logit term on the full-resolution stack."""

    def __init__(self, in_channels, hidden=(128, 64), dropout=0.0):
        super().__init__()
        layers = []
        prev = in_channels
        for h in hidden:
            layers += [nn.Conv2d(prev, h, 1), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout2d(dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(prev, 1, 1)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.body(x))


class LocalContext(nn.Module):
    """`c`: centre-masked depthwise k x k conv + 1x1 convs; receptive field is exactly `kernel`.

    A single masked layer keeps the pixel's own features out (stacked convs would leak them).
    kernel=1 disables the term."""

    def __init__(self, in_channels, kernel=9, hidden=64, dilation=1):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError(f"kernel must be a positive odd int, got {kernel}")
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")
        self.kernel = kernel
        self.dilation = dilation
        self.in_channels = in_channels
        if kernel == 1:
            self.depthwise = None
            self.body = None
            self.out = None
            return
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel,
                                   padding=dilation * (kernel // 2), dilation=dilation,
                                   groups=in_channels, bias=False)
        mask = torch.ones(1, 1, kernel, kernel)
        mask[0, 0, kernel // 2, kernel // 2] = 0.0
        self.register_buffer("centre_mask", mask)
        self.body = nn.Sequential(nn.Conv2d(in_channels, hidden, 1), nn.ReLU())
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def receptive_field(self):
        return 1 + (self.kernel - 1) * self.dilation

    def forward(self, x):
        if self.depthwise is None:
            return x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        neighbourhood = F.conv2d(x, self.depthwise.weight * self.centre_mask,
                                 padding=self.dilation * (self.kernel // 2),
                                 dilation=self.dilation, groups=self.in_channels)
        return self.out(self.body(neighbourhood))


class CoarseIntensity(nn.Module):
    """`m`: coarse intensity term on a `grid` x `grid` lattice, upsampled; grid=None disables."""

    def __init__(self, in_channels, context_dim=0, grid=4, hidden=64):
        super().__init__()
        self.grid = grid
        self.context_dim = context_dim if grid is not None else 0
        if grid is None:
            self.body = None
            self.out = None
            return
        self.body = nn.Sequential(
            nn.Conv2d(in_channels + context_dim, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 1), nn.ReLU())
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, context=None):
        height, width = x.shape[2], x.shape[3]
        if self.grid is None:
            return x.new_zeros(x.shape[0], 1, height, width)
        pooled = F.adaptive_avg_pool2d(x, self.grid)
        if self.context_dim:
            if context is None:
                raise ValueError("CoarseIntensity was built with context but got none")
            ctx = context.reshape(context.shape[0], self.context_dim, 1, 1)
            pooled = torch.cat([pooled, ctx.expand(-1, -1, self.grid, self.grid)], dim=1)
        coarse = self.out(self.body(pooled))
        return F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)


class FactoredFireModel(nn.Module):
    """Sums the factored terms into a per-pixel probability; every branch must be routed explicitly."""

    def __init__(self, branch_models, pixel_groups=None,
                 context_groups=None, year_group=None, local_kernel=9, coarse_grid=4,
                 susceptibility_hidden=(128, 64), susceptibility_dropout=0.0,
                 local_hidden=64, local_dilation=1, coarse_hidden=64, year_offset=None,
                 year_gain_group=None, year_gain=None):
        super().__init__()
        pixel_groups = list(pixel_groups or [])
        context_groups = list(context_groups or [])

        named = {b.input_name: b for b in branch_models}
        listed = pixel_groups + context_groups + ([year_group] if year_group else [])
        dupes = {n for n in listed if listed.count(n) > 1}
        if dupes:
            raise ValueError(f"group(s) listed more than once: {sorted(dupes)}")
        unrouted = sorted(set(named) - set(listed))
        if unrouted:
            raise ValueError(
                f"branch group(s) {unrouted} are not routed. List each in exactly one of "
                "pixel_groups / context_groups / year_group in decoder_config.")
        missing = sorted(set(pixel_groups + context_groups) - set(named))
        if missing:
            raise ValueError(f"decoder_config names group(s) with no branch model: {missing}")

        self.pixel_branches = nn.ModuleList([named[n] for n in pixel_groups])
        self.context_branches = nn.ModuleList([named[n] for n in context_groups])
        self.year_group = year_group

        pixel_channels = sum(b.out_channels for b in self.pixel_branches)
        if pixel_channels == 0:
            raise ValueError("pixel_groups must contribute at least one channel")
        context_dim = 0
        for b in self.context_branches:
            shape = getattr(b, "input_shape", None)
            if shape is None:
                raise ValueError(
                    f"context branch {b.input_name!r} must expose input_shape "
                    "(use model_type 'identity')")
            context_dim += int(math.prod(shape))

        self.susceptibility = PixelSusceptibility(
            pixel_channels, hidden=tuple(susceptibility_hidden), dropout=susceptibility_dropout)
        self.local = LocalContext(pixel_channels, kernel=local_kernel, hidden=local_hidden,
                                  dilation=local_dilation)
        self.coarse = CoarseIntensity(pixel_channels, context_dim=context_dim,
                                      grid=coarse_grid, hidden=coarse_hidden)
        self.year = self._build_year_offset(year_offset, year_group)
        self.year_gain = build_year_gain(year_gain, year_gain_group)
        if self.year_gain is not None and self.year is None:
            raise ValueError(
                "year_gain requires year_offset: the per-location gain multiplies gamma(t), "
                "so it is meaningless without a year offset.")

    @staticmethod
    def _build_year_offset(spec, year_group):
        return build_year_offset(spec, year_group)

    @property
    def receptive_field(self):
        return self.local.receptive_field

    def forward_terms(self, inputs):
        """The additive log-odds terms gamma, year_gain, m, s, c, each broadcastable to (B, 1, H, W)."""
        feats = [branch(inputs[branch.input_name]).permute(0, 3, 1, 2)
                 for branch in self.pixel_branches]
        x = torch.cat(feats, dim=1)

        context = None
        if len(self.context_branches):
            context = torch.cat(
                [branch(inputs[branch.input_name]).flatten(1) for branch in self.context_branches],
                dim=1)

        terms = {
            "s": self.susceptibility(x),
            "c": self.local(x),
            "m": self.coarse(x, context),
        }
        terms["gamma"] = (self.year(inputs[self.year.input_name]) if self.year is not None
                          else x.new_zeros(x.shape[0], 1, 1, 1))
        if self.year_gain is not None:
            g_res = self.year_gain(inputs[self.year_gain.input_name])
            terms["year_gain"] = terms["gamma"] * g_res
        else:
            terms["year_gain"] = x.new_zeros(x.shape[0], 1, 1, 1)
        return terms

    def forward(self, inputs):
        terms = self.forward_terms(inputs)
        logits = (terms["gamma"] + terms["year_gain"]
                  + terms["m"] + terms["s"] + terms["c"])
        # Head runs in float32 even under autocast, matching the other decoders.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            return torch.sigmoid(logits.float()).squeeze(1)
