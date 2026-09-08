"""Factored two-scale fire model.

    logit(x, t) = gamma(t) + m(x, t) + s(x, t) + c(x, t)
                  ^when      ^where,     ^pixel     ^local
                              coarsely    suscept.   spread

Additive in log-odds, so risk factorises multiplicatively and each term is
readable per pixel with no occlusion pass. The split is driven by three
measurements on this dataset rather than by architecture fashion:

  * Patch tokens cannot represent the target. Of patches containing any fire,
    74.3% are mixed at 2x2, 94.2% at 4x4, 99.1% at 8x8. So every image pathway
    here stays at full resolution or is pooled only where the DATA is genuinely
    coarse -- never patch-tokenized as a compression device.
  * Real within-chip spatial structure is gone by ~18 km and zero by 36 km
    (label autocorrelation with the chip mean removed: 0.580 at 0.56 km, 0.126 at
    8.9 km, 0.046 at 17.8 km, -0.016 at 35.6 km). So `c` is a short-range kernel
    with an explicitly measured receptive field, and there is no long-range
    attention because there is nothing at long range to attend to.
  * Pooled PR-AUC is mostly a chip-intensity metric: an oracle that knows each
    71 km chip's burn rate and predicts it flat already scores 0.3145 of the best
    model's 0.3566, and rescaling a model's chips to the true rate lifts it to
    0.4193. So `m` exists as an explicit coarse term rather than being left
    implicit.

`s` and `c` are disjoint by construction: `c` starts with a CENTRE-MASKED
depthwise conv, so the pixel's own features never enter the local term. That is
what makes the additive decomposition an actual ignition-vs-spread split rather
than an arbitrary partition of a sum.

`gamma` is a frozen, global, offline-fit per-year offset -- see
scripts/analysis/fit_year_offset.py for why it is not learned in-network and why
it is not spatially varying.
"""

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


# --------------------------------------------------------------------- encoders

class PixelTemporalEncoder(nn.Module):
    """Per-pixel temporal transformer: (B, T, H, W, C) -> (B, H, W, D).

    TSViT's stage-1 ordering (attend over time first) with the patch tokenization
    removed. Every pixel is its own length-T sequence, so nothing is spatially
    compressed and no sub-patch detail has to be reconstructed afterwards. Cost
    scales as H*W*T^2*D, which is why `dim` defaults to 32 rather than the 128 used
    by the patch-token model -- full resolution at D=128 is ~64x the stage-1 FLOPs
    and does not fit at batch_size 2.
    """

    def __init__(self, input_shape, input_name=None, dim=32, depth=2, num_heads=4,
                 mlp_ratio=2, dropout=0.1, pool="mean"):
        super().__init__()
        if len(input_shape) != 4:
            raise ValueError(f"expected (T, H, W, C) input_shape, got {input_shape}")
        steps, _, _, channels = input_shape
        if pool not in ("mean", "cls", "last"):
            raise ValueError(f"unknown pool {pool!r}")
        self.input_name = input_name
        self.out_channels = dim
        self.pool = pool
        self.proj = nn.Linear(channels, dim)
        self.temporal_pos = nn.Parameter(torch.zeros(steps, dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if pool == "cls" else None
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)
        self.layers = nn.ModuleList(
            [TransformerLayer(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch, steps, height, width, _ = x.shape
        # (B, T, H, W, C) -> (B*H*W, T, C): every pixel becomes an independent
        # sequence, so attention is purely temporal.
        x = x.permute(0, 2, 3, 1, 4).reshape(batch * height * width, steps, -1)
        x = self.proj(x) + self.temporal_pos
        if self.cls is not None:
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.pool == "cls":
            x = x[:, 0]
        elif self.pool == "last":
            x = x[:, -1]
        else:
            x = x.mean(dim=1)
        return x.reshape(batch, height, width, self.out_channels)


class CoarseTemporalEncoder(nn.Module):
    """Temporal transformer on a pooled grid: (B, T, H, W, C) -> (B, H, W, D).

    For inputs whose NATIVE resolution is already coarse -- CHIRPS, ERA5/AgERA5
    CWD, VPD, temperature, evaporation, precipitation are all >=4 km against a
    556 m pixel -- average-pooling to `grid` before the temporal encoder discards
    nothing real and cuts cost by (H*W)/(grid^2). This is principled downsampling,
    unlike patch tokenization of genuinely fine-grained bands.
    """

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
        feats = self.encoder(pooled).permute(0, 3, 1, 2)          # (B, D, g, g)
        up = F.interpolate(feats, size=(height, width), mode="bilinear", align_corners=False)
        return up.permute(0, 2, 3, 1)


# ------------------------------------------------------------------------ terms

class YearOffset(nn.Module):
    """Frozen global year term `gamma(t)`, looked up by the raw `md_year` input.

    The offsets themselves are fit offline (SOI(Aug-Oct, Y-1) + log basin burn(Y-1),
    forward-chained) because the network cannot identify a year effect: it sees
    ~360 year-constant scalars against 10-13 distinct year-values. `md_year` is used
    ONLY as a lookup key, never as a learned feature -- the extrapolating content
    lives in the climate/burn regressors, which is why an absolute year here does
    not reintroduce the generalization problem that got `md_year` dropped from the
    v11-lineage configs.

    Requires the group carrying `md_year` to be configured with `normalize: false`,
    or the key arrives scaled and the lookup is meaningless.
    """

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


class PixelSusceptibility(nn.Module):
    """`s`: strictly pointwise (1x1) logit contribution over the full-res stack.

    This is the term the routing audit was about: every per-pixel value reaches the
    output at full resolution here, rather than 14 of 152 as in the patch-token
    model. 1x1 convs only, so it contains no spatial mixing by construction.
    """

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
    """`c`: neighbourhood contribution, with the centre pixel masked out.

    A learnable depthwise k x k kernel whose centre tap is forced to zero, followed
    by 1x1 convs. Two consequences that matter:

      * the receptive field is EXACTLY `kernel` -- not an emergent property of a
        conv stack -- so it can be swept and reported in km (k=9 -> 5.0 km,
        k=17 -> 9.5 km at 556 m/px);
      * the pixel's own features never enter `c`, so `s` and `c` are disjoint and
        the additive split is a real ignition-vs-spread decomposition. Stacked
        convs cannot give this: masking only the first layer still lets the centre
        leak back in at the second.

    `kernel=1` degenerates to no context at all, which makes it the pointwise
    control arm.
    """

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
        mask[0, 0, kernel // 2, kernel // 2] = 0.0          # centre tap permanently off
        self.register_buffer("centre_mask", mask)
        self.body = nn.Sequential(nn.Conv2d(in_channels, hidden, 1), nn.ReLU())
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)                      # starts as an exact no-op
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
    """`m`: coarse burn-intensity contribution on a `grid` x `grid` lattice.

    Defaults to a 4x4 lattice = 17.8 km cells, which is where within-chip label
    autocorrelation has already decayed to 0.046 -- finer than that is `s`/`c`'s
    job, coarser loses nothing. Supervise this term with the per-chip area loss
    (`weighted_bce_area`); that loss matches each chip's total independently, which
    is exactly right for a chip-intensity term and exactly why it cannot teach the
    global year factor that `gamma` carries.

    `grid=None` disables the term entirely (it returns exact zeros and holds no
    parameters), mirroring `local_kernel=1` for `c`. That makes every ablation of
    the factorisation a config change rather than a code change.
    """

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


# ------------------------------------------------------------------------ model

class FactoredFireModel(nn.Module):
    """Combines the four terms into a per-pixel fire probability.

    Routing is EXPLICIT: every branch's `input_name` must be listed in exactly one
    of `pixel_groups` / `context_groups` / `year_group`, or construction fails. The
    patch-token model routed by `hasattr(branch, "input_shape")`, so adding that
    attribute to an encoder silently changed its pathway; naming the groups makes a
    misrouted branch a loud error instead of a quiet regression.

    Resolution is the ENCODER's business, not the decoder's: a group is coarse
    because it uses `coarse_temporal`, not because the decoder demoted it. That
    keeps "how finely is this modality resolved" in one place, next to the reason
    (the band's native resolution).
    """

    def __init__(self, branch_models, num_classes=1, pixel_groups=None,
                 context_groups=None, year_group=None, local_kernel=9, coarse_grid=4,
                 susceptibility_hidden=(128, 64), susceptibility_dropout=0.0,
                 local_hidden=64, local_dilation=1, coarse_hidden=64, year_offset=None):
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "FactoredFireModel is additive in log-odds and is binary-only; "
                f"got num_classes={num_classes}. Use decoder_fusion for multiclass.")
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

        self.num_classes = num_classes
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
                    "(use model_type 'identity' or 'projection')")
            context_dim += int(math.prod(shape))

        self.susceptibility = PixelSusceptibility(
            pixel_channels, hidden=tuple(susceptibility_hidden), dropout=susceptibility_dropout)
        self.local = LocalContext(pixel_channels, kernel=local_kernel, hidden=local_hidden,
                                  dilation=local_dilation)
        self.coarse = CoarseIntensity(pixel_channels, context_dim=context_dim,
                                      grid=coarse_grid, hidden=coarse_hidden)
        self.year = self._build_year_offset(year_offset, year_group)

    @staticmethod
    def _build_year_offset(spec, year_group):
        if not spec:
            return None
        if not year_group:
            raise ValueError("year_offset given but year_group is unset")
        spec = dict(spec)
        path = spec.pop("coeffs_path", None)
        spec.pop("terms", None)                 # documentation only; the table is authoritative
        offsets = spec.pop("offsets", None)
        kw = {k: spec.pop(k) for k in ("trainable", "strict") if k in spec}
        if spec:
            raise ValueError(f"unknown year_offset keys: {sorted(spec)}")
        if (path is None) == (offsets is None):
            raise ValueError("year_offset needs exactly one of coeffs_path / offsets")
        if path is not None:
            return YearOffset.from_json(path, input_name=year_group, **kw)
        return YearOffset(offsets, input_name=year_group, **kw)

    @property
    def receptive_field(self):
        """Effective receptive field in pixels: the local kernel, exactly."""
        return self.local.receptive_field

    def forward_terms(self, inputs):
        """The four additive log-odds terms, each broadcastable to (B, 1, H, W).

        Exposed because it IS the explanation: gamma = when, m = where coarsely,
        s = this pixel's own susceptibility, c = what the neighbourhood adds.
        """
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
        return terms

    def forward(self, inputs):
        terms = self.forward_terms(inputs)
        logits = terms["gamma"] + terms["m"] + terms["s"] + terms["c"]
        # Head runs in float32 even under autocast, matching the other decoders.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            return torch.sigmoid(logits.float()).squeeze(1)


# ---------------------------------------------------------------------- factories

def get_pixel_temporal(input_shape, input_name=None, **kwargs):
    return PixelTemporalEncoder(input_shape, input_name=input_name, **kwargs)


def get_coarse_temporal(input_shape, input_name=None, **kwargs):
    return CoarseTemporalEncoder(input_shape, input_name=input_name, **kwargs)


def decoder_factored(branch_models, num_classes=1, **kwargs):
    return FactoredFireModel(branch_models, num_classes=num_classes, **kwargs)
