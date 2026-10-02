"""BranchNorm: opt-in per-branch normalization before the fusion concat."""

import torch

from aic_risk_modeling.train import models

H = W = 16
STEPS = 10
N_CTX = 12


class FakeBranch(torch.nn.Module):
    """A branch with a controllable output scale / channel count for testing."""

    def __init__(self, name, out_channels, scale=1.0, const=False):
        super().__init__()
        self.input_name = name
        self.out_channels = out_channels
        self.scale = scale
        self.const = const

    def forward(self, x):
        b = x.shape[0]
        base = (torch.ones if self.const else torch.randn)(
            b, H, W, self.out_channels)
        return self.scale * base


def _rms(t):
    return t.float().pow(2).mean().sqrt().item()


def test_groupnorm_equalizes_scale():
    # Three sources spanning 4 orders of magnitude -> all ~unit RMS after norm.
    bn = models.BranchNorm([8, 8, 8], ["a", "b", "c"], mode="groupnorm")
    feats = [s * torch.randn(4, 8, H, W) for s in (0.01, 1.0, 100.0)]
    out = bn(feats)
    rms = [_rms(o) for o in out]
    assert all(0.7 < r < 1.4 for r in rms), rms
    assert max(rms) / min(rms) < 1.3, rms


def test_disabled_is_paramless_identity():
    bn = models.BranchNorm([8, 16], ["a", "b"], mode=None)
    feats = [torch.randn(2, 8, H, W), torch.randn(2, 16, H, W)]
    out = bn(feats)
    assert out is feats or all(o is f for o, f in zip(out, feats))
    assert list(bn.parameters()) == []


def test_scale_mode_one_gain_per_source():
    bn = models.BranchNorm([8, 16, 1], ["a", "b", "c"], mode="scale")
    assert sum(p.numel() for p in bn.parameters()) == 3
    feats = [torch.randn(2, c, H, W) for c in (8, 16, 1)]
    out = bn(feats)
    for o, f in zip(out, feats):
        assert o.shape == f.shape
        assert torch.allclose(o, f)


def test_groupnorm_single_channel_not_zeroed():
    # GroupNorm would zero a constant 1-ch source; fall back to a scale gain.
    bn = models.BranchNorm([1], ["x"], mode="groupnorm")
    assert isinstance(bn.norms[0], models._ScaleGain)
    const = torch.full((3, 1, H, W), 5.0)
    out = bn([const])[0]
    assert torch.allclose(out, const), out.mean().item()


def test_exclude_passes_through():
    bn = models.BranchNorm([8, 8], ["keep", "skip"], mode="groupnorm",
                           exclude=["skip"])
    assert isinstance(bn.norms[1], torch.nn.Identity)
    feats = [torch.randn(2, 8, H, W), 50.0 * torch.randn(2, 8, H, W)]
    out = bn(feats)
    assert torch.allclose(out[1], feats[1])      # excluded: untouched
    assert not torch.allclose(out[0], feats[0])  # normed: changed


def test_fusion_decoder_default_has_no_branch_norm_params():
    branches = [FakeBranch("a", 8), FakeBranch("b", 16)]
    off = models.decoder_fusion(branches)
    assert not any("branch_norm" in k for k in off.state_dict()), \
        "default must add no branch_norm params (checkpoint-compat)"
    on = models.decoder_fusion([FakeBranch("a", 8), FakeBranch("b", 16)],
                               branch_norm="groupnorm")
    assert any("branch_norm" in k for k in on.state_dict())


def test_fusion_decoder_forward_with_branch_norm():
    inputs = {"a": torch.zeros(2, 1), "b": torch.zeros(2, 1)}
    for mode in (None, "groupnorm", "scale"):
        model = models.decoder_fusion(
            [FakeBranch("a", 8, scale=0.01), FakeBranch("b", 16, scale=100.0)],
            branch_norm=mode).eval()
        with torch.no_grad():
            out = model(inputs)
        assert tuple(out.shape) == (2, H, W), (mode, out.shape)


def test_film_decoder_forward_with_branch_norm():
    branches = [
        models.get_identity([N_CTX, 5], "md_monthly"),   # climate context
        models.get_identity([H, W, 64], "im_single"),    # spatial (identity)
        models.get_unet_lite([H, W, 7], "im_single_cnn"),  # spatial (cnn)
    ]
    model = models.decoder_film(
        branches, branch_norm="groupnorm",
        branch_norm_exclude=["im_single"]).eval()
    inputs = {
        "md_monthly": torch.randn(2, N_CTX, 5),
        "im_single": torch.randn(2, H, W, 64),
        "im_single_cnn": torch.randn(2, H, W, 7),
    }
    with torch.no_grad():
        out = model(inputs)
    assert tuple(out.shape) == (2, H, W), out.shape


def test_mtsvit_decoder_forward_with_branch_norm():
    branches = [
        models.get_identity([STEPS, H, W, 6], "im_annual"),   # modality
        models.get_identity([N_CTX, 5], "md_monthly"),        # context
        models.get_identity([H, W, 64], "im_single"),         # spatial (identity)
        models.get_unet_lite([H, W, 7], "im_single_cnn"),     # spatial (cnn)
    ]
    model = models.decoder_mtsvit(
        branches, embed_dim=32, patch_size=8, temporal_depth=1, spatial_depth=1,
        num_heads=2, branch_norm="groupnorm",
        branch_norm_exclude=["<transformer>", "im_single"]).eval()
    assert isinstance(model.branch_norm.norms[0], torch.nn.Identity)
    inputs = {
        "im_annual": torch.randn(2, STEPS, H, W, 6),
        "md_monthly": torch.randn(2, N_CTX, 5),
        "im_single": torch.randn(2, H, W, 64),
        "im_single_cnn": torch.randn(2, H, W, 7),
    }
    with torch.no_grad():
        out = model(inputs)
    assert tuple(out.shape) == (2, H, W), out.shape
