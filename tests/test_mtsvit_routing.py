"""MTSViTFusion branch routing (modalities / temporal context / spatial features) and gamma."""

import pytest
import torch

from aic_risk_modeling.train import models

H = W = 32          # small grid keeps the CPU forward pass cheap
PATCH = 8
EMBED = 32
STEPS = 10
N_CTX = 12


def _build_model(spatial_in_encoder=False):
    """One branch of each kind, including a non-identity unet_lite branch."""
    branches = [
        models.get_identity([STEPS, H, W, 6], "im_annual"),     # modality
        models.get_identity([STEPS, H, W, 7], "im_monthly"),    # modality
        models.get_identity([N_CTX, 5], "md_monthly"),          # context
        models.get_identity([H, W, 64], "im_single"),           # spatial (id)
        models.get_unet_lite([H, W, 7], "im_single_cnn"),       # spatial (cnn)
    ]
    model = models.decoder_mtsvit(
        branches, num_classes=1, embed_dim=EMBED, patch_size=PATCH,
        temporal_depth=1, spatial_depth=1, num_heads=2,
        spatial_in_encoder=spatial_in_encoder)
    return model


def test_non_identity_branch_routes_to_spatial():
    model = _build_model()
    spatial_names = [b.input_name for b in model.spatial_branches]

    assert model.temporal_names == ["im_annual", "im_monthly"], model.temporal_names
    assert model.context_names == ["md_monthly"], model.context_names
    assert "im_single_cnn" in spatial_names, spatial_names
    assert "im_single" in spatial_names, spatial_names


def test_head_includes_dropped_branch_channels():
    model = _build_model()
    spatial_out = sum(b.out_channels for b in model.spatial_branches)
    upsample_out = model.head[0].in_channels - spatial_out
    # im_single identity (64) + im_single_cnn unet_lite fusion features (16).
    assert spatial_out == 64 + 16, spatial_out
    assert model.head[0].in_channels == upsample_out + 80


def test_unet_lite_exposes_fusion_features():
    fusion = models.get_unet_lite([H, W, 7], "im_single_cnn")
    assert fusion.out_channels == 16, fusion.out_channels
    out = fusion(torch.randn(2, H, W, 7))
    assert tuple(out.shape) == (2, H, W, 16), out.shape

    standalone = models.get_unet_lite([H, W, 7], "x", for_fusion=False)
    assert standalone.out_channels == 1, standalone.out_channels
    out = standalone(torch.randn(2, H, W, 7))
    assert tuple(out.shape) == (2, H, W, 1), out.shape


def test_spatial_in_encoder_option():
    # Default off: original architecture, so pre-option checkpoints still load.
    off = _build_model(spatial_in_encoder=False)
    assert not hasattr(off, "spatial_patch_embeds")
    assert off.modality_fuse.in_features == len(off.temporal_names) * EMBED, \
        off.modality_fuse.in_features

    on = _build_model(spatial_in_encoder=True)
    spatial_names = [b.input_name for b in on.spatial_branches]
    assert set(on.spatial_patch_embeds.keys()) == set(spatial_names), \
        list(on.spatial_patch_embeds.keys())
    for branch in on.spatial_branches:
        conv = on.spatial_patch_embeds[branch.input_name]
        assert conv.in_channels == branch.out_channels, branch.input_name
        assert conv.out_channels == EMBED
        assert conv.kernel_size == (PATCH, PATCH)
        assert conv.stride == (PATCH, PATCH)
    n_sources = len(on.temporal_names) + len(on.spatial_branches)
    assert on.modality_fuse.in_features == n_sources * EMBED, \
        on.modality_fuse.in_features
    assert on.modality_fuse.out_features == EMBED


def test_forward_pass_shape():
    inputs = {
        "im_annual": torch.randn(2, STEPS, H, W, 6),
        "im_monthly": torch.randn(2, STEPS, H, W, 7),
        "md_monthly": torch.randn(2, N_CTX, 5),
        "im_single": torch.randn(2, H, W, 64),
        "im_single_cnn": torch.randn(2, H, W, 7),
    }
    for spatial_in_encoder in (False, True):
        model = _build_model(spatial_in_encoder=spatial_in_encoder).eval()
        with torch.no_grad():
            out = model(inputs)
        assert tuple(out.shape) == (2, H, W), (spatial_in_encoder, out.shape)


def _build_gamma_model(offsets, num_classes=1):
    """As `_build_model`, plus an md_year identity branch and a gamma offset."""
    branches = [
        models.get_identity([STEPS, H, W, 6], "im_annual"),     # modality
        models.get_identity([STEPS, H, W, 7], "im_monthly"),    # modality
        models.get_identity([N_CTX, 5], "md_monthly"),          # context
        models.get_unet_lite([H, W, 7], "im_single_cnn"),       # spatial (cnn)
        models.get_identity([1, 1], "md_year"),                 # year key (carved out)
    ]
    return models.decoder_mtsvit(
        branches, num_classes=num_classes, embed_dim=EMBED, patch_size=PATCH,
        temporal_depth=1, spatial_depth=1, num_heads=2,
        year_group="md_year", year_offset={"offsets": offsets})


def _gamma_inputs(year):
    return {
        "im_annual": torch.randn(2, STEPS, H, W, 6),
        "im_monthly": torch.randn(2, STEPS, H, W, 7),
        "md_monthly": torch.randn(2, N_CTX, 5),
        "im_single_cnn": torch.randn(2, H, W, 7),
        "md_year": torch.full((2, 1, 1), float(year)),
    }


def test_year_group_is_carved_out_of_routing():
    from aic_risk_modeling.train.factored import YearOffset
    model = _build_gamma_model({"2023": 0.0, "2024": 0.7})
    routed = set(model.temporal_names) | set(model.context_names) | {
        b.input_name for b in model.spatial_branches}
    assert "md_year" not in routed, routed          # not mis-routed as (T, F) context
    assert isinstance(model.year, YearOffset), type(model.year)
    assert model.year_group == "md_year"


def test_default_model_is_gamma_free():
    model = _build_model()
    assert model.year is None and model.year_group is None


def test_gamma_shifts_forward_between_years():
    offsets = {"2023": 0.0, "2024": 0.7}
    delta = offsets["2024"] - offsets["2023"]
    model = _build_gamma_model(offsets).eval()

    # The lookup is exact and per-sample: gamma(2023) == 0 -> (B, 1, 1, 1).
    g23 = model.year(torch.full((2, 1, 1), 2023.0))
    assert tuple(g23.shape) == (2, 1, 1, 1) and torch.allclose(g23, torch.zeros_like(g23))

    inputs = _gamma_inputs(2023)
    with torch.no_grad():
        out23 = model(inputs)
        out24 = model(dict(inputs, md_year=torch.full((2, 1, 1), 2024.0)))
    # Where unsaturated, a gamma delta shifts logits by exactly delta.
    assert (out24 >= out23 - 1e-6).all()
    lo, hi = 1e-4, 1 - 1e-4
    unsat = (out23 > lo) & (out23 < hi) & (out24 > lo) & (out24 < hi)
    assert unsat.any(), "no unsaturated pixels to check the exact shift"
    shift = torch.logit(out24[unsat]) - torch.logit(out23[unsat])
    assert torch.allclose(shift, torch.full_like(shift, delta), atol=1e-3), shift.mean().item()


def test_gamma_is_binary_only():
    with pytest.raises(ValueError, match="binary-only"):
        _build_gamma_model({"2023": 0.0, "2024": 0.7}, num_classes=3)
