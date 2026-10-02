"""Branch-rebalance knobs: get_projection and transformer_out_channels."""

import os

import pytest
import torch

from aic_risk_modeling.train import models, trainer

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

H = W = 32          # small grid keeps the CPU forward pass cheap
PATCH = 8
EMBED = 32
STEPS = 10
N_CTX = 12


def _build_branches(project_wide=False):
    """One branch of each kind; optionally project the wide identity branch."""
    if project_wide:
        wide = models.get_projection([H, W, 64], "im_single", out_channels=16)
    else:
        wide = models.get_identity([H, W, 64], "im_single")
    return [
        models.get_identity([STEPS, H, W, 6], "im_annual"),     # modality
        models.get_identity([STEPS, H, W, 7], "im_monthly"),    # modality
        models.get_identity([N_CTX, 5], "md_monthly"),          # context
        wide,                                                   # spatial (wide)
        models.get_unet_lite([H, W, 15], "im_single_cnn"),      # spatial (cnn)
    ]


def _inputs(batch=2):
    return {
        "im_annual": torch.randn(batch, STEPS, H, W, 6),
        "im_monthly": torch.randn(batch, STEPS, H, W, 7),
        "md_monthly": torch.randn(batch, N_CTX, 5),
        "im_single": torch.randn(batch, H, W, 64),
        "im_single_cnn": torch.randn(batch, H, W, 15),
    }


def test_projection_shape_and_routing():
    proj = models.get_projection([H, W, 64], "im_single", out_channels=16)
    assert proj.out_channels == 16
    out = proj(torch.randn(2, H, W, 64))
    assert tuple(out.shape) == (2, H, W, 16), out.shape
    model = models.decoder_mtsvit(
        _build_branches(project_wide=True), num_classes=1, embed_dim=EMBED,
        patch_size=PATCH, temporal_depth=1, spatial_depth=1, num_heads=2)
    assert "im_single" in [b.input_name for b in model.spatial_branches]


def test_model_kwargs_plumbing():
    input_features = {
        "im_single": {
            "feature_names": [f"f{i}" for i in range(64)],
            "timesteps": [],
            "shape": [H, W],
            "model_type": "projection",
            "model_kwargs": {"out_channels": 12},
        },
    }
    branch = trainer.build_all_models(input_features)[0]
    assert isinstance(branch, models.ProjectionModel)
    assert branch.out_channels == 12
    input_features["im_single"].pop("model_kwargs")
    input_features["im_single"]["model_type"] = "identity"
    branch = trainer.build_all_models(input_features)[0]
    assert branch.out_channels == 64


def test_transformer_out_channels_and_head_in():
    # Default: EMBED=32 halves to the 16 floor -> head_in = 16 + 64 + 16.
    default = models.decoder_mtsvit(
        _build_branches(), num_classes=1, embed_dim=EMBED, patch_size=PATCH,
        temporal_depth=1, spatial_depth=1, num_heads=2)
    assert default.head[0].in_channels == 16 + 64 + 16

    # Rebalanced: floor 24 + projected wide branch -> head_in = 24 + 16 + 16.
    rebalanced = models.decoder_mtsvit(
        _build_branches(project_wide=True), num_classes=1, embed_dim=EMBED,
        patch_size=PATCH, temporal_depth=1, spatial_depth=1, num_heads=2,
        transformer_out_channels=24)
    assert rebalanced.head[0].in_channels == 24 + 16 + 16

    out = rebalanced(_inputs())
    assert tuple(out.shape) == (2, H, W), out.shape


def test_real_configs_head_arithmetic(repo_config):
    # v22 (identity im_single, default floor): 16 + 64 + 16 + 16 = 112.
    # v24 (projection 16, floor 48):           48 + 16 + 16 + 16 = 96.
    for name, expected in (("mtsvit_test_v22", 112), ("mtsvit_test_v24", 96)):
        config = repo_config(name)
        model = trainer.build_decoder(
            config["decoder"],
            trainer.build_all_models(config["input_features"]),
            config.get("decoder_config"),
            num_classes=config.get("num_classes") or 1)
        assert model.head[0].in_channels == expected, (
            name, model.head[0].in_channels)


def test_v22_checkpoint_still_loads():
    ckpt = os.path.join(_REPO_ROOT, "out", "mtsvit_test_v22.pt")
    if not os.path.exists(ckpt):
        pytest.skip("no local checkpoint")
    model = trainer.load_model(ckpt)
    assert model.head[0].in_channels == 112
