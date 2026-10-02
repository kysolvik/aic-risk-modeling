"""UNetLite base_filters sets the branch width fed to the fusion head."""

import torch

from aic_risk_modeling.train import models, trainer


def test_default_width_is_unchanged():
    enc = models.get_unet_lite([32, 32, 14], "im_single_cnn")
    assert enc.out_channels == 16, enc.out_channels
    assert enc.e1.conv.conv1.pointwise.out_channels == 16
    assert enc.e2.conv.conv1.pointwise.out_channels == 32
    assert enc.bottleneck.conv1.pointwise.out_channels == 64
    out = enc(torch.randn(2, 32, 32, 14))
    assert tuple(out.shape) == (2, 32, 32, 16), out.shape


def test_width_scales_output_channels():
    for b in (16, 32, 48, 64):
        enc = models.get_unet_lite([32, 32, 14], "im_single_cnn",
                                   base_filters=b)
        assert enc.out_channels == b, (b, enc.out_channels)
        out = enc(torch.randn(2, 32, 32, 14))
        assert tuple(out.shape) == (2, 32, 32, b), out.shape
        assert torch.isfinite(out).all()


def test_width_preserves_full_resolution():
    enc = models.get_unet_lite([32, 32, 6], "x", base_filters=48).eval()
    x = torch.randn(1, 32, 32, 6)
    with torch.no_grad():
        base = enc(x)
        x2 = x.clone()
        x2[0, 0, 0] += 5.0
        moved = enc(x2)
    assert not torch.allclose(base[0, 0, 0], moved[0, 0, 0])
    assert torch.allclose(base[0, 31, 31], moved[0, 31, 31], atol=1e-5)


def test_non_fusion_head_still_collapses_to_one_channel():
    enc = models.get_unet_lite([32, 32, 5], "x", for_fusion=False,
                               base_filters=32)
    assert enc.out_channels == 1, enc.out_channels
    out = enc(torch.randn(2, 32, 32, 5))
    assert tuple(out.shape) == (2, 32, 32, 1), out.shape


def test_base_filters_routes_through_model_kwargs():
    branches = trainer.build_all_models({
        "im_single_cnn": {
            "model_type": "unet_lite",
            "feature_names": ["a"] * 14,
            "timesteps": [],
            "shape": [32, 32],
            "model_kwargs": {"base_filters": 48},
        },
    })
    assert len(branches) == 1
    assert branches[0].out_channels == 48, branches[0].out_channels


def test_wider_branch_widens_the_mtsvit_head():
    """The whole point: base_filters moves the head's full-res channel share."""
    def head_in(base_filters, transformer_out_channels):
        branches = trainer.build_all_models({
            "im_annual": {"model_type": "identity",
                          "feature_names": ["a"] * 7,
                          "timesteps": [-2, -1], "shape": [32, 32],
                          "stack_timesteps": True},
            "im_single_cnn": {"model_type": "unet_lite",
                              "feature_names": ["s"] * 14,
                              "timesteps": [], "shape": [32, 32],
                              "model_kwargs": {"base_filters": base_filters}},
        })
        decoder = trainer.build_decoder(
            "mtsvit", branches,
            {"embed_dim": 32, "patch_size": 8, "temporal_depth": 1,
             "spatial_depth": 1, "num_heads": 2, "mlp_ratio": 1,
             "transformer_out_channels": transformer_out_channels,
             "spatial_in_encoder": True})
        return decoder.head[0].in_channels

    # v44's allocation: 64 transformer + 16 full-res = 80, 20% full-res.
    assert head_in(16, 64) == 80
    # v53's: 16 transformer + 64 full-res = 80, same head width, 80% full-res.
    assert head_in(64, 16) == 80
