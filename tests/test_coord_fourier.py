"""CoordFourierForFusion: Fourier coordinate encoder and its routing to the spatial head."""

import torch

from aic_risk_modeling.train import models


def test_coord_fourier_shape_and_channels():
    enc = models.get_coord_fourier([1, 2], "md_single")
    assert enc.out_channels == 16, enc.out_channels
    out = enc(torch.randn(2, 1, 2))
    assert tuple(out.shape) == (2, 128, 128, 16), out.shape
    assert torch.isfinite(out).all()


def test_coord_fourier_broadcast_is_constant_across_grid():
    enc = models.get_coord_fourier([1, 2], "md_single").eval()
    out = enc(torch.randn(3, 1, 2))
    assert torch.allclose(out[:, 0, 0], out[:, 50, 70])


def test_coord_fourier_encoding_is_deterministic():
    torch.manual_seed(0)
    a = models.get_coord_fourier([1, 2], "md_single")
    torch.manual_seed(0)
    b = models.get_coord_fourier([1, 2], "md_single")
    assert torch.equal(a.freq_proj, b.freq_proj)


def test_coord_branch_routes_to_spatial_in_mtsvit():
    # Broadcast fusion branches emit PATCH_SIZE x PATCH_SIZE, so the grid must be 128.
    H = W = 128
    branches = [
        models.get_identity([10, H, W, 6], "im_annual"),   # modality
        models.get_identity([12, 5], "md_monthly"),        # climate context
        models.get_coord_fourier([1, 2], "md_single"),     # coords -> spatial
    ]
    model = models.decoder_mtsvit(
        branches, num_classes=1, embed_dim=32, patch_size=8,
        temporal_depth=1, spatial_depth=1, num_heads=2)
    assert model.context_names == ["md_monthly"], model.context_names
    assert "md_single" in [b.input_name for b in model.spatial_branches]

    model.eval()
    out = model({
        "im_annual": torch.randn(2, 10, H, W, 6),
        "md_monthly": torch.randn(2, 12, 5),
        "md_single": torch.randn(2, 1, 2),
    })
    assert tuple(out.shape) == (2, H, W), out.shape
