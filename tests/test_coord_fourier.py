"""CoordFourierForFusion: Fourier coordinate encoder."""

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

