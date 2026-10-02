"""PixelMLP / PixelLSTM: per-pixel baseline encoders with no spatial mixing."""

import torch

from aic_risk_modeling.train import models


def test_pixel_mlp_shape_and_channels():
    enc = models.get_pixel_mlp([16, 16, 42], "im_annual", out_channels=32)
    assert enc.out_channels == 32, enc.out_channels
    out = enc(torch.randn(2, 16, 16, 42))
    assert tuple(out.shape) == (2, 16, 16, 32), out.shape
    assert torch.isfinite(out).all()


def test_pixel_mlp_is_pointwise():
    enc = models.get_pixel_mlp([8, 8, 6], "x", out_channels=4).eval()
    x = torch.randn(1, 8, 8, 6)
    base = enc(x)[:, 0, 0].clone()
    x[:, 5, 5] += 3.0  # perturb a far pixel
    assert torch.allclose(base, enc(x)[:, 0, 0], atol=1e-6)


def test_pixel_lstm_shape_and_channels():
    enc = models.get_pixel_lstm([6, 16, 16, 7], "im_annual", hidden=32)
    assert enc.out_channels == 32, enc.out_channels
    out = enc(torch.randn(2, 6, 16, 16, 7))
    assert tuple(out.shape) == (2, 16, 16, 32), out.shape
    assert torch.isfinite(out).all()


def test_pixel_lstm_is_pointwise():
    enc = models.get_pixel_lstm([4, 8, 8, 3], "x", hidden=8).eval()
    x = torch.randn(1, 4, 8, 8, 3)
    base = enc(x)[:, 0, 0].clone()
    x[:, :, 5, 5] += 3.0  # perturb a far pixel's whole sequence
    assert torch.allclose(base, enc(x)[:, 0, 0], atol=1e-5)


def test_pixel_branches_train_in_fusion_decoder():
    # coord_fourier (broadcast) forces the 128x128 grid.
    H = W = 128
    branches = [
        models.get_pixel_lstm([6, H, W, 7], "im_annual", hidden=16),
        models.get_pixel_mlp([H, W, 14], "im_single_cnn", out_channels=16),
        models.get_coord_fourier([1, 2], "md_single"),
    ]
    model = models.decoder_fusion(branches, num_classes=1)
    inputs = {
        "im_annual": torch.randn(2, 6, H, W, 7),
        "im_single_cnn": torch.randn(2, H, W, 14),
        "md_single": torch.randn(2, 1, 2),
    }
    out = model(inputs)
    assert tuple(out.shape) == (2, H, W), out.shape
    assert torch.isfinite(out).all()
    loss = torch.nn.functional.binary_cross_entropy(out, torch.zeros_like(out))
    loss.backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all()
             for p in model.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0
