"""decoder_simple / SimpleReadout: bare per-pixel linear readout baseline."""

import torch

from aic_risk_modeling.train import models


def _model(H=128, W=128, C=259):
    branch = models.get_pixel_mlp([H, W, C], "im_all",
                                  hidden=(256, 128, 64), out_channels=32)
    return models.decoder_simple([branch], num_classes=1), C


def test_simple_readout_shape_and_finite():
    model, C = _model()
    out = model({"im_all": torch.randn(2, 128, 128, C)})
    assert tuple(out.shape) == (2, 128, 128), out.shape
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_simple_readout_is_pointwise():
    model, C = _model(H=16, W=16, C=8)
    model.eval()
    x = torch.randn(1, 16, 16, C)
    base = model({"im_all": x})[:, 0, 0].clone()
    x[:, 7, 7] += 5.0
    assert torch.allclose(base, model({"im_all": x})[:, 0, 0], atol=1e-6)


def test_simple_readout_has_no_hidden_head():
    model, _ = _model()
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 1, [type(m).__name__ for m in model.modules()]
    assert convs[0].kernel_size == (1, 1), convs[0].kernel_size
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())


def test_simple_readout_backprops():
    model, C = _model()
    out = model({"im_all": torch.randn(2, 128, 128, C)})
    loss = torch.nn.functional.binary_cross_entropy(
        out.clamp(1e-6, 1 - 1e-6), torch.zeros_like(out))
    loss.backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all()
             for p in model.parameters() if p.requires_grad]
    assert grads and all(grads)
