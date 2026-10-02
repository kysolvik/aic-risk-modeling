"""ConvLSTM branches (convlstm, convlstm_bottleneck) as fusion-decoder encoders."""

import pytest
import torch

from aic_risk_modeling.train import models

T, H, W, C = 3, 16, 16, 5


@pytest.mark.parametrize("factory, channels", [
    (models.get_convlstm, 128),
    (models.get_convlstm_bottleneck, 64),
])
def test_convlstm_branch_shape_and_channels(factory, channels):
    enc = factory([T, H, W, C], "im_annual")
    assert enc.out_channels == channels
    out = enc(torch.randn(2, T, H, W, C))
    assert tuple(out.shape) == (2, H, W, channels), out.shape
    assert torch.isfinite(out).all()


def test_forget_gate_bias_starts_at_one():
    cell = models.ConvLSTM2d(4, 8, 3)
    bias = cell.gates.bias.detach()
    assert torch.all(bias[8:16] == 1.0)
    assert torch.all(bias[:8] == 0.0) and torch.all(bias[16:] == 0.0)


def test_convlstm_trains_in_fusion_decoder():
    branches = [
        models.get_convlstm_bottleneck([T, H, W, C], "im_annual"),
        models.get_identity([H, W, 2], "im_single"),
    ]
    model = models.decoder_fusion(branches)
    out = model({"im_annual": torch.randn(2, T, H, W, C),
                 "im_single": torch.randn(2, H, W, 2)})
    assert tuple(out.shape) == (2, H, W), out.shape
    torch.nn.functional.binary_cross_entropy(out, torch.zeros_like(out)).backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all()
             for p in model.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0
