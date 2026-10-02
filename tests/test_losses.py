"""Weighted BCE, its deflation inverse, and the area_ratio metric."""

import pytest
import torch

from aic_risk_modeling.train import losses
from aic_risk_modeling.train.metrics import SegmentationMetrics


def _inflate(p, w):
    """The pointwise optimum of weighted BCE: q = w*p / (w*p + 1 - p)."""
    return w * p / (w * p + 1.0 - p)


def test_deflation_inverts_wbce_optimum():
    for w in (1.0, 9.0, 20.0):
        p = torch.tensor([0.01, 0.05, 0.1, 0.3, 0.7, 0.9, 0.99])
        q = _inflate(p, w)
        assert torch.allclose(losses.deflate_probs(q, w), p, atol=1e-6)

    # argmin of elementwise WBCE at rate p is the closed-form inflated q.
    grid = torch.linspace(1e-4, 1.0 - 1e-4, 20001)
    for p, w in ((0.1, 9.0), (0.02, 20.0)):
        wbce = -(w * p * torch.log(grid) + (1.0 - p) * torch.log(1.0 - grid))
        q_star = grid[wbce.argmin()]
        assert abs(q_star - _inflate(torch.tensor(p), w)) < 1e-3


def test_get_loss_pos_weight_is_configurable():
    # All-positive labels: loss = bce * pos_weight, so the ratio is exact.
    y_true = torch.ones(2, 2)
    y_pred = torch.full((2, 2), 0.5)
    l9 = losses.get_loss("weighted_binary_crossentropy", pos_weight=9.0)
    l20 = losses.get_loss("weighted_binary_crossentropy", pos_weight=20.0)
    ratio = (l20(y_true, y_pred) / l9(y_true, y_pred)).item()
    assert abs(ratio - 20.0 / 9.0) < 1e-5
    with pytest.raises(ValueError, match="weighted_binary_crossentropy"):
        losses.get_loss("nope")


def test_metrics_area_ratio():
    w = 9.0
    torch.manual_seed(2)
    y_true = (torch.rand(4, 16, 16) > 0.9).float()
    p = torch.full_like(y_true, y_true.mean().item())
    m = SegmentationMetrics(pos_weight=w)
    m.update(y_true, _inflate(p, w))
    result = m.compute()
    assert abs(result['area_ratio'] - 1.0) < 1e-3

    m = SegmentationMetrics()
    m.update(y_true, y_true * 0.5)
    assert abs(m.compute()['area_ratio'] - 0.5) < 1e-6

    # Same keys on every path: the training CSV's DictWriter freezes fieldnames at epoch 1.
    keys = {'binary_iou', 'roc_auc', 'pr_auc', 'area_ratio'}
    assert set(SegmentationMetrics().compute().keys()) == keys  # no updates
    m = SegmentationMetrics()
    m.update(torch.zeros(1, 4, 4), torch.zeros(1, 4, 4))  # no positives
    assert set(m.compute().keys()) == keys
    m = SegmentationMetrics()
    m.update(torch.ones(1, 4, 4), torch.full((1, 4, 4), 0.9))
    assert set(m.compute().keys()) == keys
