"""weighted_bce_area: the summed-prediction vs summed-label area term of the opt-in area loss."""

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


def test_area_term_zero_when_counts_match():
    w = 9.0
    y_true = torch.zeros(1, 4, 4)
    y_true[0, :2, :2] = 1.0  # 4 fire pixels
    p = torch.full((1, 4, 4), 4.0 / 16.0)
    area = losses.area_log_ratio(pos_weight=w)
    assert area(y_true, _inflate(p, w)).item() < 1e-4
    assert area(y_true, _inflate(p * 2.0, w)).item() > 0.01


def test_area_term_empty_labels_stable():
    y_true = torch.zeros(2, 4, 4)
    y_pred = torch.full((2, 4, 4), 0.3)
    val = losses.area_log_ratio(pos_weight=1.0)(y_true, y_pred)
    expected = torch.log1p(torch.tensor(0.3 * 16.0)) ** 2
    assert torch.isfinite(val)
    assert torch.allclose(val, expected, atol=1e-4)


def test_area_term_gradient_flows():
    y_true = torch.zeros(1, 4, 4)
    y_true[0, 0, 0] = 1.0
    y_pred = torch.full((1, 4, 4), 0.2, requires_grad=True)
    losses.area_log_ratio(pos_weight=9.0)(y_true, y_pred).backward()
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()
    assert y_pred.grad.abs().sum() > 0


def test_area_term_block_size():
    # 4x4 chip with block_size=2: four 2x2 blocks compared independently.
    y_true = torch.zeros(1, 4, 4)
    y_true[0, :2, :2] = 1.0  # block (0,0): actual 4, others 0
    y_pred = torch.full((1, 4, 4), 0.25)  # every block: expected 1
    val = losses.area_log_ratio(pos_weight=1.0, block_size=2)(y_true, y_pred)
    per_block = [(torch.log1p(torch.tensor(1.0))
                  - torch.log1p(torch.tensor(a))) ** 2
                 for a in (4.0, 0.0, 0.0, 0.0)]
    expected = torch.stack(per_block).mean()
    assert torch.allclose(val, expected, atol=1e-5)


def test_compound_is_wbce_plus_weighted_area():
    w, aw = 9.0, 0.5
    torch.manual_seed(0)
    y_true = (torch.rand(2, 8, 8) > 0.8).float()
    y_pred = torch.rand(2, 8, 8).clamp(0.01, 0.99)
    compound = losses.weighted_bce_area(w, area_weight=aw)(y_true, y_pred)
    manual = (losses.weighted_bce(w)(y_true, y_pred)
              + aw * losses.area_log_ratio(w)(y_true, y_pred))
    assert torch.allclose(compound, manual, atol=1e-6)

    # sample_weight reaches the BCE term only.
    sw = torch.rand(2, 8, 8)
    compound_sw = losses.weighted_bce_area(w, area_weight=aw)(
        y_true, y_pred, sample_weight=sw)
    manual_sw = (losses.weighted_bce(w)(y_true, y_pred, sample_weight=sw)
                 + aw * losses.area_log_ratio(w)(y_true, y_pred))
    assert torch.allclose(compound_sw, manual_sw, atol=1e-6)

    zero_area = losses.weighted_bce_area(w, area_weight=0.0)(y_true, y_pred)
    assert torch.allclose(zero_area, losses.weighted_bce(w)(y_true, y_pred),
                          atol=1e-6)


def test_get_loss_wiring():
    w, aw = 9.0, 0.5
    torch.manual_seed(1)
    y_true = (torch.rand(2, 8, 8) > 0.8).float()
    y_pred = torch.rand(2, 8, 8).clamp(0.01, 0.99)
    via_get = losses.get_loss('weighted_bce_area', pos_weight=w,
                              area_weight=aw)(y_true, y_pred)
    manual = losses.weighted_bce_area(w, area_weight=aw)(y_true, y_pred)
    assert torch.allclose(via_get, manual, atol=1e-6)

    old = losses.get_loss('weighted_binary_crossentropy', pos_weight=w)
    assert torch.allclose(old(y_true, y_pred),
                          losses.weighted_bce(w)(y_true, y_pred), atol=1e-6)

    with pytest.raises(ValueError, match='weighted_bce_area'):
        losses.get_loss('nope')


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
