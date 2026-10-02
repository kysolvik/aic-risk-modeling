"""eval/calibration: ECE/MCE, reliability bins, Platt/isotonic calibrators."""

import builtins

import numpy as np
import pytest

from aic_risk_modeling.eval.calibration import (
    apply_isotonic,
    apply_platt,
    expected_calibration_error,
    fit_calibrator,
    fit_isotonic,
    fit_platt,
    plot_reliability_diagram,
    reliability_bins,
)
from aic_risk_modeling.eval.eval import calc_stats


def test_perfectly_calibrated_has_low_ece():
    # Labels ~ Bernoulli(score): calibrated by construction.
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.0, 1.0, size=200_000)
    labels = rng.uniform(0.0, 1.0, size=scores.shape) < scores
    ece, mce, _ = expected_calibration_error(scores, labels, n_bins=15)
    assert ece < 0.01, ece
    assert mce < 0.05, mce


def test_overconfident_has_large_ece():
    # ~0.99 everywhere but half burn: gap ~0.49 in the top bin.
    rng = np.random.default_rng(1)
    scores = np.full(100_000, 0.99)
    labels = rng.uniform(0.0, 1.0, size=scores.shape) < 0.5
    ece, mce, _ = expected_calibration_error(scores, labels, n_bins=15)
    assert ece > 0.4, ece
    assert mce > 0.4, mce


def test_counts_sum_to_n_and_empty_bins_are_safe():
    scores = np.full(1000, 0.05)
    labels = np.zeros(1000, dtype=bool)
    bins = reliability_bins(scores, labels, n_bins=15)
    assert bins["count"].sum() == scores.size
    assert bins["count"][0] == scores.size
    assert np.isnan(bins["conf"][1:]).all()  # empty bins are NaN, not 0
    ece, mce, _ = expected_calibration_error(scores, labels, n_bins=15)
    assert np.isfinite(ece) and np.isfinite(mce)
    # conf ~0.05 vs freq 0.0 in the only populated bin.
    assert abs(ece - 0.05) < 1e-6, ece


def test_constant_half_prediction_matches_base_rate_gap():
    # Constant 0.5 prediction: ECE = |0.5 - base_rate|.
    base_rate = 0.2
    rng = np.random.default_rng(2)
    scores = np.full(100_000, 0.5)
    labels = rng.uniform(0.0, 1.0, size=scores.shape) < base_rate
    ece, _, _ = expected_calibration_error(scores, labels, n_bins=15)
    assert abs(ece - abs(0.5 - base_rate)) < 0.01, ece


def test_score_of_one_lands_in_last_bin():
    bins = reliability_bins(np.array([1.0, 1.0]), np.array([1, 0]), n_bins=10)
    assert bins["count"][-1] == 2
    assert bins["count"][:-1].sum() == 0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _overconfident_set(seed, factor=2.0, bias=0.0, n=400_000):
    """Well-calibrated logits, then distorted to forge a miscalibrated model."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, size=n)
    labels = rng.uniform(size=logits.shape) < _sigmoid(logits)
    scores = _sigmoid(logits * factor + bias)
    return scores, labels


def test_platt_recovers_known_scaling():
    # 2x-sharpened logits, no bias: Platt should recover a~0.5, b~0.
    scores, labels = _overconfident_set(10, factor=2.0)
    a, b = fit_platt(scores, labels)
    assert abs(a - 0.5) < 0.05, a
    assert abs(b) < 0.05, b


def test_platt_fixes_pure_bias():
    # Pure logit shift: Platt's intercept translates the curve back.
    scores, labels = _overconfident_set(11, factor=1.0, bias=1.5)
    ece_raw, _, _ = expected_calibration_error(scores, labels)
    a, b = fit_platt(scores, labels)
    ece_platt, _, _ = expected_calibration_error(apply_platt(scores, a, b), labels)
    assert ece_platt < ece_raw, (ece_raw, ece_platt)
    assert ece_platt < 0.02, ece_platt
    assert abs(b + 1.5) < 0.1, b  # intercept recovers the -1.5 shift


def test_isotonic_reduces_ece_on_distorted_model():
    scores, labels = _overconfident_set(12, factor=1.7, bias=-0.8)
    ece_raw, _, _ = expected_calibration_error(scores, labels)
    iso = fit_isotonic(scores, labels)
    ece_iso, _, _ = expected_calibration_error(apply_isotonic(scores, iso), labels)
    assert ece_iso < ece_raw, (ece_raw, ece_iso)
    assert ece_iso < 0.02, ece_iso


def test_apply_calibrators_preserve_shape_and_ranking():
    scores = np.array([[0.05, 0.4], [0.6, 0.95]])
    labels = np.array([[0, 0], [1, 1]])
    iso = fit_isotonic(scores, labels)
    a, b = fit_platt(scores, labels)
    for out in (apply_platt(scores, a, b), apply_isotonic(scores, iso)):
        assert out.shape == scores.shape
        assert np.all(np.argsort(out.ravel()) == np.argsort(scores.ravel()))


def test_fit_calibrator_dispatch():
    scores, labels = _overconfident_set(13, factor=2.0)
    for method in ("platt", "isotonic"):
        transform, info = fit_calibrator(method, scores, labels)
        out = transform(scores)
        assert out.shape == scores.shape
        assert isinstance(info, str) and method[:4] in info
        ece_after, _, _ = expected_calibration_error(out, labels)
        assert ece_after < 0.03, (method, ece_after)
    with pytest.raises(ValueError):
        fit_calibrator("bogus", scores, labels)


def test_quantile_binning_balances_counts():
    # Imbalanced toward 0: equal-width bins pile into bin 0, quantile bins spread evenly.
    rng = np.random.default_rng(14)
    scores = rng.beta(0.3, 8.0, size=200_000)  # mass near 0, long thin tail
    labels = rng.uniform(size=scores.shape) < scores

    uni = reliability_bins(scores, labels, n_bins=10, strategy="uniform")
    qua = reliability_bins(scores, labels, n_bins=10, strategy="quantile")

    assert uni["count"].sum() == scores.size
    assert qua["count"].sum() == scores.size
    assert uni["count"].max() / scores.size > 0.5
    assert qua["count"].max() / scores.size < 0.2
    for strategy in ("uniform", "quantile"):
        ece, mce, _ = expected_calibration_error(
            scores, labels, n_bins=10, strategy=strategy)
        assert np.isfinite(ece) and np.isfinite(mce)


def test_calc_stats_returns_calibrated_array():
    scores, labels = _overconfident_set(20, factor=2.0, n=20_000)
    gt = labels.astype(int)

    stats, calibrated = calc_stats(scores, gt, calibration_method="platt")
    assert isinstance(stats, dict)
    assert calibrated is not None and calibrated.shape == scores.shape
    assert not np.allclose(calibrated, scores)  # actually transformed

    stats2, none_cal = calc_stats(scores, gt, calibration_method="none")
    assert none_cal is None


def test_plot_returns_false_without_matplotlib(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] == "matplotlib":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    bins = reliability_bins(np.array([0.2, 0.8]), np.array([0, 1]), n_bins=10)
    assert plot_reliability_diagram(bins, 0.0, 0.0, str(tmp_path / "x.png")) is False
