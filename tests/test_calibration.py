"""eval/calibration: ECE/MCE, reliability bins, Platt/isotonic calibrators, calibrated GeoTIFFs."""

import builtins
import os
import tempfile

import numpy as np
import pytest

from aic_risk_modeling.eval.calibration import (
    apply_isotonic,
    apply_platt,
    apply_temperature,
    expected_calibration_error,
    fit_calibrator,
    fit_isotonic,
    fit_platt,
    fit_temperature,
    plot_reliability_diagram,
    reliability_bins,
)
from aic_risk_modeling.eval.eval import (
    calc_stats,
    write_calibrated_predictions,
)


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


def test_apply_temperature_identity_at_one():
    scores = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    out = apply_temperature(scores, 1.0)
    assert np.allclose(out, scores, atol=1e-5), out
    assert out.shape == scores.shape


def test_apply_temperature_preserves_half_crossing_and_monotonicity():
    scores = np.array([0.05, 0.4, 0.5, 0.6, 0.95])
    out = apply_temperature(scores, 2.5)
    assert abs(out[2] - 0.5) < 1e-9, out
    assert out[0] > scores[0] and out[1] > scores[1]
    assert out[3] < scores[3] and out[4] < scores[4]
    assert np.all(np.diff(out) > 0), out  # still strictly increasing


def test_fit_temperature_recovers_known_scaling():
    # Logits sharpened 2x -> fitted T ~2.
    rng = np.random.default_rng(3)
    logits = rng.normal(0.0, 2.0, size=400_000)
    labels = rng.uniform(size=logits.shape) < _sigmoid(logits)
    overconfident = _sigmoid(logits * 2.0)
    T = fit_temperature(overconfident, labels)
    assert abs(T - 2.0) < 0.15, T


def test_fit_temperature_near_one_when_calibrated():
    rng = np.random.default_rng(4)
    logits = rng.normal(0.0, 2.0, size=400_000)
    labels = rng.uniform(size=logits.shape) < _sigmoid(logits)
    scores = _sigmoid(logits)
    T = fit_temperature(scores, labels)
    assert abs(T - 1.0) < 0.1, T


def test_fit_and_apply_reduces_ece_on_overconfident_model():
    rng = np.random.default_rng(5)
    logits = rng.normal(0.0, 2.0, size=400_000)
    labels = rng.uniform(size=logits.shape) < _sigmoid(logits)
    overconfident = _sigmoid(logits * 2.0)
    ece_before, _, _ = expected_calibration_error(overconfident, labels)
    T = fit_temperature(overconfident, labels)
    ece_after, _, _ = expected_calibration_error(
        apply_temperature(overconfident, T), labels)
    assert ece_after < ece_before, (ece_before, ece_after)
    assert ece_after < 0.02, ece_after


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


def test_platt_fixes_bias_that_temperature_cannot():
    # Pure logit shift: temperature can't translate the curve, Platt's intercept can.
    scores, labels = _overconfident_set(11, factor=1.0, bias=1.5)
    ece_raw, _, _ = expected_calibration_error(scores, labels)

    T = fit_temperature(scores, labels)
    ece_temp, _, _ = expected_calibration_error(apply_temperature(scores, T), labels)

    a, b = fit_platt(scores, labels)
    ece_platt, _, _ = expected_calibration_error(apply_platt(scores, a, b), labels)

    assert ece_platt < ece_temp, (ece_temp, ece_platt)
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
    for method in ("temperature", "platt", "isotonic"):
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


def test_write_calibrated_csv_round_trip():
    import pandas as pd

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "pred.csv")
        out = os.path.join(d, "pred_cal.csv")
        pd.DataFrame({"pred": [0.1, 0.6, 0.9], "id": [7, 8, 9]}).to_csv(src, index=False)
        calibrated = np.array([0.2, 0.5, 0.7])

        write_calibrated_predictions(src, out, calibrated)
        got = pd.read_csv(out)
        assert np.allclose(got["pred"].values, calibrated)
        assert list(got["id"].values) == [7, 8, 9]  # other columns preserved


def test_write_calibrated_tif_round_trip():
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "pred.tif")
        out = os.path.join(d, "pred_cal.tif")
        transform = from_origin(-120.5, 38.2, 0.01, 0.01)
        prof = dict(driver="GTiff", height=4, width=5, count=1, dtype="float32",
                    crs="EPSG:4326", transform=transform, compress="lzw")
        scores = np.linspace(0.0, 1.0, 20, dtype="float32").reshape(4, 5)
        with rio.open(src, "w", **prof) as dst:
            dst.write(scores, 1)
        calibrated = (scores * 0.5).astype("float32")

        write_calibrated_predictions(src, out, calibrated)
        with rio.open(out) as ds:
            assert ds.crs == rio.crs.CRS.from_epsg(4326)
            assert ds.transform == transform
            assert ds.count == 1 and ds.dtypes[0] == "float32"
            assert np.allclose(ds.read(1), calibrated)


def test_plot_returns_false_without_matplotlib(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] == "matplotlib":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    bins = reliability_bins(np.array([0.2, 0.8]), np.array([0, 1]), n_bins=10)
    assert plot_reliability_diagram(bins, 0.0, 0.0, str(tmp_path / "x.png")) is False
