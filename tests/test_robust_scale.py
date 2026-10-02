"""robust_norm: IQR/1.349 scaling from tfdv quantiles, robust to nodata sentinels."""

import math

import numpy as np
import tensorflow as tf
from tensorflow_metadata.proto.v0 import statistics_pb2

from aic_risk_modeling.train import data_norm as dn


def _make_stats(name, values, corrupt_zeros=0):
    """A one-feature DatasetFeatureStatisticsList with a real QUANTILES hist."""
    arr = np.concatenate([np.asarray(values, np.float64),
                          np.zeros(corrupt_zeros, np.float64)])
    edges = [float(np.percentile(arr, p)) for p in range(0, 101, 10)]  # 11 edges
    sl = statistics_pb2.DatasetFeatureStatisticsList()
    ds = sl.datasets.add()
    ds.num_examples = len(arr)
    feat = ds.features.add()
    feat.path.step.append(name)
    ns = feat.num_stats
    ns.mean = float(arr.mean())
    ns.std_dev = float(arr.std())
    ns.min = float(arr.min())
    ns.max = float(arr.max())
    ns.median = float(np.median(arr))
    hist = ns.histograms.add()
    hist.type = statistics_pb2.Histogram.QUANTILES
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = hist.buckets.add()
        b.low_value, b.high_value, b.sample_count = lo, hi, len(arr) / 10.0
    return sl


def _expected_iqr_scale(values):
    q25, q75 = np.percentile(values, [25, 75])
    return (q75 - q25) / 1.349


def test_robust_scale_from_quantiles():
    vals = np.linspace(295.0, 305.0, 1001)  # clean temp-like K values
    sl = _make_stats("temp", vals, corrupt_zeros=5)  # 5 sentinel 0 K pixels
    s = dn.get_norm_stats(sl, "temp")
    exp = _expected_iqr_scale(vals)  # sentinel is outside q25..q75, so ignored
    assert s["robust_scale"] is not None
    assert math.isclose(s["robust_scale"], exp, rel_tol=0.05), (s["robust_scale"], exp)
    # std_dev is inflated by the sentinel zeros; the robust scale is not.
    clean_std = float(np.std(vals))
    assert s["stddev"] > 3 * clean_std, (s["stddev"], clean_std)      # inflated
    assert math.isclose(s["robust_scale"], exp, rel_tol=0.05)         # unaffected


def test_normalizer_uses_robust_scale_and_fills_sentinel():
    vals = np.linspace(295.0, 305.0, 1001)
    sl = _make_stats("temp", vals, corrupt_zeros=5)
    s = dn.get_norm_stats(sl, "temp")

    orig = dn.load_stats_from_text
    dn.load_stats_from_text = lambda path: sl
    try:
        fn = dn.create_normalizer("dummy.pbtxt", ["temp"], robust_features={"temp"})
    finally:
        dn.load_stats_from_text = orig

    raw = tf.constant([[0.0, s["median"], float("nan"), s["median"] + s["robust_scale"]]])
    out = fn({"temp": tf.identity(raw)})["temp"].numpy()[0]
    # sentinel(min) -> center -> 0 ; median -> 0 ; nan -> 0 ; +1 robust unit -> ~1
    assert abs(out[0]) < 1e-4, out
    assert abs(out[1]) < 1e-4, out
    assert abs(out[2]) < 1e-4, out
    assert math.isclose(out[3], 1.0, abs_tol=1e-3), out


def test_json_stats_fall_back_to_stddev():
    # JSON stats have no quantiles -> robust features fall back to std_dev.
    stats = {"features": {"temp": {"mean": 300.0, "stddev": 4.0, "min": 0.0,
                                   "max": 310.0, "median": 301.0}}}
    s = dn.get_norm_stats(stats, "temp")
    assert "robust_scale" not in s

    orig = dn.load_stats_json
    dn.load_stats_json = lambda path: stats
    try:
        fn = dn.create_normalizer("dummy.json", ["temp"], robust_features={"temp"})
    finally:
        dn.load_stats_json = orig

    raw = tf.constant([[0.0, 301.0, 305.0]])  # min->median->0 ; median->0 ; +4K->1
    out = fn({"temp": tf.identity(raw)})["temp"].numpy()[0]
    assert abs(out[0]) < 1e-4, out
    assert abs(out[1]) < 1e-4, out
    assert math.isclose(out[2], (305.0 - 301.0) / (4.0 + 1e-7), abs_tol=1e-3), out


def test_non_robust_feature_unchanged():
    vals = np.linspace(295.0, 305.0, 1001)
    sl = _make_stats("temp", vals)
    s = dn.get_norm_stats(sl, "temp")
    orig = dn.load_stats_from_text
    dn.load_stats_from_text = lambda path: sl
    try:
        fn = dn.create_normalizer("dummy.pbtxt", ["temp"], robust_features=set())
    finally:
        dn.load_stats_from_text = orig
    raw = tf.constant([[s["mean"], s["mean"] + s["stddev"]]])
    out = fn({"temp": tf.identity(raw)})["temp"].numpy()[0]
    assert abs(out[0]) < 1e-4, out
    assert math.isclose(out[1], 1.0, abs_tol=1e-3), out


def test_degenerate_quantiles_return_none():
    sl = _make_stats("flat", np.full(1000, 7.0))  # zero IQR
    s = dn.get_norm_stats(sl, "flat")
    assert s["robust_scale"] is None
