"""lag1: shifts md_oni one month later so its last value is Oct-Dec(Y-1), not Nov-Jan."""

import numpy as np
import tensorflow as tf

from aic_risk_modeling.train import data_norm, transforms
from aic_risk_modeling.train.data_loader import apply_transforms


def test_lag1_shifts_and_edge_pads():
    x = tf.constant([[1., 2., 3., 4., 5.], [10., 20., 30., 40., 50.]])   # batched (B, T)
    got = transforms.lag1(x).numpy()
    want = np.array([[1., 1., 2., 3., 4.], [10., 10., 20., 30., 40.]])
    assert got.shape == x.shape, "lag1 must keep the vector length"
    assert np.array_equal(got, want), got


def test_lag1_unbatched():
    got = transforms.lag1(tf.constant([7., 8., 9.])).numpy()
    assert np.array_equal(got, [7., 7., 8.]), got


def test_lag1_feature_is_still_normalized():
    group = {"feature_names": ["md_mei", "md_oni"], "timesteps": [], "normalize": True,
             "transforms": {"md_oni": "lag1"}}
    got = data_norm.get_normalize_list({"input_features": {"md_monthly": group},
                                        "output_features": {"feature_names": [], "normalize": False,
                                                            "transforms": {}, "timesteps": []}})
    assert got == ["md_mei", "md_oni"], got
    group["transforms"] = {"md_oni": "gt0"}
    got = data_norm.get_normalize_list({"input_features": {"md_monthly": group},
                                        "output_features": {"feature_names": [], "normalize": False,
                                                            "transforms": {}, "timesteps": []}})
    assert got == ["md_mei"], got


def test_lag1_via_apply_transforms():
    ex = {"md_oni": tf.constant([[0.5, 1.0, 1.5, 2.0]]), "md_soi": tf.constant([[1., 2., 3., 4.]])}
    out = apply_transforms(ex, {"md_oni": "lag1"}, [])
    assert np.array_equal(out["md_oni"].numpy(), [[0.5, 0.5, 1.0, 1.5]])
    assert np.array_equal(out["md_soi"].numpy(), [[1., 2., 3., 4.]]), "other features untouched"
