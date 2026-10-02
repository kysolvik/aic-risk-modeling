"""Per-fire-type loss weighting and configurable WBCE pos_weight."""

import numpy as np
import tensorflow as tf
import torch

from aic_risk_modeling.train import data_loader, losses

# Background -> 1.0, unlisted fire types -> pos_weight, listed types -> absolute.
TYPE_WEIGHTS = {"3": 45.0, "4": 45.0}
POS_WEIGHT = 9.0


def test_build_type_weight_map_values():
    raw = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    weights = data_loader.build_type_weight_map(raw, TYPE_WEIGHTS, POS_WEIGHT)
    expected = np.array([[1.0, 9.0, 9.0, 45.0, 45.0]], dtype=np.float32)
    assert weights.dtype == tf.float32
    assert np.allclose(weights.numpy(), expected)


def _tiny_dataset(raw_type):
    """A 1-batch dataset of feature dicts with one input + the fire-type band."""
    raw_type = np.asarray(raw_type, dtype=np.int64)[None, ...]  # add example axis
    feat = np.zeros_like(raw_type, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices(
        {"im_feat": tf.constant(feat), "im_viirs_type": tf.constant(raw_type)})
    return ds.batch(1)


_INPUT_CFG = {"im_main": {"feature_names": ["im_feat"], "transforms": {},
                          "timesteps": [], "stack_timesteps": False}}
_OUTPUT_CFG = {"feature_names": ["im_viirs_type"],
               "transforms": {"im_viirs_type": "gt0_bool"},
               "timesteps": [], "stack_timesteps": False}


def test_select_bands_emits_weighted_3tuple():
    raw = [[0, 1, 2], [3, 4, 0], [1, 2, 3]]
    sw_config = {"feature_name": "im_viirs_type", "type_weights": TYPE_WEIGHTS}
    ds = data_loader.select_bands_transform(
        _tiny_dataset(raw), _INPUT_CFG, _OUTPUT_CFG,
        sample_weight_config=sw_config, pos_weight=POS_WEIGHT)

    batch = next(iter(ds.as_numpy_iterator()))
    assert len(batch) == 3, "sample_weight config should yield (inputs, labels, weights)"
    _, labels, weights = batch
    expected_w = np.array([[1, 9, 9], [45, 45, 1], [9, 9, 45]], dtype=np.float32)
    assert np.allclose(weights[0], expected_w)
    assert np.array_equal(labels[0], np.array(raw) > 0)


def test_select_bands_without_config_is_2tuple():
    """Backward compat: predict/probe scripts pass no sample_weight config."""
    ds = data_loader.select_bands_transform(
        _tiny_dataset([[0, 3], [4, 1]]), _INPUT_CFG, _OUTPUT_CFG)
    batch = next(iter(ds.as_numpy_iterator()))
    assert len(batch) == 2, "no sample_weight config should yield (inputs, labels)"


def test_weighted_bce_sample_weight_is_authoritative():
    y_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y_pred = torch.tensor([[0.8, 0.2], [0.3, 0.6]])
    sample_weight = torch.tensor([[45.0, 1.0], [1.0, 9.0]])
    loss_fn = losses.weighted_bce(POS_WEIGHT)

    bce = losses._bce_elementwise(y_true, y_pred)
    expected = (bce * sample_weight).mean()
    got = loss_fn(y_true, y_pred, sample_weight)
    assert torch.allclose(got, expected), "sample_weight should replace the internal pos_weight"

    fallback = loss_fn(y_true, y_pred)
    class_w = y_true * POS_WEIGHT + (1.0 - y_true)
    assert torch.allclose(fallback, (bce * class_w).mean())
    assert not torch.allclose(got, fallback)


def test_binary_losses_accept_sample_weight():
    y_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y_pred = torch.tensor([[0.7, 0.2], [0.4, 0.6]])
    sample_weight = torch.tensor([[45.0, 1.0], [1.0, 9.0]])
    fns = [
        losses.binary_crossentropy,
        losses.weighted_bce(POS_WEIGHT),
        losses.focal(),
        losses.weighted_bce_dice(POS_WEIGHT),
    ]
    for fn in fns:
        out = fn(y_true, y_pred, sample_weight)
        assert torch.isfinite(out), f"{fn} returned non-finite loss"


def test_get_loss_pos_weight_is_configurable():
    # All-positive labels: loss = bce * pos_weight, so the ratio is exact.
    y_true = torch.ones(2, 2)
    y_pred = torch.full((2, 2), 0.5)
    l9 = losses.get_loss("weighted_binary_crossentropy", pos_weight=9.0)
    l20 = losses.get_loss("weighted_binary_crossentropy", pos_weight=20.0)
    ratio = (l20(y_true, y_pred) / l9(y_true, y_pred)).item()
    assert abs(ratio - 20.0 / 9.0) < 1e-5
