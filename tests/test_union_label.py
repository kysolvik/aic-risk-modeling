"""Output 'combine' key: union/intersection of several label bands into one target."""

import numpy as np
import pytest
import tensorflow as tf

from aic_risk_modeling.train import data_loader

# Two overlapping sensors: some pixels seen by both, some by one, some by neither.
MODIS = np.array([[0, 0, 1, 1],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=np.float32)
VIIRS = np.array([[0, 1, 1, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0]], dtype=np.float32)

_INPUT_CFG = {"im_main": {"feature_names": ["im_feat"], "transforms": {},
                          "timesteps": [], "stack_timesteps": False}}


def _dataset(timestep_suffix=""):
    """A 1-batch dataset of feature dicts holding one input + both label bands."""
    feat = np.zeros_like(MODIS)[None, ...]
    return tf.data.Dataset.from_tensor_slices({
        "im_feat": tf.constant(feat),
        f"im_BurnDate{timestep_suffix}": tf.constant(MODIS[None, ...]),
        f"im_viirs_snpp{timestep_suffix}": tf.constant(VIIRS[None, ...]),
    }).batch(1)


def _output_cfg(combine=None, transform="gt0_bool", names=None, timesteps=None):
    cfg = {"feature_names": names or ["im_BurnDate", "im_viirs_snpp"],
           "transforms": {n: transform for n in (names or ["im_BurnDate",
                                                           "im_viirs_snpp"])},
           "timesteps": timesteps if timesteps is not None else [],
           "stack_timesteps": False}
    if combine is not None:
        cfg["combine"] = combine
    return cfg


def _labels(output_cfg, timestep_suffix=""):
    ds = data_loader.select_bands_transform(
        _dataset(timestep_suffix), _INPUT_CFG, output_cfg)
    _, labels = next(iter(ds.as_numpy_iterator()))
    return labels


def test_combine_helper_any_and_all():
    stacked = tf.constant(np.stack([MODIS, VIIRS], axis=-1) > 0)
    union = data_loader._combine_output_bands(stacked, "any").numpy()
    inter = data_loader._combine_output_bands(stacked, "all").numpy()
    assert np.array_equal(union, (MODIS > 0) | (VIIRS > 0))
    assert np.array_equal(inter, (MODIS > 0) & (VIIRS > 0))
    assert union.shape == MODIS.shape


def test_combine_helper_rejects_unknown_mode():
    stacked = tf.constant(np.stack([MODIS, VIIRS], axis=-1) > 0)
    with pytest.raises(ValueError, match='either'):
        data_loader._combine_output_bands(stacked, "either")


def test_union_matches_elementwise_or():
    labels = _labels(_output_cfg(combine="any"))
    assert labels.shape == (1,) + MODIS.shape, "band axis should be reduced away"
    assert np.array_equal(labels[0], (MODIS > 0) | (VIIRS > 0))
    assert labels[0].sum() > (MODIS > 0).sum()
    assert labels[0].sum() > (VIIRS > 0).sum()


def test_intersection_matches_elementwise_and():
    labels = _labels(_output_cfg(combine="all"))
    assert np.array_equal(labels[0], (MODIS > 0) & (VIIRS > 0))


def test_union_dtype_matches_single_band_path():
    """The merged label must be drop-in for the existing single-band output."""
    single = _labels(_output_cfg(names=["im_BurnDate"]))
    union = _labels(_output_cfg(combine="any"))
    assert union.dtype == single.dtype == np.bool_
    assert union.shape == single.shape


def test_union_works_on_float_transform():
    """Combine reduces in bool, so a float transform (gt0) works too."""
    labels = _labels(_output_cfg(combine="any", transform="gt0"))
    assert labels.dtype == np.float32, "dtype should be preserved, not forced to bool"
    assert np.array_equal(labels[0], ((MODIS > 0) | (VIIRS > 0)).astype(np.float32))


def test_union_with_timesteps():
    """The real config uses timesteps [0], which appends _0 to each band name."""
    labels = _labels(_output_cfg(combine="any", timesteps=[0]),
                     timestep_suffix="_0")
    assert np.array_equal(labels[0], (MODIS > 0) | (VIIRS > 0))


def test_no_combine_single_band_unchanged():
    labels = _labels(_output_cfg(names=["im_BurnDate"]))
    assert labels.shape == (1,) + MODIS.shape, "single band should drop the band axis"
    assert np.array_equal(labels[0], MODIS > 0)


def test_no_combine_multi_band_unchanged():
    """Without `combine`, multiple output bands still stack, they don't merge."""
    labels = _labels(_output_cfg())
    assert labels.shape == (1,) + MODIS.shape + (2,), "band axis should be kept"
    assert np.array_equal(labels[0, ..., 0], MODIS > 0)
    assert np.array_equal(labels[0, ..., 1], VIIRS > 0)


def test_union_label_feeds_the_loss():
    torch = pytest.importorskip("torch")
    from aic_risk_modeling.train import losses
    labels = _labels(_output_cfg(combine="any"))
    y_true = torch.from_numpy(labels.astype(np.float32))
    y_pred = torch.full(y_true.shape, 0.5)
    loss = losses.weighted_bce(27.0)(y_true, y_pred)
    assert torch.isfinite(loss), "union label should give a finite weighted BCE"
