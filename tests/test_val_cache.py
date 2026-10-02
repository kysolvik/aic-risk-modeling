"""trainer._cache_dataset_to_disk: on-disk validation cache replay and staleness."""

import os
import tempfile

import numpy as np
import tensorflow as tf

from aic_risk_modeling.train.trainer import _cache_dataset_to_disk


def _counting_dataset(values, counter):
    def gen():
        for v in values:
            counter[0] += 1
            yield np.full((4,), v, dtype=np.float32)
    return tf.data.Dataset.from_generator(
        gen, output_signature=tf.TensorSpec((4,), tf.float32)).batch(2)


def _collect(ds):
    return np.concatenate([b for b in ds.as_numpy_iterator()])


def test_second_pass_replays_from_disk():
    with tempfile.TemporaryDirectory() as d:
        counter = [0]
        ds = _cache_dataset_to_disk(_counting_dataset(range(6), counter), d)
        first = _collect(ds)
        assert counter[0] == 6
        second = _collect(ds)
        third = _collect(ds)
        assert counter[0] == 6, "upstream pipeline re-ran after epoch 1"
        assert np.array_equal(first, second) and np.array_equal(first, third)


def test_cache_is_written_to_disk():
    with tempfile.TemporaryDirectory() as d:
        ds = _cache_dataset_to_disk(_counting_dataset(range(6), [0]), d)
        _collect(ds)
        files = os.listdir(d)
        assert any(f.startswith('val') and '.data' in f for f in files), files


def test_stale_cache_from_earlier_run_is_not_served():
    with tempfile.TemporaryDirectory() as d:
        old = _cache_dataset_to_disk(_counting_dataset([1, 1, 1, 1], [0]), d)
        _collect(old)
        new = _cache_dataset_to_disk(_counting_dataset([7, 7, 7, 7], [0]), d)
        assert np.all(_collect(new) == 7), "stale cache leaked into a new run"


def test_creates_missing_cache_dir():
    with tempfile.TemporaryDirectory() as d:
        sub = os.path.join(d, 'nested', 'val_cache')
        ds = _cache_dataset_to_disk(_counting_dataset(range(4), [0]), sub)
        _collect(ds)
        assert os.path.isdir(sub)
