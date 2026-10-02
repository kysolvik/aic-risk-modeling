"""data_loader: tf.data input order is reproducible under a fixed seed."""

import os
import tempfile

import tensorflow as tf

from aic_risk_modeling.train import data_loader

SCHEMA_PBTXT = """
feature {
  name: "id"
  type: INT
  shape { dim { size: 1 } }
}
"""

N_FILES = 3  # spread examples across files so list_files/interleave order matters


def _write_dir(path, ids):
    """Write a dir of GZIP TFRecords plus a schema.pbtxt the loader can read."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "schema.pbtxt"), "w") as f:
        f.write(SCHEMA_PBTXT)
    opts = tf.io.TFRecordOptions(compression_type="GZIP")
    writers = [
        tf.io.TFRecordWriter(os.path.join(path, f"data-{i}.tfrecord.gz"), opts)
        for i in range(N_FILES)
    ]
    for k, i in enumerate(ids):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={"id": tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))}
            )
        )
        writers[k % N_FILES].write(example.SerializeToString())
    for w in writers:
        w.close()


def _make_dirs(root):
    d0, d1 = os.path.join(root, "a"), os.path.join(root, "b")
    _write_dir(d0, range(0, 100))
    _write_dir(d1, range(100, 200))
    return [d0, d1]


def _order(data_dirs, seed):
    """Return the sequence of `id`s produced by a seeded merged dataset."""
    ds = data_loader.build_merged_dataset(
        data_dirs=data_dirs,
        tfrecord_pattern="*.tfrecord.gz",
        shuffle=True,
        batch_size=1,
        seed=seed,
    )
    return [int(batch["id"][0][0]) for batch in ds]


def test_same_seed_is_reproducible():
    tf.random.set_seed(54)  # mirror trainer.py
    with tempfile.TemporaryDirectory() as root:
        dirs = _make_dirs(root)
        first = _order(dirs, seed=54)
        second = _order(dirs, seed=54)
    assert sorted(first) == list(range(200)), "every example should appear exactly once"
    assert first == second, "same seed must give an identical data order"


def test_different_seed_changes_order():
    tf.random.set_seed(54)
    with tempfile.TemporaryDirectory() as root:
        dirs = _make_dirs(root)
        base = _order(dirs, seed=54)
        other = _order(dirs, seed=999)
    assert sorted(other) == list(range(200)), "every example should appear exactly once"
    assert base != other, "a different seed should give a different data order"
