"""patch_features: band patching keyed by md_id and stats.pbtxt recomputation."""

import glob
import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2, statistics_pb2

import patch_features as pf
import pool_stats_pbtxt as pool
from aic_risk_modeling.train import data_norm as dn

_REPO = os.path.join(os.path.dirname(__file__), "..")

WIDTH = 16
IDS = list(range(100, 106))
RNG = np.random.default_rng(0)
KEEP = {i: RNG.normal(5, 2, WIDTH).astype(np.float32) for i in IDS}
UPD_OLD = {i: RNG.normal(0, 1, WIDTH).astype(np.float32) for i in IDS}
UPD_NEW = {i: RNG.normal(10, 3, WIDTH).astype(np.float32) for i in IDS}
NEW = {i: RNG.integers(0, 7, WIDTH).astype(np.int64) for i in IDS}


def _example(md_id, **bands):
    feats = {"md_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[md_id]))}
    for name, v in bands.items():
        if np.issubdtype(v.dtype, np.integer):
            feats[name] = tf.train.Feature(int64_list=tf.train.Int64List(value=v.tolist()))
        else:
            feats[name] = tf.train.Feature(float_list=tf.train.FloatList(value=v.tolist()))
    return tf.train.Example(features=tf.train.Features(feature=feats))


def _write_shards(directory, groups):
    os.makedirs(directory, exist_ok=True)
    opts = tf.io.TFRecordOptions(compression_type="GZIP")
    for k, examples in enumerate(groups):
        path = os.path.join(directory, f"part-{k:05d}.tfrecord.gz")
        with tf.io.TFRecordWriter(path, opts) as w:
            for ex in examples:
                w.write(ex.SerializeToString())


def _make_fixture(root):
    year = os.path.join(root, "src", "allpreds_2020")
    corrected = os.path.join(root, "corrected")
    _write_shards(year, [
        [_example(i, im_keep=KEEP[i], im_upd=UPD_OLD[i]) for i in IDS[:3]],
        [_example(i, im_keep=KEEP[i], im_upd=UPD_OLD[i]) for i in IDS[3:]],
    ])
    # Reversed order and a different shard split: patching must key on md_id.
    rev = IDS[::-1]
    _write_shards(corrected, [
        [_example(i, im_upd=UPD_NEW[i], im_new=NEW[i]) for i in rev[:4]],
        [_example(i, im_upd=UPD_NEW[i], im_new=NEW[i]) for i in rev[4:]],
    ])

    schema = schema_pb2.Schema()
    for name, t, w in [("md_id", schema_pb2.INT, 1), ("im_keep", schema_pb2.FLOAT, WIDTH),
                       ("im_upd", schema_pb2.FLOAT, WIDTH)]:
        f = schema.feature.add()
        f.name, f.type = name, t
        f.shape.dim.add().size = w
    with open(os.path.join(year, "schema.pbtxt"), "w") as f:
        f.write(text_format.MessageToString(schema))

    sl = statistics_pb2.DatasetFeatureStatisticsList()
    ds = sl.datasets.add()
    ds.num_examples = len(IDS)
    ds.features.add().CopyFrom(pf.feature_stats_proto(
        "im_keep", np.stack([KEEP[i] for i in IDS]), "float_list", len(IDS), WIDTH))
    ds.features.add().CopyFrom(pf.feature_stats_proto(
        "im_upd", np.stack([UPD_OLD[i] for i in IDS]), "float_list", len(IDS), WIDTH))
    ds.features.add().CopyFrom(pf.feature_stats_proto(
        "md_id", np.array(IDS), "int64_list", len(IDS), 1))
    with open(os.path.join(year, "stats.pbtxt"), "w") as f:
        f.write(text_format.MessageToString(sl))
    return year, corrected


def _run(root, year, corrected, *extra):
    pf.main(["--corrected_dir", corrected, "--data_dirs", year,
             "--output_root", os.path.join(root, "out"),
             "--features", "im_upd,im_new", *extra])
    return os.path.join(root, "out", "allpreds_2020")


def _feature_map(path):
    sl = dn.load_stats_from_text(path)
    return [f.path.step[0] for f in sl.datasets[0].features], \
        {f.path.step[0]: f for f in sl.datasets[0].features}


def test_end_to_end_stats_patch():
    with tempfile.TemporaryDirectory() as root:
        year, corrected = _make_fixture(root)
        out = _run(root, year, corrected)

        for path in sorted(glob.glob(os.path.join(out, "*.tfrecord.gz"))):
            for ex in pf._examples(path):
                f = ex.features.feature
                i = f["md_id"].int64_list.value[0]
                np.testing.assert_array_equal(f["im_upd"].float_list.value, UPD_NEW[i])
                np.testing.assert_array_equal(f["im_keep"].float_list.value, KEEP[i])
                np.testing.assert_array_equal(f["im_new"].int64_list.value, NEW[i])

        order, feats = _feature_map(os.path.join(out, "stats.pbtxt"))
        assert order == ["im_keep", "im_upd", "md_id", "im_new"], order
        _, src_feats = _feature_map(os.path.join(year, "stats.pbtxt"))
        for name in ("im_keep", "md_id"):
            assert (feats[name].SerializeToString()
                    == src_feats[name].SerializeToString()), name

        upd = np.stack([UPD_NEW[i] for i in IDS]).astype(np.float64)
        ns = feats["im_upd"].num_stats
        assert np.isclose(ns.mean, upd.mean()) and np.isclose(ns.std_dev, upd.std())
        assert ns.common_stats.tot_num_values == upd.size
        assert ns.common_stats.num_non_missing == len(IDS)
        new = np.stack([NEW[i] for i in IDS]).astype(np.float64)
        assert feats["im_new"].type == statistics_pb2.FeatureNameStatistics.INT
        assert feats["im_new"].num_stats.num_zeros == int((new == 0).sum())

        sl = dn.load_stats_from_text(os.path.join(out, "stats.pbtxt"))
        got = dn.get_norm_stats(sl, "im_upd")
        assert np.isclose(got["median"], np.median(upd))
        edges = np.quantile(upd, np.linspace(0, 1, 11))
        q25, q75 = np.interp([2.5, 7.5], np.arange(11), edges)
        assert np.isclose(got["robust_scale"], (q75 - q25) / 1.349), got
        pooled = pool.pool_stats([out])["features"]
        assert np.isclose(pooled["im_upd"]["mean"], upd.mean())
        assert np.isclose(pooled["im_new"]["mean"], new.mean())

        schema = schema_pb2.Schema()
        with open(os.path.join(out, "schema.pbtxt")) as f:
            text_format.Parse(f.read(), schema)
        assert "im_new" in {f.name for f in schema.feature}


def test_partial_and_dry_runs_write_no_stats():
    with tempfile.TemporaryDirectory() as root:
        year, corrected = _make_fixture(root)
        out = _run(root, year, corrected, "--limit_shards", "1")
        assert glob.glob(os.path.join(out, "*.tfrecord.gz"))
        assert not os.path.exists(os.path.join(out, "stats.pbtxt"))
    with tempfile.TemporaryDirectory() as root:
        year, corrected = _make_fixture(root)
        out = _run(root, year, corrected, "--dry_run")
        assert not os.path.exists(out)


def test_refuses_in_place():
    with tempfile.TemporaryDirectory() as root:
        year, corrected = _make_fixture(root)
        with pytest.raises(ValueError, match='in place'):
            pf.main(["--corrected_dir", corrected, "--data_dirs", year,
                     "--output_root", os.path.dirname(year)])


_TFDV_DIR = os.path.join(_REPO, "..", "data", "monthly_test_2023")


def test_matches_real_tfdv_output():
    """Our proto vs tfdv's own stats.pbtxt for a band of a real local export."""
    shards = sorted(glob.glob(os.path.join(_TFDV_DIR, "*.tfrecord.gz")))
    if not shards:
        pytest.skip("no local tfdv export")
    _, tfdv = _feature_map(os.path.join(_TFDV_DIR, "stats.pbtxt"))
    first = next(pf._examples(shards[0])).features.feature
    names = [n for n in sorted(first) if n in tfdv
             and first[n].WhichOneof("kind") == "float_list"][:5]
    assert names
    for name in names:
        rows = [np.asarray(ex.features.feature[name].float_list.value)
                for s in shards for ex in pf._examples(s)]
        ref = tfdv[name]
        mine = pf.feature_stats_proto(name, np.stack(rows), "float_list",
                                      len(rows), len(rows[0]))
        a, b = mine.num_stats, ref.num_stats
        assert a.common_stats.tot_num_values == b.common_stats.tot_num_values, name
        assert a.common_stats.num_non_missing == b.common_stats.num_non_missing, name
        assert a.num_zeros == b.num_zeros, name
        for field in ("mean", "std_dev", "min", "max"):
            assert np.isclose(getattr(a, field), getattr(b, field),
                              rtol=1e-4, atol=1e-6), (name, field)
        spread = b.max - b.min
        assert abs(a.median - b.median) < 0.02 * spread, (name, a.median, b.median)
        assert [h.type for h in a.histograms] == [h.type for h in b.histograms]
        qa, qb = a.histograms[1].buckets, b.histograms[1].buckets
        assert len(qa) == len(qb)
        for x, y in zip(qa, qb):
            assert abs(x.high_value - y.high_value) < 0.02 * spread, name
