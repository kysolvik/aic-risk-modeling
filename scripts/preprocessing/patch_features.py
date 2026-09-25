"""Replace feature bands inside existing allpreds_* tfrecords.

Static fields (elevation, slope, accessibility, governance, ...) are duplicated
into every year's export, so re-exporting them means rewriting every year even
though the corrected data is one grid's worth of rasters. This script does that
rewrite: it reads a corrected export, keys it by `md_id`, then streams each
year's shards replacing only the named features and copying every other
feature through byte-for-byte. A band the year export never had at all is
*added* rather than an error, which is how a new field gets backfilled.

Year-varying (dynamic) fields work the same way, one run per year: pass that
year's corrected export as --corrected_dir and that year's dir as --data_dirs.

Why key on `md_id`: Beam gives no ordering guarantee, so record N of a shard in
the corrected export is not record N of the same-named shard in a year export.
Position-based zipping silently misaligns tiles. `md_x`/`md_y` are floats and a
poor key; `md_id` is an int64 and exact.

The source directories are never modified -- output goes to
`<output_root>/<source dir name>/<same shard name>`.

Cost: measured ~16 MB/s uncompressed single-threaded (gzip re-compression
dominates), i.e. ~1 s per 17 MB record. With `--workers 8` a ~1800-tile year
takes a few minutes. Shards are independent, so workers scale ~linearly.

Sidecars written to each output dir:
  * schema.pbtxt (+ schema.json) -- copied from the source, extended with any
    band this run added (`--skip_schema` to opt out). `data_loader` builds its
    parse spec from that file per-dir, so an output dir without one is
    unreadable and one that omits an added band ignores it.
  * stats.pbtxt -- the source's tfdv stats with the patched bands' entries
    recomputed from the corrected values (replaced in place if the band
    existed, appended if it is new); every other entry is copied untouched
    (`--skip_stats` to opt out). Quantiles are exact rather than sketched. No
    second pass over the data is needed: the patched values are exactly the
    corrected table rows matched in that dir. Stats are skipped for a dir that
    was only partially patched (--limit_shards, or unmatched md_ids under
    --allow_missing) rather than written wrong.

AFTER RUNNING: re-pool the per-year stats into the training stats JSON with
pool_stats_pbtxt.py (training reads `config['stats_path']`).

Examples:
    # Preview coverage, what would change, and the stats diff; writes nothing.
    python scripts/preprocessing/patch_features.py \\
        --corrected_dir gs://aic-amazon/data/corrected_2018/ \\
        --data_dirs gs://aic-amazon/data/fullgrid_v3/allpreds_2018/ \\
        --output_root gs://aic-amazon/data/fullgrid_v4/ \\
        --features im_foo,im_bar --dry_run

    # Patch each year from its own corrected export, 8 shards at a time.
    for y in 2013 2014 ...; do
      python scripts/preprocessing/patch_features.py \\
          --corrected_dir gs://aic-amazon/data/corrected_$y/ \\
          --data_dirs gs://aic-amazon/data/fullgrid_v3/allpreds_$y/ \\
          --output_root gs://aic-amazon/data/fullgrid_v4/ \\
          --features im_foo,im_bar --workers 8
    done
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
from collections import defaultdict

import numpy as np
import tensorflow as tf

try:  # needed for the schema/stats sidecars
    from google.protobuf import text_format
    from tensorflow_metadata.proto.v0 import schema_pb2, statistics_pb2
except ImportError:  # pragma: no cover - environment without tfmd
    text_format = schema_pb2 = statistics_pb2 = None

DEFAULT_HIST_BUCKETS = 10  # tfdv's default for both STANDARD and QUANTILES


def _compression(path):
    return "GZIP" if path.endswith(".gz") else ""


def _examples(path):
    """Yield every tf.train.Example in one tfrecord file."""
    for rec in tf.data.TFRecordDataset([path], compression_type=_compression(path)):
        yield tf.train.Example.FromString(rec.numpy())


def _first_example(path):
    for example in _examples(path):
        return example
    raise ValueError(f"{path} is empty")


def _list_shards(directory, pattern="*.tfrecord.gz"):
    shards = sorted(tf.io.gfile.glob(os.path.join(directory, pattern)))
    if not shards:
        raise ValueError(f"no shards matching {pattern!r} in {directory}")
    return shards


def _md_id(feats):
    return str(feats["md_id"].int64_list.value[0])


def _is_int(kind):
    return kind == "int64_list"


def _require_tfmd(what):
    if statistics_pb2 is None:
        raise ImportError(f"tensorflow_metadata is required to {what}")


def _feature_values(feature):
    """The repeated-value list of whichever oneof a tf.train.Feature holds."""
    kind = feature.WhichOneof("kind")
    if kind is None:
        return None, None
    return kind, getattr(feature, kind).value


def _as_kind(values, kind):
    """A float32 row cast back to the python type `kind`'s repeated field takes.

    The corrected table is float32 for every band, but protobuf type-checks
    repeated fields: extending an `int64_list` with python floats raises.
    """
    if _is_int(kind):
        return [int(round(float(v))) for v in values.tolist()]
    if kind == "float_list":
        return [float(v) for v in values.tolist()]
    raise ValueError(f"cannot write a numeric band into a {kind}")


def _set_values(feature, kind, values):
    """Write `values` into `feature`, replacing whatever it held."""
    target = getattr(feature, kind).value
    del target[:]
    target.extend(_as_kind(values, kind))


# ---------------------------------------------------------------------------
# Corrected table
# ---------------------------------------------------------------------------

def build_corrected_table(corrected_dir, features, cache_dir, pattern):
    """Materialize the corrected bands as an mmap-able array.

    Returns (table_path, index_path, meta). The table is a raw float32 file of
    shape meta["shape"] = (n_tiles, n_features, width), opened with
    `_open_table`; meta["index"] maps str(md_id) -> row, and meta["kinds"]
    records each band's protobuf oneof kind, which is the only way to know what
    type a band added to a year export should have. Each tile is appended to
    disk as it is read, so peak memory is one record regardless of band count
    (a full-year export is ~5 GB -- holding it in RAM got the process killed),
    and worker processes share one copy via the page cache.
    """
    shards = _list_shards(corrected_dir, pattern)
    if not features:
        features = sorted(k for k in _first_example(shards[0]).features.feature
                          if k.startswith("im_"))
        print(f"inferred corrected features ({len(features)}): "
              f"{', '.join(features)}")

    os.makedirs(cache_dir, exist_ok=True)
    table_path = os.path.join(cache_dir, "corrected_table.f32")
    index_path = os.path.join(cache_dir, "corrected_index.json")
    out = open(table_path, "wb")
    index, width, kinds = {}, None, {}
    for shard in shards:
        for example in _examples(shard):
            feats = example.features.feature
            if "md_id" not in feats:
                raise ValueError(f"{shard}: corrected record has no md_id")
            md_id = _md_id(feats)
            if md_id in index:
                raise ValueError(
                    f"duplicate md_id {md_id} in the corrected export -- the "
                    f"table would be ambiguous")
            band = []
            for name in features:
                if name not in feats:
                    raise ValueError(
                        f"corrected export is missing {name!r} (tile {md_id})")
                kind, values = _feature_values(feats[name])
                if kind not in ("float_list", "int64_list"):
                    raise ValueError(
                        f"{name!r} is a {kind} on tile {md_id}; only numeric "
                        f"bands can be patched")
                if kinds.setdefault(name, kind) != kind:
                    raise ValueError(
                        f"{name!r} is a {kind} on tile {md_id} but a "
                        f"{kinds[name]} on an earlier tile")
                if width is None:
                    width = len(values)
                if len(values) != width:
                    raise ValueError(
                        f"{name!r} on tile {md_id} has length {len(values)}, "
                        f"expected {width}")
                band.append(np.asarray(values, dtype=np.float32))
            index[md_id] = len(index)
            out.write(np.stack(band).tobytes())
        print(f"  read {shard} ({len(index)} tiles so far)")
    out.close()
    if not index:
        raise ValueError(f"corrected export {corrected_dir} is empty")

    meta = {"index": index, "features": list(features), "width": width,
            "kinds": kinds, "shape": [len(index), len(features), width]}
    with open(index_path, "w") as f:
        json.dump(meta, f)
    table = _open_table(table_path, meta)
    for i, name in enumerate(features):
        if _is_int(kinds[name]):
            peak = float(np.abs(table[:, i]).max())
            if peak > 2 ** 24:
                print(f"  WARNING: {name} is int64 and reaches {peak:.0f}; "
                      f"the float32 table cannot round-trip it exactly")
    print(f"corrected table: {table.shape} tiles x features x px "
          f"({table.nbytes / 1e6:.0f} MB on disk) -> {table_path}")
    return table_path, index_path, meta


def _open_table(table_path, meta):
    """Read-only memmap of the table `build_corrected_table` wrote."""
    return np.memmap(table_path, dtype=np.float32, mode="r",
                     shape=tuple(meta["shape"]))


# ---------------------------------------------------------------------------
# Shard patching (runs in workers)
# ---------------------------------------------------------------------------

_WORKER = {}


def _init_worker(table_path, index_path):
    with open(index_path) as f:
        meta = json.load(f)
    _WORKER["table"] = _open_table(table_path, meta)
    _WORKER["index"] = meta["index"]
    _WORKER["features"] = meta["features"]
    _WORKER["kinds"] = meta["kinds"]


def patch_shard(job):
    """Rewrite one shard with the corrected bands replaced. Returns a stat dict.

    `rows` lists the table row written into each matched record, so the
    caller can compute the patched bands' stats without re-reading the output.
    """
    src, dst, dry_run, allow_missing = job
    table, index = _WORKER["table"], _WORKER["index"]
    features, kinds = _WORKER["features"], _WORKER["kinds"]
    stats = {"shard": os.path.basename(src), "records": 0, "missing": 0,
             "rows": [],
             "changed": {name: 0 for name in features},
             "added": {name: 0 for name in features}}

    writer = None
    if not dry_run:
        tf.io.gfile.makedirs(os.path.dirname(dst))
        writer = tf.io.TFRecordWriter(
            dst, tf.io.TFRecordOptions(compression_type=_compression(dst)))
    try:
        for example in _examples(src):
            feats = example.features.feature
            stats["records"] += 1
            md_id = _md_id(feats)
            row = index.get(md_id)
            if row is None:
                stats["missing"] += 1
                if not allow_missing:
                    raise KeyError(
                        f"{src}: md_id {md_id} is absent from the corrected "
                        f"export (pass --allow_missing to copy such records "
                        f"through unpatched)")
            else:
                stats["rows"].append(row)
                for i, name in enumerate(features):
                    new = table[row, i]
                    if name not in feats:
                        # A band the year export never had: add it, taking the
                        # value type from the corrected export since there is
                        # no target field to read it off.
                        _set_values(feats[name], kinds[name], new)
                        stats["added"][name] += 1
                        continue
                    kind, values = _feature_values(feats[name])
                    if len(values) != len(new):
                        raise ValueError(
                            f"{src}: {name!r} length {len(values)} != corrected "
                            f"length {len(new)} on tile {md_id}")
                    if not np.array_equal(np.asarray(values, dtype=np.float32),
                                          new):
                        stats["changed"][name] += 1
                    _set_values(feats[name], kind, new)
            if writer is not None:
                writer.write(example.SerializeToString())
    finally:
        if writer is not None:
            writer.close()
    return stats


def verify_shard(src, dst, features):
    """Assert the first record of `dst` differs from `src` only in `features`.

    A band that `src` lacks entirely counts as a difference: patching may
    legitimately introduce a feature the year export never had, but only one
    that was asked for. Nothing may ever disappear.
    """
    fa = _first_example(src).features.feature
    fb = _first_example(dst).features.feature
    dropped = set(fa) - set(fb)
    if dropped:
        raise AssertionError(f"{dst}: features disappeared: {sorted(dropped)}")
    differing = (set(fb) - set(fa)) | {
        k for k in fa if fa[k].SerializeToString() != fb[k].SerializeToString()}
    unexpected = differing - set(features)
    if unexpected:
        raise AssertionError(f"{dst}: unexpected features changed: {unexpected}")
    return differing


def features_missing_from(directory, features, pattern):
    """Which of `features` the first record of `directory` does not carry."""
    present = _first_example(_list_shards(directory, pattern)[0]).features.feature
    return [name for name in features if name not in present]


# ---------------------------------------------------------------------------
# Sidecars: schema.pbtxt / schema.json / stats.pbtxt
# ---------------------------------------------------------------------------

def _read_pbtxt(path, message):
    with tf.io.gfile.GFile(path) as f:
        text_format.Parse(f.read(), message)
    return message


def _write_text(path, text):
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(text)


def write_schema(src_dir, dst_dir, added, kinds, width, dry_run):
    """Copy the schema sidecars to `dst_dir`, declaring the `added` bands.

    `data_loader.load_schema_from_gcs` builds the parse spec from the
    schema.pbtxt sitting next to the shards, so an output dir without one is
    unreadable, and one that omits a band this run added leaves that band
    unparsed no matter that it is in the records.
    """
    src_schema = os.path.join(src_dir, "schema.pbtxt")
    dst_schema = os.path.join(dst_dir, "schema.pbtxt")
    note = f" (+{', '.join(added)})" if added else " (copy)"
    if not tf.io.gfile.exists(src_schema):
        print(f"  no schema.pbtxt in {src_dir} -- the patched dir will need "
              f"one before training can read it")
        return
    if dry_run:
        print(f"  would write {dst_schema}{note}")
        return
    tf.io.gfile.makedirs(dst_dir)
    if not added:
        tf.io.gfile.copy(src_schema, dst_schema, overwrite=True)
    else:
        _require_tfmd("declare added bands in schema.pbtxt; pass --skip_schema "
                      "and edit the schema by hand otherwise")
        schema = _read_pbtxt(src_schema, schema_pb2.Schema())
        present = {f.name for f in schema.feature}
        for name in added:
            if name in present:
                continue
            feature = schema.feature.add()
            feature.name = name
            feature.type = (schema_pb2.FeatureType.INT if _is_int(kinds[name])
                            else schema_pb2.FeatureType.FLOAT)
            feature.presence.min_fraction = 1.0
            feature.presence.min_count = 1
            feature.shape.dim.add().size = width
        _write_text(dst_schema, text_format.MessageToString(schema))
    print(f"  wrote {dst_schema}{note}")

    src_json = os.path.join(src_dir, "schema.json")
    dst_json = os.path.join(dst_dir, "schema.json")
    if not tf.io.gfile.exists(src_json):
        return
    if not added:
        tf.io.gfile.copy(src_json, dst_json, overwrite=True)
        return
    with tf.io.gfile.GFile(src_json) as f:
        meta = json.load(f)
    for name in added:
        meta.setdefault("features", {}).setdefault(
            name, "int64" if _is_int(kinds[name]) else "float")
    with tf.io.gfile.GFile(dst_json, "w") as f:
        json.dump(meta, f, indent=2)


def _quantile_bucket_count(stats_list):
    """Bucket count of the first QUANTILES histogram in an existing stats file."""
    for dataset in stats_list.datasets:
        for feature in dataset.features:
            for hist in feature.num_stats.histograms:
                if hist.type == statistics_pb2.Histogram.QUANTILES and hist.buckets:
                    return len(hist.buckets)
    return DEFAULT_HIST_BUCKETS


def feature_stats_proto(name, values, kind, n_records, width,
                        n_buckets=DEFAULT_HIST_BUCKETS):
    """A tfdv-layout FeatureNameStatistics for one band's patched values.

    `values` is every value written for the band in the dir (any shape).
    Mirrors what tfdv emits for a fixed-length numeric feature: common_stats
    with a QUANTILES num_values_histogram, moments, then a STANDARD
    (equal-width) and a QUANTILES (equal-count) histogram. The quantiles are
    exact rather than sketched. Only finite values count, as in data_stats.
    """
    _require_tfmd("write stats.pbtxt; pass --skip_stats to opt out")
    v = np.asarray(values, dtype=np.float64).ravel()
    if _is_int(kind):
        v = np.round(v)  # what _as_kind actually wrote
    v = v[np.isfinite(v)]

    fs = statistics_pb2.FeatureNameStatistics()
    fs.path.step.append(name)
    fs.type = (statistics_pb2.FeatureNameStatistics.INT if _is_int(kind)
               else statistics_pb2.FeatureNameStatistics.FLOAT)
    num = fs.num_stats
    common = num.common_stats
    common.num_non_missing = n_records
    common.min_num_values = width
    common.max_num_values = width
    common.avg_num_values = float(width)
    common.tot_num_values = v.size
    nvh = common.num_values_histogram
    nvh.type = statistics_pb2.Histogram.QUANTILES
    for _ in range(n_buckets):
        b = nvh.buckets.add()
        b.low_value = b.high_value = float(width)
        b.sample_count = n_records / n_buckets
    if v.size == 0:
        return fs

    num.mean = float(v.mean())
    num.std_dev = float(v.std())
    num.num_zeros = int(np.count_nonzero(v == 0))
    num.min = float(v.min())
    num.max = float(v.max())
    num.median = float(np.median(v))

    counts, edges = np.histogram(v, bins=n_buckets)
    standard = num.histograms.add()
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        b = standard.buckets.add()
        b.low_value, b.high_value, b.sample_count = float(lo), float(hi), float(c)

    qedges = np.quantile(v, np.linspace(0, 1, n_buckets + 1))
    quantiles = num.histograms.add()
    quantiles.type = statistics_pb2.Histogram.QUANTILES
    for lo, hi in zip(qedges[:-1], qedges[1:]):
        b = quantiles.buckets.add()
        b.low_value, b.high_value = float(lo), float(hi)
        b.sample_count = v.size / n_buckets
    return fs


def write_stats(src_dir, dst_dir, rows, table, meta, n_records, dry_run):
    """Write `dst_dir/stats.pbtxt`: the source's, with patched bands recomputed.

    Existing entries for patched bands are replaced in place (order kept); new
    bands are appended. Every other entry is carried over untouched.
    """
    src_stats = os.path.join(src_dir, "stats.pbtxt")
    dst_stats = os.path.join(dst_dir, "stats.pbtxt")
    if not tf.io.gfile.exists(src_stats):
        print(f"  no stats.pbtxt in {src_dir} -- skipping stats")
        return
    _require_tfmd("write stats.pbtxt; pass --skip_stats to opt out")
    stats_list = _read_pbtxt(src_stats, statistics_pb2.DatasetFeatureStatisticsList())
    if len(stats_list.datasets) != 1:
        raise ValueError(f"{src_stats}: expected 1 dataset, found "
                         f"{len(stats_list.datasets)}")
    dataset = stats_list.datasets[0]
    n_buckets = _quantile_bucket_count(stats_list)
    by_name = {f.path.step[0]: f for f in dataset.features if f.path.step}

    rows = np.sort(np.asarray(rows))
    for i, name in enumerate(meta["features"]):
        new = feature_stats_proto(name, table[rows, i], meta["kinds"][name],
                                  n_records, meta["width"], n_buckets)
        old = by_name.get(name)
        if old is None:
            dataset.features.add().CopyFrom(new)
            print(f"  {name:24s} added     mean {new.num_stats.mean:.6g}  "
                  f"std {new.num_stats.std_dev:.6g}  "
                  f"median {new.num_stats.median:.6g}")
        else:
            o = old.num_stats
            before = (o.mean, o.std_dev, o.median)  # CopyFrom overwrites `o`
            old.CopyFrom(new)
            print(f"  {name:24s} updated   mean {before[0]:.6g} -> "
                  f"{new.num_stats.mean:.6g}  std {before[1]:.6g} -> "
                  f"{new.num_stats.std_dev:.6g}  median {before[2]:.6g} -> "
                  f"{new.num_stats.median:.6g}")
    if dry_run:
        print(f"  would write {dst_stats}")
        return
    tf.io.gfile.makedirs(dst_dir)
    _write_text(dst_stats, text_format.MessageToString(stats_list))
    print(f"  wrote {dst_stats}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def plan_jobs(args, features):
    """Per-dir (src, out, bands to add) plus the flat list of shard jobs."""
    dirs, jobs = [], []
    for data_dir in args.data_dirs:
        out_dir = os.path.join(args.output_root,
                               os.path.basename(data_dir.rstrip("/")))
        if out_dir.rstrip("/") == data_dir.rstrip("/"):
            raise ValueError(f"output dir {out_dir} is the source dir; "
                             f"refusing to patch in place")
        shards = _list_shards(data_dir, args.tfrecord_pattern)
        if args.limit_shards:
            shards = shards[:args.limit_shards]
        dirs.append((data_dir, out_dir,
                     features_missing_from(data_dir, features,
                                           args.tfrecord_pattern)))
        jobs.extend((shard, os.path.join(out_dir, os.path.basename(shard)),
                     args.dry_run, args.allow_missing) for shard in shards)
    return dirs, jobs


def run_jobs(jobs, table_path, index_path, workers):
    if workers > 1:
        with mp.Pool(workers, initializer=_init_worker,
                     initargs=(table_path, index_path)) as pool:
            return pool.map(patch_shard, jobs)
    _init_worker(table_path, index_path)
    return [patch_shard(job) for job in jobs]


def report(results, features, additions):
    records = sum(r["records"] for r in results)
    missing = sum(r["missing"] for r in results)
    print(f"\npatched {records} records ({missing} unmatched md_id)")
    for name in features:
        changed = sum(r["changed"][name] for r in results)
        added = sum(r["added"][name] for r in results)
        line = f"  {name:24s} changed in {changed:6d} / {records} records"
        if added:
            line += f", added to {added}"
        elif changed == 0:
            line += ("  <- IDENTICAL everywhere; the corrected export did not "
                     "actually change this band")
        print(line)
    if missing and additions:
        added_names = sorted({n for v in additions for n in v})
        print(f"\n!! {missing} records were copied through unpatched and so "
              f"carry no {', '.join(added_names)}. Parsing the patched dirs "
              f"with a schema that declares those bands WILL FAIL on those "
              f"records; drop those tiles or extend the corrected export.")


def write_sidecars(args, dirs, jobs, results, table_path, meta):
    per_dir = defaultdict(lambda: {"rows": [], "records": 0, "missing": 0})
    for (_, dst, _, _), r in zip(jobs, results):
        d = per_dir[os.path.dirname(dst)]
        d["rows"].extend(r["rows"])
        d["records"] += r["records"]
        d["missing"] += r["missing"]

    table = _open_table(table_path, meta) if not args.skip_stats else None
    for src_dir, out_dir, added in dirs:
        print(f"\n{out_dir}")
        if not args.skip_schema:
            write_schema(src_dir, out_dir, added, meta["kinds"], meta["width"],
                         args.dry_run)
        if args.skip_stats:
            continue
        d = per_dir[out_dir]
        if args.limit_shards:
            print("  --limit_shards: dir only partially patched -- skipping stats")
        elif d["missing"]:
            print(f"  {d['missing']} records unmatched and not patched -- "
                  f"skipping stats (they would not reflect those records)")
        else:
            write_stats(src_dir, out_dir, d["rows"], table, meta, d["records"],
                        args.dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrected_dir", required=True,
                        help="directory of corrected tfrecords, keyed by md_id")
    parser.add_argument("--data_dirs", nargs="+", required=True,
                        help="year directories to patch (never modified)")
    parser.add_argument("--output_root", required=True,
                        help="patched years are written to <root>/<dir name>/")
    parser.add_argument("--features", default=None,
                        help="comma-separated bands to replace/add; default: "
                             "every im_* band in the corrected export")
    parser.add_argument("--tfrecord_pattern", default="*.tfrecord.gz")
    parser.add_argument("--cache_dir", default=None,
                        help="local dir to materialize the corrected table in "
                             "(default: a fresh temp dir, removed on exit)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit_shards", type=int, default=None,
                        help="patch only the first N shards per year (for "
                             "testing; stats are then skipped)")
    parser.add_argument("--allow_missing", action="store_true",
                        help="copy records whose md_id is absent from the corrected "
                             "export through unpatched instead of failing")
    parser.add_argument("--skip_schema", action="store_true",
                        help="do not copy/patch schema.pbtxt into the output dirs")
    parser.add_argument("--skip_stats", action="store_true",
                        help="do not write a patched stats.pbtxt into the output dirs")
    parser.add_argument("--dry_run", action="store_true",
                        help="report coverage, what would change and the stats "
                             "diff; write nothing")
    args = parser.parse_args(argv)

    features = ([f.strip() for f in args.features.split(",")]
                if args.features else None)
    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="patch_features_")
    try:
        table_path, index_path, meta = build_corrected_table(
            args.corrected_dir, features, cache_dir, args.tfrecord_pattern)
        features = meta["features"]

        dirs, jobs = plan_jobs(args, features)
        additions = [added for _, _, added in dirs if added]
        print(f"{len(jobs)} shards across {len(dirs)} dirs; "
              f"{len(meta['index'])} tiles in the corrected table; "
              f"features: {', '.join(features)}")
        for src_dir, _, added in dirs:
            if added:
                print(f"  {src_dir}: ADDING {', '.join(added)} "
                      f"(absent from that export)")
        if additions and args.allow_missing:
            print("\n!! --allow_missing together with added bands writes a ragged "
                  "directory: records whose md_id is not in the corrected export "
                  "keep no value at all for the added bands, and "
                  "tf.io.FixedLenFeature raises on a record that is missing a "
                  "feature. Only safe if the corrected export covers every tile.")
        if args.dry_run:
            print("DRY RUN -- nothing will be written")

        results = run_jobs(jobs, table_path, index_path, args.workers)
        report(results, features, additions)
        write_sidecars(args, dirs, jobs, results, table_path, meta)
    finally:
        if args.cache_dir is None:
            shutil.rmtree(cache_dir, ignore_errors=True)

    if not args.dry_run:
        print()
        for src, dst, _, _ in jobs[:3]:
            differing = verify_shard(src, dst, features)
            print(f"verified {os.path.basename(dst)}: "
                  f"only {sorted(differing) or 'nothing'} differs")
        print("\nNEXT: re-pool the per-year stats into the training stats JSON, e.g.\n"
              "  python scripts/preprocessing/pool_stats_pbtxt.py "
              f"--data_dirs {os.path.join(args.output_root, 'allpreds_<year>')} ... "
              "--output <...>/stats_<years>.json")


if __name__ == "__main__":
    sys.exit(main())
