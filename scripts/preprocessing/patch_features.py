"""Replace feature bands inside existing allpreds_* tfrecords.

Static fields (elevation, slope, accessibility, governance, ...) are duplicated
into every year's export, so re-exporting them means rewriting every year even
though the corrected data is one grid's worth of rasters. This script does that
rewrite: it reads a corrected *static* export, keys it by `md_id`, then streams
each year's shards replacing only the named features and copying every other
feature through byte-for-byte. It can also handle dynamic field rewrites, and
a band the year export never had at all is *added* rather than an error, which
is how a new static field gets backfilled into every existing year.

Why key on `md_id`: Beam gives no ordering guarantee, so record N of a shard in
the corrected export is not record N of the same-named shard in a year export.
Position-based zipping silently misaligns tiles. `md_x`/`md_y` are floats and a
poor key; `md_id` is an int64 and exact.

The source directories are never modified -- output goes to
`<output_root>/<source dir name>/<same shard name>`.

Cost: measured ~16 MB/s uncompressed single-threaded (gzip re-compression
dominates), i.e. ~1 s per 17 MB record. With `--workers 8` a ~1800-tile year
takes a few minutes. Shards are independent, so workers scale ~linearly.

AFTER RUNNING: regenerate the stats file. Training reads
`config['stats_path']`, defaulting to `data_dirs[0]/stats.pbtxt`, so a patched
directory with a stale stats file changes nothing about normalization -- which
is usually the entire point of the patch.

Each output dir also gets a `schema.pbtxt` copied from its source, extended
with any band this run added (`--skip_schema` to opt out). `data_loader`
builds its parse spec from that file per-dir, so an output dir without one is
unreadable and one that omits an added band ignores it. Stats are NOT copied:
they are regenerated, per above.

Examples:
    # Check coverage and what would change, without writing anything.
    python scripts/preprocessing/patch_features.py \\
        --corrected_dir gs://aic-amazon/data/corrected_v3/ \\
        --data_dirs gs://aic-amazon/data/fullgrid_v2/allpreds_2018/ \\
        --output_root gs://aic-amazon/data/fullgrid_v3/ \\
        --features im_Elevation,im_accessibility,im_gov_type --dry_run

    # Patch every year, 8 shards at a time.
    python scripts/preprocessing/patch_features.py \\
        --corrected_dir gs://aic-amazon/data/corrected_v3/ \\
        --data_dirs gs://aic-amazon/data/fullgrid_v2/allpreds_20{18,19,20,21,22,23,24}/ \\
        --output_root gs://aic-amazon/data/fullgrid_v3/ \\
        --features im_Elevation,im_accessibility,im_gov_type --workers 8
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import tensorflow as tf

try:  # only needed to declare newly added bands in schema.pbtxt
    from google.protobuf import text_format
    from tensorflow_metadata.proto.v0 import schema_pb2
except ImportError:  # pragma: no cover - environment without tfmd
    text_format = schema_pb2 = None


def _compression(path):
    return "GZIP" if path.endswith(".gz") else ""


def _list_shards(directory, pattern="*.tfrecord.gz"):
    shards = sorted(tf.io.gfile.glob(os.path.join(directory, pattern)))
    if not shards:
        raise ValueError(f"no shards matching {pattern!r} in {directory}")
    return shards


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
    if kind == "int64_list":
        return [int(round(float(v))) for v in values.tolist()]
    if kind == "float_list":
        return [float(v) for v in values.tolist()]
    raise ValueError(f"cannot write a numeric band into a {kind}")


def _set_values(feature, kind, values):
    """Write `values` into `feature`, replacing whatever it held."""
    target = getattr(feature, kind).value
    del target[:]
    target.extend(_as_kind(values, kind))


def infer_corrected_features(corrected_shards):
    """Every `im_*` band present in the first record of the static export."""
    for rec in tf.data.TFRecordDataset(corrected_shards[:1],
                                       compression_type=_compression(corrected_shards[0])):
        example = tf.train.Example.FromString(rec.numpy())
        return sorted(k for k in example.features.feature if k.startswith("im_"))
    raise ValueError("corrected export is empty")


def build_corrected_table(corrected_dir, features, cache_dir, pattern):
    """Materialize the corrected corrected bands as an mmap-able array.

    Returns (npy_path, index_path). The array is (n_tiles, n_features, width)
    float32; the index maps str(md_id) -> row, and the sidecar also records
    each band's protobuf oneof kind, which is the only way to know what type a
    band added to a year export should have. Written to disk rather than held
    in memory so that worker processes share one copy via the page cache
    instead of each inheriting their own (~800 MB for 1800 tiles x 7 bands).
    """
    shards = _list_shards(corrected_dir, pattern)
    if not features:
        features = infer_corrected_features(shards)
        print(f"inferred corrected features: {', '.join(features)}")

    rows, index, width, kinds = [], {}, None, {}
    for shard in shards:
        for rec in tf.data.TFRecordDataset([shard],
                                           compression_type=_compression(shard)):
            example = tf.train.Example.FromString(rec.numpy())
            feats = example.features.feature
            if "md_id" not in feats:
                raise ValueError(f"{shard}: corrected record has no md_id")
            md_id = str(feats["md_id"].int64_list.value[0])
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
            index[md_id] = len(rows)
            rows.append(np.stack(band))

    table = np.stack(rows)  # (n_tiles, n_features, width)
    tf.io.gfile.makedirs(cache_dir)
    npy_path = os.path.join(cache_dir, "corrected_table.npy")
    index_path = os.path.join(cache_dir, "corrected_index.json")
    np.save(npy_path, table)
    with open(index_path, "w") as f:
        json.dump({"index": index, "features": list(features),
                   "width": width, "kinds": kinds}, f)
    for name, kind in kinds.items():
        if kind == "int64_list":
            peak = float(np.abs(table[:, list(features).index(name)]).max())
            if peak > 2 ** 24:
                print(f"  WARNING: {name} is int64 and reaches {peak:.0f}; "
                      f"the float32 table cannot round-trip it exactly")
    print(f"corrected table: {table.shape} tiles x features x px "
          f"({table.nbytes / 1e6:.0f} MB) -> {npy_path}")
    return npy_path, index_path


_WORKER = {}


def _init_worker(npy_path, index_path):
    with open(index_path) as f:
        meta = json.load(f)
    _WORKER["table"] = np.load(npy_path, mmap_mode="r")
    _WORKER["index"] = meta["index"]
    _WORKER["features"] = meta["features"]
    _WORKER["kinds"] = meta["kinds"]


def patch_shard(job):
    """Rewrite one shard with the corrected bands replaced. Returns a stat dict."""
    src, dst, dry_run, allow_missing = job
    table, index = _WORKER["table"], _WORKER["index"]
    features, kinds = _WORKER["features"], _WORKER["kinds"]
    stats = {"shard": os.path.basename(src), "records": 0, "missing": 0,
             "changed": {name: 0 for name in features},
             "added": {name: 0 for name in features}}

    writer = None
    if not dry_run:
        tf.io.gfile.makedirs(os.path.dirname(dst))
        writer = tf.io.TFRecordWriter(
            dst, tf.io.TFRecordOptions(compression_type=_compression(dst)))
    try:
        for rec in tf.data.TFRecordDataset([src],
                                           compression_type=_compression(src)):
            example = tf.train.Example.FromString(rec.numpy())
            feats = example.features.feature
            stats["records"] += 1
            md_id = str(feats["md_id"].int64_list.value[0])
            row = index.get(md_id)
            if row is None:
                stats["missing"] += 1
                if not allow_missing:
                    raise KeyError(
                        f"{src}: md_id {md_id} is absent from the corrected "
                        f"export (pass --allow_missing to copy such records "
                        f"through unpatched)")
            else:
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
    a = next(iter(tf.data.TFRecordDataset([src], compression_type=_compression(src))))
    b = next(iter(tf.data.TFRecordDataset([dst], compression_type=_compression(dst))))
    ea = tf.train.Example.FromString(a.numpy())
    eb = tf.train.Example.FromString(b.numpy())
    src_names, dst_names = set(ea.features.feature), set(eb.features.feature)
    dropped = src_names - dst_names
    if dropped:
        raise AssertionError(f"{dst}: features disappeared: {sorted(dropped)}")
    added = dst_names - src_names
    differing = added | {k for k in src_names
                         if ea.features.feature[k].SerializeToString()
                         != eb.features.feature[k].SerializeToString()}
    unexpected = differing - set(features)
    if unexpected:
        raise AssertionError(f"{dst}: unexpected features changed: {unexpected}")
    return differing


def features_missing_from(directory, features, pattern):
    """Which of `features` the first record of `directory` does not carry."""
    shard = _list_shards(directory, pattern)[0]
    for rec in tf.data.TFRecordDataset([shard],
                                       compression_type=_compression(shard)):
        present = set(tf.train.Example.FromString(rec.numpy()).features.feature)
        return [name for name in features if name not in present]
    raise ValueError(f"{directory}: first shard is empty")


def write_schema(src_dir, dst_dir, added, kinds, width, dry_run):
    """Copy the schema sidecars to `dst_dir`, declaring the `added` bands.

    `data_loader.load_schema_from_gcs` builds the parse spec from the
    schema.pbtxt sitting next to the shards, so an output dir without one is
    unreadable, and one that omits a band this run added leaves that band
    unparsed no matter that it is in the records.
    """
    src_schema = os.path.join(src_dir, 'schema.pbtxt')
    dst_schema = os.path.join(dst_dir, 'schema.pbtxt')
    if not tf.io.gfile.exists(src_schema):
        print(f"  no schema.pbtxt in {src_dir} -- the patched dir will need "
              f"one before training can read it")
        return
    if dry_run:
        print(f"  would write {dst_schema}"
              + (f" (+{', '.join(added)})" if added else " (copy)"))
        return
    tf.io.gfile.makedirs(dst_dir)
    if not added:
        tf.io.gfile.copy(src_schema, dst_schema, overwrite=True)
    else:
        if schema_pb2 is None:
            raise ImportError(
                "tensorflow_metadata is required to declare added bands in "
                "schema.pbtxt; install it or pass --skip_schema and edit the "
                "schema by hand")
        schema = schema_pb2.Schema()
        with tf.io.gfile.GFile(src_schema) as f:
            text_format.Parse(f.read(), schema)
        present = {f.name for f in schema.feature}
        for name in added:
            if name in present:
                continue
            feature = schema.feature.add()
            feature.name = name
            feature.type = (schema_pb2.FeatureType.INT
                            if kinds[name] == 'int64_list'
                            else schema_pb2.FeatureType.FLOAT)
            feature.presence.min_fraction = 1.0
            feature.presence.min_count = 1
            feature.shape.dim.add().size = width
        with tf.io.gfile.GFile(dst_schema, 'w') as f:
            f.write(text_format.MessageToString(schema))
    print(f"  wrote {dst_schema}"
          + (f" (+{', '.join(added)})" if added else " (copy)"))

    src_json = os.path.join(src_dir, 'schema.json')
    if added and tf.io.gfile.exists(src_json):
        with tf.io.gfile.GFile(src_json) as f:
            meta = json.load(f)
        for name in added:
            meta.setdefault('features', {}).setdefault(
                name, 'int64' if kinds[name] == 'int64_list' else 'float')
        with tf.io.gfile.GFile(os.path.join(dst_dir, 'schema.json'), 'w') as f:
            json.dump(meta, f, indent=2)
    elif tf.io.gfile.exists(src_json):
        tf.io.gfile.copy(src_json, os.path.join(dst_dir, 'schema.json'),
                         overwrite=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corrected_dir', required=True,
                        help='directory of corrected tfrecords, keyed by md_id')
    parser.add_argument('--data_dirs', nargs='+', required=True,
                        help='year directories to patch (never modified)')
    parser.add_argument('--output_root', required=True,
                        help='patched years are written to <root>/<dir name>/')
    parser.add_argument('--features', default=None,
                        help='comma-separated bands to replace; default: every '
                             'im_* band in the corrected export')
    parser.add_argument('--tfrecord_pattern', default='*.tfrecord.gz')
    parser.add_argument('--cache_dir', default=None,
                        help='where to materialize the corrected table '
                             '(default: <output_root>/_corrected_cache, must be local)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit_shards', type=int, default=None,
                        help='patch only the first N shards per year (for testing)')
    parser.add_argument('--allow_missing', action='store_true',
                        help='copy records whose md_id is absent from the corrected '
                             'export through unpatched instead of failing')
    parser.add_argument('--skip_schema', action='store_true',
                        help='do not copy/patch schema.pbtxt into the output dirs')
    parser.add_argument('--dry_run', action='store_true',
                        help='report coverage and what would change; write nothing')
    args = parser.parse_args()

    features = ([f.strip() for f in args.features.split(',')]
                if args.features else None)
    cache_dir = args.cache_dir or os.path.join(
        args.output_root if not args.output_root.startswith('gs://') else '.',
        '_corrected_cache')
    npy_path, index_path = build_corrected_table(
        args.corrected_dir, features, cache_dir, args.tfrecord_pattern)
    with open(index_path) as f:
        meta = json.load(f)
    features = meta['features']
    kinds = meta['kinds']
    width = meta['width']
    n_tiles = len(meta['index'])

    jobs, dirs, additions = [], [], {}
    for data_dir in args.data_dirs:
        name = os.path.basename(data_dir.rstrip('/'))
        out_dir = os.path.join(args.output_root, name)
        shards = _list_shards(data_dir, args.tfrecord_pattern)
        if args.limit_shards:
            shards = shards[:args.limit_shards]
        missing_feats = features_missing_from(data_dir, features,
                                              args.tfrecord_pattern)
        if missing_feats:
            additions[data_dir] = missing_feats
        dirs.append((data_dir, out_dir, missing_feats))
        for shard in shards:
            dst = os.path.join(out_dir, os.path.basename(shard))
            jobs.append((shard, dst, args.dry_run, args.allow_missing))
    print(f"{len(jobs)} shards across {len(args.data_dirs)} dirs; "
          f"{n_tiles} tiles in the corrected table; features: {', '.join(features)}")
    for data_dir, missing_feats in additions.items():
        print(f"  {data_dir}: ADDING {', '.join(missing_feats)} "
              f"(absent from that export)")
    if additions and args.allow_missing:
        print("\n!! --allow_missing together with added bands writes a ragged "
              "directory: records whose md_id is not in the corrected export "
              "keep no value at all for the added bands, and "
              "tf.io.FixedLenFeature raises on a record that is missing a "
              "feature. Only safe if the corrected export covers every tile.")
    if args.dry_run:
        print("DRY RUN -- nothing will be written")

    if args.workers > 1:
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(npy_path, index_path)) as pool:
            results = pool.map(patch_shard, jobs)
    else:
        _init_worker(npy_path, index_path)
        results = [patch_shard(job) for job in jobs]

    records = sum(r['records'] for r in results)
    missing = sum(r['missing'] for r in results)
    print(f"\npatched {records} records ({missing} unmatched md_id)")
    for name in features:
        changed = sum(r['changed'][name] for r in results)
        added = sum(r['added'][name] for r in results)
        line = f"  {name:24s} changed in {changed:6d} / {records} records"
        if added:
            line += f", added to {added}"
        elif changed == 0:
            line += ('  <- IDENTICAL everywhere; the corrected export did not '
                     'actually change this band')
        print(line)

    if missing and additions:
        added_names = sorted({n for v in additions.values() for n in v})
        print(f"\n!! {missing} records were copied through unpatched and so "
              f"carry no {', '.join(added_names)}. Parsing the patched dirs "
              f"with a schema that declares those bands WILL FAIL on those "
              f"records; drop those tiles or extend the corrected export.")

    if not args.skip_schema:
        print()
        for src_dir, out_dir, missing_feats in dirs:
            write_schema(src_dir, out_dir, missing_feats, kinds, width,
                         args.dry_run)

    if not args.dry_run:
        for src, dst, _, _ in jobs[:min(3, len(jobs))]:
            differing = verify_shard(src, dst, features)
            print(f"verified {os.path.basename(dst)}: "
                  f"only {sorted(differing) or 'nothing'} differs")
        print("\nNEXT: regenerate the stats file for the patched dirs, e.g.\n"
              "  python -m aic_risk_modeling.train.data_stats "
              f"--data_dirs {os.path.join(args.output_root, '<year dir>')} "
              "--output_path <...>/stats.json\n"
              "and point config['stats_path'] at it (it otherwise defaults to "
              "data_dirs[0]/stats.pbtxt).")


if __name__ == '__main__':
    sys.exit(main())
