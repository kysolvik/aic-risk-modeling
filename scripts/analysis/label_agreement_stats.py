#!/usr/bin/env python
"""Per-pixel agreement tables for the three fire products, by land cover and year.

The training target is an OR of three imperfect detectors of one latent event
("did this cell burn this year"): MCD64A1 burn scars (`im_BurnDate_0`), MOD14
active fire (`im_mod14_0`) and VIIRS SNPP hotspots (`im_viirs_snpp_0`). This
script reduces the v3 TFRecords to the sufficient statistics for fitting a
latent-class measurement model over them (scripts/analysis/fit_label_model.py):

  patterns   the 2x2x2 contingency table of (d_MCD64, d_MOD14, d_VIIRS) per
             (year, land-cover stratum), at two spatial tolerances
  deltas     |dDOY| histograms for each co-detecting pair, plus the same
             histogram under a within-stratum permutation null, so the fit can
             tell "these two saw the same fire" from "this cell burns a lot"

WHY TWO SPATIAL TOLERANCES. The products sit on different native grids -- MCD64
500 m, VIIRS 463 m, MOD14 926.6 m -- against a ~555 m chip pixel, and MCD64A1's
Collection 6 algorithm grows its burned training samples outward from active-fire
detections, so its scars sit ADJACENT to the detections that seeded them.
notes/v43_union_target.txt measured that dilating VIIRS by one pixel lifts
P(VIIRS|MODIS) from 43% to 77%. Scoring agreement at exact pixel coincidence
therefore charges registration and scale error to the detectors as if it were
missed fire, and biases the coarse product's sensitivity down. Both tables are
emitted so the tolerance is chosen from evidence rather than asserted.

The union label is NEVER dilated -- it stays the raw OR, bit-identical to the
pipeline's `combine: "any"`, because that is the frozen evaluation label.

Cheap despite ~100 GB of source: only 5 bands enter the feature_spec, and each
128x128 chip collapses to a handful of integer counts.

Shard sampling defaults to skipping shard 0 and striding, because chips are laid
out spatially and shard 0 is not a representative sample of the basin
(notes/data_version_audit.txt).

Run directly:
    .venv/bin/python scripts/analysis/label_agreement_stats.py \
        --data_dirs gs://aic-amazon/data/fullgrid_v3/allpreds_20{13..22}/ \
        --output_dir out/label_model/ --workers 8 --max_shards 6
    .venv/bin/python scripts/analysis/label_agreement_stats.py \
        --output_dir out/label_model/ --combine
"""

import argparse
import io
import itertools
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import tensorflow as tf  # noqa: E402

from aic_risk_modeling.train import data_loader  # noqa: E402

# Order is load-bearing: the pattern index is d_0 + 2*d_1 + 4*d_2 and
# fit_label_model.py / the training config list products in this same order.
PRODUCTS = ["im_BurnDate_0", "im_mod14_0", "im_viirs_snpp_0"]
SHORT = {"im_BurnDate_0": "mcd64", "im_mod14_0": "mod14",
         "im_viirs_snpp_0": "viirs"}

# Prior-year MapBiomas forest fraction in [0,1]. Lag -1, so stratifying on it
# cannot leak same-year information into anything fitted here.
STRAT_BAND = "im_forest_-1"
DEFAULT_FOREST_EDGES = [0.25, 0.75]

# Default spatial tolerance, in chip pixels, for the "tolerant" table. MOD14's
# native cell spans ~1.7 chip pixels; MCD64 is the reference grid and is not
# dilated.
DEFAULT_DILATE = {"im_BurnDate_0": 0, "im_mod14_0": 1, "im_viirs_snpp_0": 1}

# |dDOY| bin edges, in days. The first bin is the MCD64 Collection 6 compositing
# window (W = 8 days); Giglio et al. 2018 report 44% of MCD64 cells detected on
# the same day as an active fire and 68% within two days, so real co-detections
# should pile up hard in bin 0.
DOY_EDGES = [2.0, 8.0, 32.0, 128.0]


def _compression(path):
    return "GZIP" if path.endswith(".gz") else ""


def _join(*parts):
    return "/".join(p.strip("/") if i else p.rstrip("/")
                    for i, p in enumerate(parts))


def _write_parquet(df, out_path):
    parent = out_path.rsplit("/", 1)[0]
    tf.io.gfile.makedirs(parent)
    if out_path.startswith("gs://"):
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "part.parquet")
            df.to_parquet(local, index=False)
            tf.io.gfile.copy(local, out_path, overwrite=True)
    else:
        df.to_parquet(out_path, index=False)


def _read_parquet(path):
    if path.startswith("gs://"):
        with tf.io.gfile.GFile(path, "rb") as f:
            return pd.read_parquet(io.BytesIO(f.read()))
    return pd.read_parquet(path)


def year_of(data_dir):
    m = re.search(r"(\d{4})", os.path.basename(data_dir.rstrip("/")))
    if not m:
        raise ValueError(f"no 4-digit year in directory name {data_dir!r}")
    return int(m.group(1))


def select_shards(directory, pattern, max_shards=None, skip_first=True, seed=0):
    """Shards spread across the listing, not the first N.

    Chips are tiled spatially, so a prefix of the shard list is a corner of the
    basin rather than a sample of it; shard 0 in particular has repeatedly
    misrepresented band distributions in this dataset.
    """
    shards = sorted(tf.io.gfile.glob(os.path.join(directory, pattern)))
    if not shards:
        raise ValueError(f"no shards matching {pattern!r} in {directory}")
    if skip_first:
        shards = shards[1:] or shards
    if max_shards and max_shards < len(shards):
        idx = np.linspace(0, len(shards) - 1, max_shards).round().astype(int)
        shards = [shards[i] for i in sorted(set(idx.tolist()))]
    return shards


def _dilate_nowrap(mask, radius):
    """Dilation without wraparound: pad with False, then max over shifts."""
    if not radius:
        return mask
    h, w = mask.shape
    pad = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    pad[radius:radius + h, radius:radius + w] = mask
    out = np.zeros_like(mask)
    for dy, dx in itertools.product(range(-radius, radius + 1), repeat=2):
        out |= pad[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
    return out


def stratum_index(forest_frac, edges):
    idx = np.zeros(forest_frac.shape, dtype=np.int64)
    for edge in edges:
        idx += (forest_frac >= edge).astype(np.int64)
    return idx


def reduce_record(rec, edges, dilate_by, rng):
    """One chip -> (pattern rows, delta rows). Pure counting, no I/O."""
    doy = {name: np.asarray(rec[name], dtype=np.float32).reshape(128, 128)
           for name in PRODUCTS}
    hits = {name: band > 0 for name, band in doy.items()}
    forest = np.asarray(rec[STRAT_BAND], dtype=np.float32).reshape(128, 128)
    strata = stratum_index(np.nan_to_num(forest, nan=0.0), edges)

    rows = []
    for mode, radii in (("exact", {n: 0 for n in PRODUCTS}), ("tolerant", dilate_by)):
        det = [_dilate_nowrap(hits[n], radii[n]) for n in PRODUCTS]
        pattern = (det[0].astype(np.int64)
                   + 2 * det[1].astype(np.int64)
                   + 4 * det[2].astype(np.int64))
        # One bincount over stratum*8 + pattern gives the whole table at once.
        flat = (strata * 8 + pattern).ravel()
        counts = np.bincount(flat, minlength=(edges and len(edges) + 1 or 1) * 8)
        for cell, n in enumerate(counts):
            if n:
                rows.append({"mode": mode, "stratum": cell // 8,
                             "pattern": cell % 8, "count": int(n)})

    # |dDOY| for each co-detecting pair, at exact coincidence only (under
    # dilation "whose day-of-year" is ambiguous). The null shuffles one product's
    # detection dates within the same stratum, which preserves how much each
    # product fires and when in the season, and destroys only the pairing -- so a
    # peak above the null is co-detection of one fire rather than a busy cell.
    deltas = []
    for i, j in itertools.combinations(range(3), 2):
        a, b = PRODUCTS[i], PRODUCTS[j]
        co = hits[a] & hits[b]
        if not co.any():
            continue
        for stratum in np.unique(strata[co]):
            sel = co & (strata == stratum)
            d_obs = np.abs(doy[a][sel] - doy[b][sel])
            pool_a = doy[a][hits[a] & (strata == stratum)]
            pool_b = doy[b][hits[b] & (strata == stratum)]
            if pool_a.size and pool_b.size:
                n = int(sel.sum())
                d_null = np.abs(rng.choice(pool_a, n) - rng.choice(pool_b, n))
            else:
                d_null = np.empty(0, dtype=np.float32)
            for kind, arr in (("observed", d_obs), ("null", d_null)):
                if not arr.size:
                    continue
                binned = np.digitize(arr, DOY_EDGES)
                for b_idx, n in enumerate(np.bincount(binned,
                                                      minlength=len(DOY_EDGES) + 1)):
                    if n:
                        deltas.append({"pair": f"{SHORT[a]}|{SHORT[b]}",
                                       "stratum": int(stratum), "kind": kind,
                                       "bin": b_idx, "count": int(n)})
    return rows, deltas


_SPEC = {}


def _init_worker(spec_dirs):
    for directory in spec_dirs:
        schema = data_loader.load_schema_from_gcs(directory)
        full = data_loader.schema_to_feature_spec(schema)
        want = PRODUCTS + [STRAT_BAND, "md_year"]
        missing = [n for n in want if n not in full]
        if missing:
            raise KeyError(f"{directory} schema is missing {missing}")
        _SPEC[directory] = {n: full[n] for n in want}


def extract_shard(job):
    """Reduce one shard to two parquets of counts. Returns a stat dict."""
    shard, directory, year, out_prefix, edges, dilate_by, seed = job
    t0 = time.time()
    spec = _SPEC[directory]
    ds = tf.data.TFRecordDataset([shard], compression_type=_compression(shard))
    ds = ds.map(lambda x: tf.io.parse_single_example(x, spec),
                num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    rng = np.random.default_rng(seed)
    pattern_rows, delta_rows, n_chips = [], [], 0
    for rec in ds.as_numpy_iterator():
        rows, deltas = reduce_record(rec, edges, dilate_by, rng)
        pattern_rows.extend(rows)
        delta_rows.extend(deltas)
        n_chips += 1

    if not pattern_rows:
        return {"shard": shard, "chips": 0, "seconds": time.time() - t0}

    patterns = (pd.DataFrame(pattern_rows)
                .groupby(["mode", "stratum", "pattern"], as_index=False)["count"].sum())
    patterns["year"] = year
    _write_parquet(patterns, f"{out_prefix}_patterns.parquet")

    if delta_rows:
        deltas = (pd.DataFrame(delta_rows)
                  .groupby(["pair", "stratum", "kind", "bin"], as_index=False)["count"].sum())
        deltas["year"] = year
        _write_parquet(deltas, f"{out_prefix}_deltas.parquet")

    return {"shard": shard, "chips": n_chips, "seconds": time.time() - t0}


def combine(output_dir, edges):
    """Sum the per-shard parquets into two tidy tables plus a readable summary."""
    parts = sorted(tf.io.gfile.glob(_join(output_dir, "shards", "*_patterns.parquet")))
    if not parts:
        raise SystemExit(f"no per-shard parquets under {output_dir}/shards/")
    patterns = (pd.concat([_read_parquet(p) for p in parts], ignore_index=True)
                .groupby(["mode", "year", "stratum", "pattern"], as_index=False)["count"].sum())
    _write_parquet(patterns, _join(output_dir, "patterns.parquet"))

    dparts = sorted(tf.io.gfile.glob(_join(output_dir, "shards", "*_deltas.parquet")))
    if dparts:
        deltas = (pd.concat([_read_parquet(p) for p in dparts], ignore_index=True)
                  .groupby(["pair", "year", "stratum", "kind", "bin"], as_index=False)["count"].sum())
        _write_parquet(deltas, _join(output_dir, "deltas.parquet"))

    print(f"[combine] {len(parts)} shards, "
          f"{int(patterns['count'].sum()):,} pixels\n")
    for mode in ("exact", "tolerant"):
        sub = patterns[patterns["mode"] == mode]
        if sub.empty:
            continue
        print(f"--- {mode} ---")
        print(f"{'stratum':>8} {'pixels':>14} {'mcd64%':>8} {'mod14%':>8} "
              f"{'viirs%':>8} {'union%':>8} {'all3/union':>11}")
        for stratum, grp in sub.groupby("stratum"):
            tot = grp["count"].sum()
            by = grp.groupby("pattern")["count"].sum().reindex(range(8), fill_value=0)
            marg = [sum(by[k] for k in range(8) if k & (1 << i)) for i in range(3)]
            union = tot - by[0]
            print(f"{stratum:>8} {tot:>14,} "
                  + " ".join(f"{100.0 * m / tot:>7.2f}%" for m in marg)
                  + f" {100.0 * union / tot:>7.2f}% "
                  + f"{(by[7] / union if union else 0):>10.3f}")
        print()
    print(f"strata are forest-fraction bins with edges {edges}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dirs", nargs="+", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tfrecord_pattern", default="*.tfrecord.gz")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_shards", type=int, default=6,
                        help="shards per year, spread across the listing")
    parser.add_argument("--keep_first_shard", action="store_true",
                        help="include shard 0 (excluded by default: it is a "
                             "spatial corner, not a sample)")
    parser.add_argument("--forest_edges", type=float, nargs="*",
                        default=DEFAULT_FOREST_EDGES)
    parser.add_argument("--dilate", type=int, nargs=3, default=None,
                        metavar=("MCD64", "MOD14", "VIIRS"),
                        help="tolerant-table radii in chip pixels "
                             f"(default {[DEFAULT_DILATE[p] for p in PRODUCTS]})")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()

    edges = list(args.forest_edges)
    if args.combine:
        combine(args.output_dir, edges)
        return

    if not args.data_dirs:
        parser.error("--data_dirs is required unless --combine")
    dilate_by = (DEFAULT_DILATE if args.dilate is None
                 else dict(zip(PRODUCTS, args.dilate)))

    jobs = []
    for directory in args.data_dirs:
        year = year_of(directory)
        shards = select_shards(directory, args.tfrecord_pattern, args.max_shards,
                               skip_first=not args.keep_first_shard)
        for shard in shards:
            stem = os.path.basename(shard).split(".")[0]
            prefix = _join(args.output_dir, "shards", f"{year}_{stem}")
            jobs.append((shard, directory, year, prefix, edges, dilate_by,
                         args.seed + len(jobs)))

    print(f"[stats] {len(jobs)} shards across {len(args.data_dirs)} years, "
          f"forest edges {edges}, tolerant radii "
          f"{[dilate_by[p] for p in PRODUCTS]}", flush=True)

    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.data_dirs,)) as pool:
        for i, stat in enumerate(pool.imap_unordered(extract_shard, jobs), 1):
            print(f"  [{i}/{len(jobs)}] {os.path.basename(stat['shard'])} "
                  f"{stat['chips']} chips in {stat['seconds']:.1f}s", flush=True)
    print(f"[stats] done in {time.time() - t0:.0f}s. Now run with --combine.")


if __name__ == "__main__":
    main()
