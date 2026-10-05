"""Reduce fullgrid TFRecords to a chip-year panel of burn counts + chip-aggregated drivers.

Only the reduced bands are parsed. Dead/frozen bands (im_gov_type, annual CHIRPS CWD, GLAD
alerts) are excluded; nodata sentinels (im_fire_type, elevation, accessibility) are masked.
Usage: extract_chip_panel.py --data_dirs DIR ... --output_dir out/chip_panel/ [--combine]"""

import argparse
import io
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import tensorflow as tf

from aic_risk_modeling.train import data_loader

PATCH_PIXELS = 128 * 128

# _-12 = Jan(Y-1) ... _-1 = Dec(Y-1): every driver predates the target year.
MONTHLY_TIMESTEPS = [str(-i) for i in range(12, 0, -1)]
ANNUAL_TIMESTEPS = [str(-i) for i in range(6, 0, -1)]

MONTHLY_BANDS = [
    "im_chirps_cwd_monthly",
    "im_cwd_monthly",
    "im_Vapour_Pressure_Deficit_at_Maximum_Temperature_monthly",
    "im_total_precipitation_sum_monthly",
    "im_Temperature_Air_2m_Max_24h_monthly",
    "im_EVI_monthly",
    "im_NDVI_monthly",
]

ANNUAL_BANDS = [
    "im_ag", "im_pasture", "im_forest",
    "im_BurnDate", "im_viirs_snpp", "im_mod14",
    "im_EVI", "im_NDVI",
]

STATIC_BANDS = [
    "im_Elevation", "im_Slope", "im_treecover2000", "im_accessibility",
    "im_Population_Density", "im_Nighttime_Lights", "im_loss", "im_lossyear",
]

SPREAD_BANDS = [
    "im_chirps_cwd_monthly",
    "im_cwd_monthly",
    "im_Vapour_Pressure_Deficit_at_Maximum_Temperature_monthly",
    "im_total_precipitation_sum_monthly",
]

NODATA = {
    "im_Elevation": -32767.0,
    "im_accessibility": -9999.0,
}
FIRE_TYPE_SENTINEL = -1e9

CLIM_INDICES = ["md_amo", "md_mei", "md_oni", "md_soi", "md_tna"]
# Climate vectors are 12*N months ending Dec(Y-1) (N=6 in v2, 10 in v3); slice from the tail,
# since fixed length-72 offsets read the wrong year on a v3 export.
CLIM_SLICES_FROM_END = {"y1": (12, 0), "y1ond": (3, 0), "y2": (24, 12)}


def clim_slice(length, months_back, months_forward):
    """slice for series[length-months_back : length-months_forward] (0 => tail)."""
    if length < 24 or length % 12:
        raise ValueError(f"climate index length {length} is not a whole number of "
                         "years >= 2; cannot anchor y1/y1ond/y2 to Dec(Y-1)")
    return slice(length - months_back, length - months_forward if months_forward else length)

SCALAR_MD = ["md_id", "md_year", "md_x", "md_y"]


def timestepped(bands, timesteps):
    return [f"{b}_{t}" for b in bands for t in timesteps]


def wanted_features():
    return (
        timestepped(MONTHLY_BANDS, MONTHLY_TIMESTEPS)
        + timestepped(ANNUAL_BANDS, ANNUAL_TIMESTEPS)
        + STATIC_BANDS
        + ["im_fire_type", "im_BurnDate_0", "im_viirs_snpp_0", "im_mod14_0"]
        + SCALAR_MD
        + CLIM_INDICES
    )


def build_feature_spec(data_dir, allow_missing=False):
    """Schema feature_spec restricted to the reduced bands; allow_missing emits absent ones as NaN."""
    schema = data_loader.load_schema_from_gcs(data_dir)
    full = data_loader.schema_to_feature_spec(schema)
    want = wanted_features()
    missing = [name for name in want if name not in full]
    if missing and not allow_missing:
        raise KeyError(
            f"{len(missing)} requested features absent from {data_dir} schema: "
            f"{missing[:8]}{'...' if len(missing) > 8 else ''}. "
            f"Pass --allow_missing to emit them as NaN.")
    return {name: full[name] for name in want if name in full}


def _compression(path):
    return "GZIP" if path.endswith(".gz") else ""


# tf.io.gfile for gs:// and local alike (os.path.exists/makedirs misbehave on gs://).
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


def _join(*parts):
    return "/".join(p.strip("/") if i else p.rstrip("/")
                    for i, p in enumerate(parts))


def _list_shards(directory, pattern="*.tfrecord.gz"):
    shards = sorted(tf.io.gfile.glob(os.path.join(directory, pattern)))
    if not shards:
        raise ValueError(f"no shards matching {pattern!r} in {directory}")
    return shards


def year_of(data_dir):
    m = re.search(r"(\d{4})", os.path.basename(data_dir.rstrip("/")))
    if not m:
        raise ValueError(f"no 4-digit year in directory name {data_dir!r}")
    return int(m.group(1))


def _clean(arr, nodata=None):
    """(finite non-sentinel values, n_bad); a single NaN would otherwise poison the chip mean."""
    bad = ~np.isfinite(arr)
    if nodata is not None:
        bad = bad | (arr == nodata)
    return arr[~bad], int(bad.sum())


def _mean_or_nan(values):
    return float(values.mean()) if values.size else np.nan


def reduce_record(rec):
    row = {}

    for name in SCALAR_MD:
        row[name] = (rec[name].reshape(-1)[0] if name in rec else np.nan)

        # The im_fire_type sentinel must go first; unmasked it pins the mean to -1.85e6.
    ft = rec.get("im_fire_type")
    if ft is None:
        sentinel = None
        row["n_sentinel"] = np.nan
        row["burn_ft"] = np.nan
    else:
        sentinel = (ft < FIRE_TYPE_SENTINEL) | ~np.isfinite(ft)
        row["n_sentinel"] = int(sentinel.sum())
        row["burn_ft"] = int(((ft > 0) & ~sentinel).sum())
    row["burn_bd"] = (int((rec["im_BurnDate_0"] > 0).sum())
                      if "im_BurnDate_0" in rec else np.nan)
    row["burn_snpp"] = (int((rec["im_viirs_snpp_0"] > 0).sum())
                        if "im_viirs_snpp_0" in rec else np.nan)
    row["burn_mod14"] = (int((rec["im_mod14_0"] > 0).sum())
                         if "im_mod14_0" in rec else np.nan)
    row["n_pixels"] = PATCH_PIXELS
        # Valid area must be constant per chip; a year-varying footprint fakes a year effect.
    row["n_valid_ft"] = (np.nan if sentinel is None
                         else PATCH_PIXELS - row["n_sentinel"])

    valid_ft = None if sentinel is None else ft[~sentinel]
    for cls in (1, 2, 3, 4):
        row[f"burn_ft_c{cls}"] = (np.nan if valid_ft is None
                                  else int((valid_ft == cls).sum()))

    for base in MONTHLY_BANDS:
        n_bad = 0
        for ts in MONTHLY_TIMESTEPS:
            arr = rec.get(f"{base}_{ts}")
            if arr is None:
                row[f"{base}_{ts}_mean"] = np.nan
                continue
            values, bad = _clean(arr)
            n_bad += bad
            row[f"{base}_{ts}_mean"] = _mean_or_nan(values)
        row[f"{base}_nbad"] = n_bad
        if base in SPREAD_BANDS:
            stack = np.concatenate(
                [_clean(rec[f"{base}_{ts}"])[0] for ts in MONTHLY_TIMESTEPS
                 if f"{base}_{ts}" in rec] or [np.array([])])
            if stack.size:
                row[f"{base}_p10"] = float(np.percentile(stack, 10))
                row[f"{base}_p90"] = float(np.percentile(stack, 90))
            else:
                row[f"{base}_p10"] = row[f"{base}_p90"] = np.nan

    for base in ANNUAL_BANDS:
        n_bad = 0
        for ts in ANNUAL_TIMESTEPS:
            arr = rec.get(f"{base}_{ts}")
            if arr is None:
                row[f"{base}_{ts}_mean"] = np.nan
                continue
            values, bad = _clean(arr)
            n_bad += bad
            row[f"{base}_{ts}_mean"] = _mean_or_nan(values)
        row[f"{base}_nbad"] = n_bad

    for base in STATIC_BANDS:
        if base not in rec:
            row[f"{base}_mean"] = row[f"{base}_nodata"] = np.nan
            continue
        values, bad = _clean(rec[base], NODATA.get(base))
        row[f"{base}_mean"] = _mean_or_nan(values)
        row[f"{base}_nodata"] = bad

    for name in CLIM_INDICES:
        if name not in rec:
            for label in CLIM_SLICES_FROM_END:
                row[f"{name}_{label}"] = np.nan
            continue
        series = rec[name].reshape(-1)
        for label, (back, fwd) in CLIM_SLICES_FROM_END.items():
            values, _ = _clean(series[clim_slice(series.size, back, fwd)])
            row[f"{name}_{label}"] = _mean_or_nan(values)

    return row


def extract_shard(job, source=None):
    """Reduce one shard to a parquet of chip rows; returns a stat dict."""
    shard, year, out_path, spec = job
    t0 = time.time()
    ds = tf.data.TFRecordDataset([shard], compression_type=_compression(shard))
    ds = ds.map(lambda x: tf.io.parse_single_example(x, spec),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    rows = []
    for rec in ds.as_numpy_iterator():
        row = reduce_record(rec)
        row["year"] = year
        row["export_source"] = source or os.path.dirname(shard.rstrip("/"))
        rows.append(row)

    if not rows:
        return {"shard": shard, "records": 0, "seconds": time.time() - t0}

    df = pd.DataFrame(rows)
    bad_year = int((df["md_year"] != year).sum())
    _write_parquet(df, out_path)
    return {"shard": shard, "records": len(df), "bad_year": bad_year,
            "seconds": time.time() - t0}


_SPEC = {}


def _init_worker(spec_dirs, allow_missing=False):
    for data_dir in spec_dirs:
        _SPEC[data_dir] = build_feature_spec(data_dir, allow_missing=allow_missing)


def _extract_shard_worker(job):
    shard, year, out_path, data_dir = job
    return extract_shard((shard, year, out_path, _SPEC[data_dir]), source=data_dir)


def validate_panel(df):
    """Panel-key checks: md_id is the same ground, coordinates and valid footprint in every year."""
    problems = []

    all_ids = set(df["md_id"].dropna().astype(int))
    n_chips = len(all_ids)
    expected = set(range(n_chips))
    if all_ids != expected:
        problems.append(
            f"md_id across the panel is not contiguous 0..{n_chips - 1} "
            f"(n={n_chips}, gaps/extras={len(all_ids ^ expected)})")

    for year, g in df.groupby("year"):
        ids = set(g["md_id"].astype(int))
        if ids != expected:
            problems.append(
                f"{year}: md_id is not exactly 0..{n_chips - 1} "
                f"(n={len(ids)}, missing={len(expected - ids)})")
        bad_year = g["md_year"].dropna()
        if len(bad_year) and not (bad_year.astype(int) == year).all():
            problems.append(f"{year}: md_year disagrees with the directory year")

    for axis in ("md_x", "md_y"):
        spread = df.groupby("md_id")[axis].agg(lambda s: s.max() - s.min())
        if (spread > 1e-5).any():
            problems.append(
                f"{axis} moves across years for "
                f"{int((spread > 1e-5).sum())} chips -- md_id is not a stable key")

    # A year-varying valid footprint would manufacture a year effect.
    if "n_valid_ft" in df.columns:
        valid = df.dropna(subset=["n_valid_ft"])
        if len(valid):
            spread = valid.groupby("md_id")["n_valid_ft"].agg(
                lambda s: s.max() - s.min())
            n_moving = int((spread > 0).sum())
            if n_moving:
                problems.append(
                    f"n_valid_ft varies across years for {n_moving} chips "
                    f"(max swing {int(spread.max())} px) -- nodata footprint "
                    f"changes by year, check before trusting year effects")
    return problems


def combine(output_dir):
    parts = sorted(tf.io.gfile.glob(_join(output_dir, "*", "*.parquet")))
    if not parts:
        raise ValueError(f"no per-shard parquet files under {output_dir}")
    df = pd.concat([_read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values(["year", "md_id"]).reset_index(drop=True)

    dup = df.duplicated(subset=["year", "md_id"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate (year, md_id) rows -- shards overlap?")

    stale = [c for c in ("burn_bd", "burn_snpp", "burn_mod14", "burn_ft")
             if c not in df.columns]
    if stale:
        raise ValueError(
            f"per-shard parquets are missing {stale}; they predate the current "
            f"extractor. Re-extract with --overwrite (same --data_dirs) so every "
            f"shard is recomputed, then re-run --combine.")

    if "export_source" not in df.columns:
        df["export_source"] = np.nan
    df["export_source"] = df["export_source"].fillna("fullgrid_v2")

    problems = validate_panel(df)
    if problems:
        print("\nVALIDATION PROBLEMS:")
        for p in problems:
            print(f"  ! {p}")
    else:
        print("\nvalidation: all checks passed")

    out_path = _join(output_dir, "panel.parquet")
    _write_parquet(df, out_path)

    print(f"\npanel: {len(df)} rows x {df.shape[1]} cols -> {out_path}")
    print(f"years: {sorted(df['year'].unique().tolist())}")
    print(f"chips per year: {df.groupby('year').size().to_dict()}")

    # Sanity check against per-year TFDV means (im_BurnDate_0: 0.017346 in 2023, 0.034581 in 2024).
    print("\nbasin-mean burned fraction by year (compare to TFDV stats):")
    print(f"  {'year':<6}{'burn_bd':>12}{'burn_snpp':>12}{'burn_mod14':>12}"
          f"{'burn_ft':>12}{'sentinel px':>14}")
    for year, g in df.groupby("year"):
        denom = g["n_pixels"].sum()
        print(f"  {year:<6}{g['burn_bd'].sum() / denom:>12.6f}"
              f"{g['burn_snpp'].sum() / denom:>12.6f}"
              f"{g['burn_mod14'].sum() / denom:>12.6f}"
              f"{g['burn_ft'].sum() / denom:>12.6f}"
              f"{int(g['n_sentinel'].sum()):>14d}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dirs", nargs="+", default=[],
                        help="allpreds_* dirs, local or gs://")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tfrecord_pattern", default="*.tfrecord.gz")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_shards", type=int, default=None,
                        help="per dir")
    parser.add_argument("--allow_missing", action="store_true",
                        help="emit absent bands as NaN (2013-2017, 2025 exports)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-extract existing shards")
    parser.add_argument("--combine", action="store_true",
                        help="only concatenate existing shard parquets")
    args = parser.parse_args()

    if args.combine:
        combine(args.output_dir)
        return 0

    if not args.data_dirs:
        parser.error("--data_dirs is required unless --combine is given")

    jobs, skipped = [], 0
    for data_dir in args.data_dirs:
        year = year_of(data_dir)
        shards = _list_shards(data_dir, args.tfrecord_pattern)
        if args.max_shards:
            shards = shards[:args.max_shards]
        for shard in shards:
            name = os.path.basename(shard).replace(".tfrecord.gz", ".parquet")
            out_path = _join(args.output_dir, str(year), name)
            if tf.io.gfile.exists(out_path) and not args.overwrite:
                skipped += 1
                continue
            jobs.append((shard, year, out_path, data_dir))

    for data_dir in args.data_dirs:
        spec = build_feature_spec(data_dir, allow_missing=args.allow_missing)
        gap = [n for n in wanted_features() if n not in spec]
        if gap:
            print(f"  {os.path.basename(data_dir.rstrip('/'))}: {len(gap)} bands "
                  f"absent, emitted as NaN -> {gap[:6]}"
                  f"{'...' if len(gap) > 6 else ''}")
    print(f"{len(jobs)} shards to extract across {len(args.data_dirs)} dirs "
          f"({skipped} already done, skipping)")
    if not jobs:
        print("nothing to do; run with --combine to build the panel")
        return 0

    t0 = time.time()
    if args.workers > 1:
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(args.data_dirs, args.allow_missing)) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(_extract_shard_worker, jobs), 1):
                results.append(res)
                print(f"[{i}/{len(jobs)}] {os.path.basename(res['shard'])}: "
                      f"{res['records']} chips in {res['seconds']:.0f}s", flush=True)
    else:
        _init_worker(args.data_dirs, args.allow_missing)
        results = []
        for i, job in enumerate(jobs, 1):
            res = _extract_shard_worker(job)
            results.append(res)
            print(f"[{i}/{len(jobs)}] {os.path.basename(res['shard'])}: "
                  f"{res['records']} chips in {res['seconds']:.0f}s", flush=True)

    total = sum(r["records"] for r in results)
    bad_year = sum(r.get("bad_year", 0) for r in results)
    elapsed = time.time() - t0
    print(f"\n{total} chips in {elapsed / 60:.1f} min "
          f"({total / max(elapsed, 1e-9):.1f} chips/s)")
    if bad_year:
        print(f"WARNING: {bad_year} records whose md_year != directory year")
    print("next: re-run with --combine to build panel.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
