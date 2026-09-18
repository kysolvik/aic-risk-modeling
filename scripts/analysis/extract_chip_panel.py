"""Reduce the fullgrid_v2 TFRecords to a chip-year panel of burned area + drivers.

One row per (md_id, year): the aggregate burned-pixel count for each label, plus
chip-level spatial aggregates of every year-varying driver. This is the input to
the burned-area ceiling test, which asks whether aggregate burn is predictable at
all once we stop trying to place fire in exact pixels.

Why this is cheap despite ~115 GB of source data: only the ~140 bands we actually
reduce are put in the feature_spec, so the 64 AlphaEarth embedding bands and the
unused monthlies are never materialized. Each 128x128 band collapses to one to
three floats, so a 17 MB record becomes ~200 numbers.

Bands deliberately excluded because they are measurably dead or frozen in
fullgrid_v2 (verified against the per-year TFDV stats):
  im_gov_type            100% zeros
  im_chirps_cwd_-1..-6   97.9-99.2% zeros (annual CHIRPS took max() of a <=0
                         quantity; the monthly band is fine and is used instead)
  im_alert/_alertdate    bit-identical across years -- geebeam_ali_inputs.py
                         pulls the GLAD alert image with no year filter, so it
                         is one snapshot replicated into every year's export

Nodata handled explicitly: im_fire_type carries a -2147483648 sentinel that
nothing else in the repo masks (it makes the band's basin mean -1.85e6 and its
year-over-year ratio exactly 1.000). im_Elevation and im_accessibility carry
-32767 / -9999.

Run directly:
    .venv/bin/python scripts/extract_chip_panel.py \
        --data_dirs gs://aic-amazon/data/fullgrid_v2/allpreds_20{18,19,20,21,22,23,24}/ \
        --output_dir out/chip_panel/ --workers 8
    .venv/bin/python scripts/extract_chip_panel.py --output_dir out/chip_panel/ --combine
"""

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tensorflow as tf  # noqa: E402

from aic_risk_modeling.train import data_loader  # noqa: E402

PATCH_PIXELS = 128 * 128

# _-12 = Jan(Y-1) ... _-1 = Dec(Y-1). All three monthly preps in
# geebeam_ali_inputs.py are called as prep_*_monthly(TARGET_YEAR-1, TARGET_YEAR-1),
# so every driver predates the target year -- which is correct for a forecast
# issued in January of year Y.
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
    "im_BurnDate", "im_viirs_snpp",
    "im_EVI", "im_NDVI",
]

STATIC_BANDS = [
    "im_Elevation", "im_Slope", "im_treecover2000", "im_accessibility",
    "im_Population_Density", "im_Nighttime_Lights", "im_loss", "im_lossyear",
]

# Bands that also get within-chip spread, since a basin-scale drought signal can
# show up as heterogeneity rather than a shift in the mean.
SPREAD_BANDS = [
    "im_chirps_cwd_monthly",
    "im_cwd_monthly",
    "im_Vapour_Pressure_Deficit_at_Maximum_Temperature_monthly",
    "im_total_precipitation_sum_monthly",
]

# Sentinel values that are finite, so np.isfinite() filtering does not catch them.
NODATA = {
    "im_Elevation": -32767.0,
    "im_accessibility": -9999.0,
}
FIRE_TYPE_SENTINEL = -1e9  # im_fire_type nodata is -2147483648

CLIM_INDICES = ["md_amo", "md_mei", "md_oni", "md_soi", "md_tna"]
# 72 values = 6 years x 12 months for Y-6..Y-1, chronological. Index 60..71 is
# Jan..Dec of Y-1; 69..71 is Oct..Dec, the state closest to a January issue date.
CLIM_SLICES = {"y1": slice(60, 72), "y1ond": slice(69, 72), "y2": slice(48, 60)}

SCALAR_MD = ["md_id", "md_year", "md_x", "md_y"]


def timestepped(bands, timesteps):
    return [f"{b}_{t}" for b in bands for t in timesteps]


def wanted_features():
    """Every feature name the panel needs, in one flat list."""
    return (
        timestepped(MONTHLY_BANDS, MONTHLY_TIMESTEPS)
        + timestepped(ANNUAL_BANDS, ANNUAL_TIMESTEPS)
        + STATIC_BANDS
        + ["im_fire_type", "im_BurnDate_0", "im_viirs_snpp_0"]
        + SCALAR_MD
        + CLIM_INDICES
    )


def build_feature_spec(data_dir, allow_missing=False):
    """Restrict the schema's feature_spec to the bands we reduce.

    The 2013-2017 and 2025 exports live in a different bucket and carry a
    slightly different band set (no AlphaEarth, no `im_viirs_snpp_-6..-2`, and
    for 2013-2017 no `im_fire_type`). With allow_missing those years still join
    the panel, with the absent columns emitted as NaN rather than silently
    dropped, so the gap stays visible downstream.
    """
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


# Path helpers. tf.io.gfile speaks gs:// and local paths with one API, so the
# same code runs on a laptop and in a container whose only durable storage is a
# bucket. os.path.exists/makedirs silently do the wrong thing on gs:// -- they
# report False and create a literal "gs:" directory -- which would break
# resumability by re-extracting every shard on every run.
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
    """os.path.join mangles gs:// on some platforms; join on / explicitly."""
    return "/".join(p.strip("/") if i else p.rstrip("/")
                    for i, p in enumerate(parts))


def _list_shards(directory, pattern="*.tfrecord.gz"):
    shards = sorted(tf.io.gfile.glob(os.path.join(directory, pattern)))
    if not shards:
        raise ValueError(f"no shards matching {pattern!r} in {directory}")
    return shards


def year_of(data_dir):
    """2023 from '.../allpreds_2023/'."""
    m = re.search(r"(\d{4})", os.path.basename(data_dir.rstrip("/")))
    if not m:
        raise ValueError(f"no 4-digit year in directory name {data_dir!r}")
    return int(m.group(1))


def _clean(arr, nodata=None):
    """Finite pixels with any sentinel removed. Returns (values, n_bad).

    Non-finite values and the finite nodata sentinels are both dropped here. A
    plain .mean() would propagate a single NaN to the whole chip-year driver and
    HistGradientBoostingRegressor consumes NaN natively, so the band would
    quietly degrade into a missingness indicator without ever raising.
    """
    bad = ~np.isfinite(arr)
    if nodata is not None:
        bad = bad | (arr == nodata)
    return arr[~bad], int(bad.sum())


def _mean_or_nan(values):
    return float(values.mean()) if values.size else np.nan


def reduce_record(rec):
    """Collapse one parsed chip to a flat dict of scalars."""
    row = {}

    for name in SCALAR_MD:
        row[name] = (rec[name].reshape(-1)[0] if name in rec else np.nan)

    # Targets. im_fire_type must have its sentinel removed before anything else;
    # unmasked it swamps a ~2% signal and pins every year's mean to -1.85e6.
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
    row["n_pixels"] = PATCH_PIXELS
    # Valid area must be constant per chip across years; if it is not, a
    # year-varying nodata footprint would manufacture a year effect.
    row["n_valid_ft"] = (np.nan if sentinel is None
                         else PATCH_PIXELS - row["n_sentinel"])

    # Fire type composition, for the type-weighting question. Types 3/4 are the
    # deforestation/degradation classes.
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
            for label in CLIM_SLICES:
                row[f"{name}_{label}"] = np.nan
            continue
        series = rec[name].reshape(-1)
        for label, sl in CLIM_SLICES.items():
            values, _ = _clean(series[sl])
            row[f"{name}_{label}"] = _mean_or_nan(values)

    return row


def extract_shard(job, source=None):
    """Reduce one shard to a parquet of chip rows. Returns a stat dict."""
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
        # Which export produced this row. 2013-2017/2025 come from a different
        # bucket and a later version of the geebeam script, so any driver step
        # change at that boundary must stay traceable.
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
    """Build each dir's restricted spec once per worker."""
    for data_dir in spec_dirs:
        _SPEC[data_dir] = build_feature_spec(data_dir, allow_missing=allow_missing)


def _extract_shard_worker(job):
    shard, year, out_path, data_dir = job
    return extract_shard((shard, year, out_path, _SPEC[data_dir]), source=data_dir)


def validate_panel(df):
    """Checks whose failure would invalidate everything downstream.

    The panel key assumption is that md_id names the same patch of ground in
    every year. If it does not, the chip effect and the year effect get mixed
    and no amount of careful modelling downstream recovers.
    """
    problems = []

    for year, g in df.groupby("year"):
        ids = set(g["md_id"].astype(int))
        if ids != set(range(1813)):
            problems.append(
                f"{year}: md_id is not exactly 0..1812 "
                f"(n={len(ids)}, missing={len(set(range(1813)) - ids)})")
        bad_year = g["md_year"].dropna()
        if len(bad_year) and not (bad_year.astype(int) == year).all():
            problems.append(f"{year}: md_year disagrees with the directory year")

    # Same chip, same coordinates, every year.
    for axis in ("md_x", "md_y"):
        spread = df.groupby("md_id")[axis].agg(lambda s: s.max() - s.min())
        if (spread > 1e-5).any():
            problems.append(
                f"{axis} moves across years for "
                f"{int((spread > 1e-5).sum())} chips -- md_id is not a stable key")

    # A year-varying valid footprint would manufacture a year effect out of
    # nothing, which is exactly the signal this study is trying to measure.
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
    """Concatenate every per-shard parquet into one panel."""
    parts = sorted(tf.io.gfile.glob(_join(output_dir, "*", "*.parquet")))
    if not parts:
        raise ValueError(f"no per-shard parquet files under {output_dir}")
    df = pd.concat([_read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values(["year", "md_id"]).reset_index(drop=True)

    dup = df.duplicated(subset=["year", "md_id"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate (year, md_id) rows -- shards overlap?")

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

    # The check that catches a broken extraction: these must match the per-year
    # TFDV means (im_BurnDate_0 was 0.017346 in 2023 and 0.034581 in 2024).
    print("\nbasin-mean burned fraction by year (compare to TFDV stats):")
    print(f"  {'year':<6}{'burn_bd':>12}{'burn_snpp':>12}{'burn_ft':>12}"
          f"{'sentinel px':>14}")
    for year, g in df.groupby("year"):
        denom = g["n_pixels"].sum()
        print(f"  {year:<6}{g['burn_bd'].sum() / denom:>12.6f}"
              f"{g['burn_snpp'].sum() / denom:>12.6f}"
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
                        help="per dir; for smoke tests")
    parser.add_argument("--allow_missing", action="store_true",
                        help="emit absent bands as NaN instead of erroring "
                             "(needed for the 2013-2017 and 2025 exports)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-extract shards whose parquet already exists")
    parser.add_argument("--combine", action="store_true",
                        help="only concatenate existing per-shard parquets")
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
