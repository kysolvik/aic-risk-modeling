"""Calibrated expected-vs-actual annual burn totals, for the timeseries figure.

Turns per-year prediction rasters into the CSV that
``scripts/analysis/plot_expected_actual.py`` draws, but replaces the ad-hoc
constant bias factor (the old ``expected_adj = expected * 1.2``) with a *proper*
post-hoc probability calibrator fit through the eval module
(``aic_risk_modeling.eval.calibration.fit_calibrator``: platt / isotonic /
temperature). Expected burned pixels for a year = sum of the calibrated
per-pixel fire probabilities; actual = count of burned label pixels.

THE ONE RULE THAT MAKES THIS HONEST: the calibrator is fit ONCE, on the
held-out ``--fit-years``, and then FROZEN and applied identically to every
evaluated year. Do NOT fit in-sample per year -- isotonic/platt would drag each
year's expected total onto that year's own base rate, so expected would track
actual *by construction* and the interannual test the figure exists for would be
destroyed. A frozen calibrator is just a fixed monotonic reweighting of the
probabilities; it corrects the model's overall over/under-confidence without
inventing year-to-year skill. Years in ``--fit-years`` are in-sample (their
agreement is not evidence); every other year is the honest out-of-sample test.

Out-of-basin pixels are stored as an exact 0 probability / 0 label, so they add
nothing to either sum; totals are basin-restricted automatically, matching
``compare_year_totals.py`` / ``scatter_expected_actual.py``.

Two input layouts (auto-detected per year, mosaic preferred):
  mosaic:  {root}/{year}_out.tif  + {root}/{year}_mask.tif      (whole basin)
  chips:   {root}/{year}/out_*.tif + {root}/{year}/mask_*.tif   (per-chip tiles)

Example:
    .venv/bin/python scripts/analysis/calibrated_year_totals.py \
        --pred-root out/baselines/factored_v1 \
        --fit-years 2023 --eval-years 2023 2024 \
        --method isotonic \
        --out-csv out/expected_actual_factored_v1_cal.csv
    .venv/bin/python scripts/analysis/plot_expected_actual.py \
        --csv out/expected_actual_factored_v1_cal.csv \
        --out out/expected_actual_factored_v1_cal.png \
        --start-year 2023 --end-year 2024
"""

import argparse
import csv
import glob
import os

import numpy as np
import rasterio as rio

# Reuse the eval module's calibrators directly -- same code the eval CLI's
# --calibration-method flag uses, so a frozen fit here matches an eval run.
from aic_risk_modeling.eval.calibration import fit_calibrator


def chip_key(path):
    """'out_-44.07--3.10.tif' -> '-44.07--3.10' (the chip's x-y coordinate)."""
    base = os.path.basename(path)
    return base.split('_', 1)[1].rsplit('.tif', 1)[0]


def year_pairs(roots, year):
    """Yield (pred_path, mask_path) for a year from the first root that has it.

    ``roots`` is searched in order; the first root holding either the whole-basin
    mosaic ``{root}/{year}_out.tif`` (preferred) or the per-chip directory
    ``{root}/{year}/`` wins. This lets years that live under different roots
    (e.g. 2023/2024 in out/baselines/... and 2025 in out/preds2025/...) share one
    frozen calibrator.
    """
    for root in roots:
        mosaic_out = os.path.join(root, f'{year}_out.tif')
        mosaic_mask = os.path.join(root, f'{year}_mask.tif')
        if os.path.exists(mosaic_out) and os.path.exists(mosaic_mask):
            yield mosaic_out, mosaic_mask
            return
        out_paths = sorted(glob.glob(os.path.join(root, str(year), 'out_*.tif')))
        if out_paths:
            for out_path in out_paths:
                mask_path = os.path.join(root, str(year),
                                         f'mask_{chip_key(out_path)}.tif')
                if os.path.exists(mask_path):
                    yield out_path, mask_path
                else:
                    print(f"  warning: no mask for {out_path}, skipping")
            return
    raise SystemExit(
        f"year {year}: no {year}_out.tif mosaic and no {year}/out_*.tif chips "
        f"under any of {list(roots)}")


def read_pair(out_path, mask_path):
    """Flat (probs in [0,1], labels 0/1 float) for one raster pair."""
    with rio.open(out_path) as src:
        probs = np.clip(src.read(1).astype(np.float32), 0.0, 1.0).reshape(-1)
    with rio.open(mask_path) as src:
        labels = (src.read(1) > 0).astype(np.float32).reshape(-1)
    if probs.shape != labels.shape:
        raise SystemExit(f"shape mismatch {out_path} vs {mask_path}")
    return probs, labels


def collect_fit_pixels(roots, fit_years, max_pixels, seed):
    """Pool (scores, labels) over the held-out fit years for calibrator fitting.

    Uniformly random-subsamples to ``max_pixels`` when the pool is larger. The
    subsample is UNIFORM (not class-balanced) on purpose: calibration must see
    the true base rate, so stratifying would bias the fit.
    """
    scores, labels = [], []
    for year in fit_years:
        for out_path, mask_path in year_pairs(roots, year):
            p, l = read_pair(out_path, mask_path)
            scores.append(p)
            labels.append(l)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    if scores.size > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(scores.size, size=max_pixels, replace=False)
        scores, labels = scores[idx], labels[idx]
    return scores, labels


def year_totals(roots, year, transform):
    """(expected_raw, expected_cal, actual) burned-pixel sums for one year.

    Streams pair-by-pair so a full-basin mosaic never needs the calibrated copy
    held in memory alongside the raw one.
    """
    exp_raw = exp_cal = actual = 0.0
    for out_path, mask_path in year_pairs(roots, year):
        p, l = read_pair(out_path, mask_path)
        exp_raw += float(p.sum(dtype=np.float64))
        exp_cal += float(np.asarray(transform(p)).sum(dtype=np.float64))
        actual += float(l.sum(dtype=np.float64))
    return exp_raw, exp_cal, actual


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred-root', nargs='+', required=True,
                    help='one or more dirs holding {year}_out.tif mosaics or '
                         '{year}/ chip dirs; searched in order per year')
    ap.add_argument('--fit-years', nargs='+', type=int, required=True,
                    help='held-out year(s) to FIT the frozen calibrator on')
    ap.add_argument('--eval-years', nargs='+', type=int, required=True,
                    help='year(s) to compute expected/actual totals for')
    ap.add_argument('--method', default='isotonic',
                    choices=('none', 'temperature', 'platt', 'isotonic'),
                    help="eval-module calibrator to fit (default isotonic); "
                         "'none' just sums raw probabilities")
    ap.add_argument('--fit-max-pixels', type=int, default=5_000_000,
                    help='cap on pooled fit pixels (uniform subsample above it)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-actual-years', nargs='*', type=int, default=(),
                    help='predict-only years to blank the actual for (e.g. 2026)')
    ap.add_argument('--out-csv', required=True)
    args = ap.parse_args()

    if args.method == 'none':
        transform = lambda s: s  # noqa: E731
        info = 'none (raw probability sums)'
    else:
        print(f"Fitting {args.method} calibrator on held-out years "
              f"{args.fit_years} ...", flush=True)
        fit_scores, fit_labels = collect_fit_pixels(
            args.pred_root, args.fit_years, args.fit_max_pixels, args.seed)
        print(f"  fit on {fit_scores.size:,} pixels "
              f"(base rate {fit_labels.mean():.4f}, "
              f"mean raw prob {fit_scores.mean():.4f})")
        transform, info = fit_calibrator(args.method, fit_scores, fit_labels)
    print(f"Calibrator: {info}  (FROZEN, applied to every eval year)")

    fit_set = set(args.fit_years)
    blank = set(args.no_actual_years)
    rows = []
    print(f"\n{'year':>6} {'in-samp':>8} {'actual':>12} {'E[raw]':>12} "
          f"{'E[cal]':>12} {'cal/act':>8}")
    for year in args.eval_years:
        exp_raw, exp_cal, actual = year_totals(args.pred_root, year, transform)
        actual_out = '' if year in blank else actual
        ratio = (exp_cal / actual) if (actual and year not in blank) else float('nan')
        print(f"{year:>6} {('fit' if year in fit_set else '-'):>8} "
              f"{('' if year in blank else f'{actual:.0f}'):>12} "
              f"{exp_raw:>12.0f} {exp_cal:>12.0f} {ratio:>8.3f}")
        # Column names match plot_expected_actual.py: it plots expected_adj as
        # the "Expected" line, so the calibrated sum goes there.
        rows.append({'year': year, 'expected': round(exp_raw, 1),
                     'actual': actual_out, 'expected_adj': round(exp_cal, 1)})

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['year', 'expected', 'actual',
                                          'expected_adj'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")
    print("Note: 'in-samp=fit' years are in-sample -- their agreement is not "
          "evidence of year skill; judge the model on the other years.")


if __name__ == '__main__':
    main()
