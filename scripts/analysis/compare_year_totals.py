"""Compare aggregate predicted vs actual burn across prediction years.

Walks per-chip prediction outputs from scripts/predict/predict.py (paired
out_{x}-{y}.tif probability rasters and mask_{x}-{y}.tif ground truth) for two
or more years and reports, per year, the total actual burned pixels vs the
total expected burned pixels (sum of predicted probabilities), plus cross-year
ratios and the per-chip correlation between the predicted and actual
year-to-year change.

This is the year-sensitivity smoking-gun check: a model that ignores
interannual drivers shows expected_yearB/expected_yearA ~= 1 even when
actual_yearB/actual_yearA is far from 1, and ~zero correlation between
per-chip delta-expected and delta-actual.

Expected counts are reported both raw and "deflated". Models trained with
weighted BCE (pos_weight w) are pushed toward the inflated pointwise optimum
q = w*p / (w*p + 1 - p); deflating with the exact inverse p = q / (w - (w-1)*q)
recovers a calibrated-scale probability, so the deflated ratio is the one
comparable to 1.0.

Example:
    python scripts/compare_year_totals.py \
        ~/research/firesat/risk_modeling/test_out/2023_mts_v11 \
        ~/research/firesat/risk_modeling/test_out/2024_mts_v11 \
        --pos-weight 9.0 --csv year_totals.csv
"""

import argparse
import csv
import glob
import os

import numpy as np
import rasterio as rio


def deflate(q, pos_weight):
    """Invert the weighted-BCE optimum q = w*p/(w*p+1-p) back to p."""
    return q / (pos_weight - (pos_weight - 1.0) * q)


def chip_key(path):
    """'out_-44.07--3.10.tif' -> '-44.07--3.10' (the chip's x-y coordinate)."""
    base = os.path.basename(path)
    return base.split('_', 1)[1].rsplit('.tif', 1)[0]


def scan_year(pred_dir, pos_weight):
    """Per-chip {key: (actual, expected_raw, expected_deflated)} for one dir."""
    out_paths = sorted(glob.glob(os.path.join(pred_dir, 'out_*.tif')))
    if not out_paths:
        raise SystemExit(f"no out_*.tif files in {pred_dir}")
    chips = {}
    for out_path in out_paths:
        key = chip_key(out_path)
        mask_path = os.path.join(pred_dir, f'mask_{key}.tif')
        if not os.path.exists(mask_path):
            print(f"  warning: no mask for {out_path}, skipping")
            continue
        with rio.open(out_path) as src:
            probs = src.read(1).astype(np.float64)
        with rio.open(mask_path) as src:
            mask = src.read(1)
        probs = np.clip(probs, 0.0, 1.0)
        chips[key] = (float((mask > 0).sum()), float(probs.sum()),
                      float(deflate(probs, pos_weight).sum()))
    return chips


def totals(chips):
    a = sum(c[0] for c in chips.values())
    e_raw = sum(c[1] for c in chips.values())
    e_def = sum(c[2] for c in chips.values())
    return a, e_raw, e_def


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pred_dirs', nargs='+',
                    help='per-year prediction dirs (out_*.tif + mask_*.tif)')
    ap.add_argument('--pos-weight', type=float, default=9.0,
                    help='pos_weight the model was trained with (for deflation)')
    ap.add_argument('--csv', default=None,
                    help='optional path for per-chip CSV output')
    args = ap.parse_args()

    years = {}
    for d in args.pred_dirs:
        label = os.path.basename(os.path.normpath(d))
        print(f"scanning {label} ...", flush=True)
        years[label] = scan_year(d, args.pos_weight)

    print(f"\n{'year dir':28s} {'chips':>6s} {'actual':>12s} {'E[raw]':>12s} "
          f"{'E[defl]':>12s} {'raw/act':>8s} {'defl/act':>9s}")
    for label, chips in years.items():
        a, e_raw, e_def = totals(chips)
        print(f"{label:28s} {len(chips):6d} {a:12.0f} {e_raw:12.0f} "
              f"{e_def:12.0f} {e_raw / a:8.3f} {e_def / a:9.3f}")

    labels = list(years)
    for i in range(len(labels) - 1):
        la, lb = labels[i], labels[i + 1]
        aa, ea_raw, ea_def = totals(years[la])
        ab, eb_raw, eb_def = totals(years[lb])
        shared = sorted(set(years[la]) & set(years[lb]))
        d_actual = [years[lb][k][0] - years[la][k][0] for k in shared]
        d_expect = [years[lb][k][2] - years[la][k][2] for k in shared]
        within_a = pearson([years[la][k][2] for k in shared],
                           [years[la][k][0] for k in shared])
        within_b = pearson([years[lb][k][2] for k in shared],
                           [years[lb][k][0] for k in shared])
        print(f"\n{lb} vs {la}  ({len(shared)} shared chips)")
        print(f"  actual ratio          {ab / aa:6.3f}")
        print(f"  expected ratio (raw)  {eb_raw / ea_raw:6.3f}")
        print(f"  expected ratio (defl) {eb_def / ea_def:6.3f}")
        print(f"  per-chip r(dE, dA)    {pearson(d_expect, d_actual):6.3f}")
        print(f"  per-chip r(E, A) within-year: {la} {within_a:.3f}, "
              f"{lb} {within_b:.3f}")

    print("\nInterpretation: expected ratio ~= 1 while actual ratio is far "
          "from 1 (and r(dE, dA) ~= 0) means the model is not differentiating "
          "years at the aggregate level.")

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['year_dir', 'chip', 'actual', 'expected_raw',
                        'expected_deflated'])
            for label, chips in years.items():
                for key, (a, e_raw, e_def) in sorted(chips.items()):
                    w.writerow([label, key, a, e_raw, e_def])
        print(f"per-chip CSV written to {args.csv}")


if __name__ == '__main__':
    main()
