"""Calibrated expected-vs-actual annual burned-pixel totals (CSV for plot_expected_actual.py).

The calibrator is fit once on --fit-years and frozen; an in-sample per-year fit would make
expected track actual by construction. --loyo calibrates each fit year on the other fit years.
Usage: calibrated_year_totals.py --pred-root R --fit-years 2018 ... --eval-years ... --out-csv C"""

import argparse
import csv
import glob
import os

import numpy as np
import rasterio as rio

from aic_risk_modeling.eval.calibration import (apply_platt, fit_calibrator, fit_isotonic,
                                                fit_platt, load_calibrator)


def chip_key(path):
    base = os.path.basename(path)
    return base.split('_', 1)[1].rsplit('.tif', 1)[0]


def year_pairs(roots, year):
    """Yield (pred, mask) paths for a year from the first root holding its mosaic or chip dir."""
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
    with rio.open(out_path) as src:
        probs = np.clip(src.read(1).astype(np.float32), 0.0, 1.0).reshape(-1)
    with rio.open(mask_path) as src:
        labels = (src.read(1) > 0).astype(np.float32).reshape(-1)
    if probs.shape != labels.shape:
        raise SystemExit(f"shape mismatch {out_path} vs {mask_path}")
    return probs, labels


def collect_fit_pixels(roots, fit_years, max_pixels, seed):
    """Per-year (scores, labels) for fitting, uniformly subsampled (calibration needs the true base rate)."""
    per_year = max_pixels // len(fit_years)
    rng = np.random.default_rng(seed)
    pixels = {}
    for year in fit_years:
        scores, labels = [], []
        for out_path, mask_path in year_pairs(roots, year):
            p, l = read_pair(out_path, mask_path)
            scores.append(p)
            labels.append(l)
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        if scores.size > per_year:
            idx = rng.choice(scores.size, size=per_year, replace=False)
            scores, labels = scores[idx], labels[idx]
        pixels[year] = (scores, labels)
    return pixels


def fit_on(method, pixels, years):
    scores = np.concatenate([pixels[y][0] for y in years])
    labels = np.concatenate([pixels[y][1] for y in years])
    print(f"  fit on {years}: {scores.size:,} pixels "
          f"(base rate {labels.mean():.4f}, mean raw prob {scores.mean():.4f})")
    return fit_calibrator(method, scores, labels)


def save_calibrator(path, method, pixels, years, meta):
    """Save the frozen all-fit-years calibrator (platt a, b or isotonic breakpoints) as npz."""
    scores = np.concatenate([pixels[y][0] for y in years])
    labels = np.concatenate([pixels[y][1] for y in years])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if method == 'platt':
        a, b = fit_platt(scores, labels)
        probe = np.linspace(0.0, 1.0, 100_001)
        ref, _ = fit_calibrator('platt', scores, labels)
        diff = float(np.abs(apply_platt(probe, a, b) - ref(probe)).max())
        if diff > 1e-9:
            raise SystemExit(f"saved calibrator != fitted platt (max diff {diff:.2e})")
        np.savez(path, method='platt', a=a, b=b, fit_years=np.array(years), **meta)
        print(f"Saved platt calibrator (a={a:.4f}, b={b:.4f}, fit {years}) -> {path}")
        return
    iso = fit_isotonic(scores, labels)
    x, y = iso.X_thresholds_, iso.y_thresholds_
    probe = np.linspace(0.0, 1.0, 100_001)
    diff = float(np.abs(np.interp(probe, x, y) - iso.predict(probe)).max())
    if diff > 1e-9:
        raise SystemExit(f"saved calibrator != iso.predict (max diff {diff:.2e})")
    np.savez(path, method='isotonic', x=x, y=y, fit_years=np.array(years), **meta)
    print(f"Saved isotonic calibrator ({x.size} breakpoints, fit {years}) -> {path}")


def year_totals(roots, year, transform):
    """(expected_raw, expected_cal, actual) burned-pixel sums for one year, streamed pair by pair."""
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
                    help='dirs with {year}_out.tif or {year}/ chips, searched in order')
    ap.add_argument('--fit-years', nargs='+', type=int, required=True,
                    help='held-out years to fit the frozen calibrator on')
    ap.add_argument('--eval-years', nargs='+', type=int, required=True,
                    help='years to total')
    ap.add_argument('--method', default='platt',
                    choices=('none', 'platt', 'isotonic'),
                    help="calibrator to fit; 'none' sums raw probabilities")
    ap.add_argument('--loyo', action='store_true',
                    help='calibrate fit years on the other fit years only')
    ap.add_argument('--fit-max-pixels', type=int, default=5_000_000,
                    help='cap on pooled fit pixels')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-actual-years', nargs='*', type=int, default=(),
                    help='predict-only years (actual left blank)')
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--save-calibrator', default=None, metavar='NPZ',
                    help='save the frozen calibrator to this npz')
    ap.add_argument('--frozen-calibrator', default=None, metavar='NPZ',
                    help='apply this frozen calibrator to non-LOYO years')
    args = ap.parse_args()
    if args.frozen_calibrator and args.save_calibrator:
        raise SystemExit('--frozen-calibrator and --save-calibrator are exclusive')
    if args.frozen_calibrator and args.method == 'none':
        raise SystemExit('--frozen-calibrator needs a fitted --method (for the LOYO years)')
    if args.save_calibrator and args.method not in ('platt', 'isotonic'):
        raise SystemExit('--save-calibrator supports --method platt or isotonic only')

    fit_set = set(args.fit_years)
    if args.loyo and len(fit_set) < 2:
        raise SystemExit('--loyo needs at least 2 --fit-years')
    calibrators = {}
    if args.method == 'none':
        calibrators[None] = (lambda s: s, [])
        print('Calibrator: none (raw probability sums)')
    else:
        print(f"Fitting {args.method} calibrator on held-out years "
              f"{args.fit_years} ...", flush=True)
        pixels = collect_fit_pixels(
            args.pred_root, args.fit_years, args.fit_max_pixels, args.seed)
        transform, info = fit_on(args.method, pixels, args.fit_years)
        calibrators[None] = (transform, list(args.fit_years))
        print(f"Calibrator: {info}  (FROZEN, applied to every non-LOYO eval year)")
        if args.save_calibrator:
            save_calibrator(args.save_calibrator, args.method, pixels, list(args.fit_years),
                          {'seed': args.seed, 'fit_max_pixels': args.fit_max_pixels,
                           'pred_root': np.array(args.pred_root)})
        if args.frozen_calibrator:
            calibrators[None] = (load_calibrator(args.frozen_calibrator), list(args.fit_years))
            print(f"Non-LOYO eval years use the SAVED calibrator {args.frozen_calibrator} "
                  f"(the refit above is not applied to them)")
        if args.loyo:
            for year in args.eval_years:
                if year in fit_set:
                    others = [y for y in args.fit_years if y != year]
                    transform, info = fit_on(args.method, pixels, others)
                    calibrators[year] = (transform, others)
                    print(f"  LOYO {year}: {info}")

    blank = set(args.no_actual_years)
    rows = []
    print(f"\n{'year':>6} {'in-samp':>8} {'actual':>12} {'E[raw]':>12} "
          f"{'E[cal]':>12} {'cal/act':>8}")
    for year in args.eval_years:
        transform, cal_years = calibrators.get(year, calibrators[None])
        in_samp = year in cal_years
        exp_raw, exp_cal, actual = year_totals(args.pred_root, year, transform)
        actual_out = '' if year in blank else actual
        ratio = (exp_cal / actual) if (actual and year not in blank) else float('nan')
        tag = 'fit' if in_samp else ('loyo' if year in calibrators else '-')
        print(f"{year:>6} {tag:>8} "
              f"{('' if year in blank else f'{actual:.0f}'):>12} "
              f"{exp_raw:>12.0f} {exp_cal:>12.0f} {ratio:>8.3f}")
        # plot_expected_actual.py draws expected_adj as the "Expected" line.
        rows.append({'year': year, 'expected': round(exp_raw, 1),
                     'actual': actual_out, 'expected_adj': round(exp_cal, 1),
                     'calibrated_on': ' '.join(map(str, cal_years))})

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['year', 'expected', 'actual',
                                          'expected_adj', 'calibrated_on'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")
    print("Note: 'in-samp=fit' years are in-sample -- their agreement is not "
          "evidence of year skill; judge the model on the other years "
          "('loyo' = calibrated on the other fit years only).")


if __name__ == '__main__':
    main()
