"""Per-chip expected vs actual burned area, one panel per model, one figure per year.

`compare_year_totals.py` reduces each year to a single basin total, which answers
"does the model track the year-to-year swing" but hides where the agreement comes
from. This draws the per-chip cloud behind that number: 1,813 points per panel,
expected burned pixels on x, actual on y, with a 1:1 reference and an OLS fit.

Everything is put on ONE scale -- expected burned pixels per chip -- so the
panels are directly comparable:

  models         sum of DEFLATED probability. They are trained with weighted BCE
                 (pos_weight 10), whose pointwise optimum is
                 q = w*p/(w*p + 1 - p); the exact inverse p = q/(w - (w-1)q)
                 puts them back on a calibrated scale. Note this is an
                 idealisation -- it assumes the model reaches that optimum, and
                 the calibration work says these models do not exactly -- but it
                 is applied identically to every model, so the comparison holds.
  climatology    sum of per-pixel burn frequency over the climatology years =
                 the expected burned pixels under climatology. Already on a
                 natural scale, so NOT deflated.
  last-year burn sum of the previous year's 0/1 burn mask = last year's burned
                 pixel count. Also not deflated.

Both baselines are windowed reads out of the full-basin label mosaics, verified
pixel-identical to the per-chip `mask_` rasters.

R^2 is from an ordinary least-squares fit of actual on expected. Chip burned area
is heavily right-skewed, so that R^2 is dominated by the largest chips; Spearman
rho is annotated alongside as the rank-based, outlier-resistant companion. Quote
both, or quote rho if the cloud is skewed.

Usage:
    .venv/bin/python scripts/analysis/scatter_expected_actual.py --year 2024 \
        --model "Factored v1=out/baselines/factored_v1/2024" \
        --model "MLP rf9=out/baselines/baseline_mlp/2024" \
        --label-dir /path/to/label_mosaics \
        --climatology /path/to/climatology_2013_2022.tif \
        --out_csv out/scatter/scatter_2024.csv --out_png out/scatter/scatter_2024.png
"""

import argparse
import glob
import os

import numpy as np

POINT = "#2a78d6"
FIT = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e4e3de"
ONE_TO_ONE = "#8a8880"


def deflate(q, pos_weight):
    """Invert the weighted-BCE optimum q = w*p/(w*p+1-p) back to p."""
    return q / (pos_weight - (pos_weight - 1.0) * q)


def chip_pairs(directory):
    """[(out_path, mask_path)] for the chips directly in `directory`."""
    out_paths = sorted(glob.glob(os.path.join(directory, "out_*.tif")))
    if not out_paths:
        raise FileNotFoundError(f"no out_*.tif in {directory}")
    pairs = []
    for p in out_paths:
        m = os.path.join(os.path.dirname(p),
                         os.path.basename(p).replace("out_", "mask_", 1))
        if not os.path.exists(m):
            raise FileNotFoundError(f"missing mask for {p}")
        pairs.append((p, m))
    return pairs


def model_series(directory, pos_weight):
    """(expected, actual) per chip for a model's prediction directory."""
    import rasterio as rio
    exp, act = [], []
    for out_path, mask_path in chip_pairs(directory):
        with rio.open(out_path) as s:
            q = np.clip(s.read(1).astype(np.float64), 0.0, 1.0)
        with rio.open(mask_path) as m:
            lab = m.read(1) > 0
        exp.append(float(deflate(q, pos_weight).sum()))
        act.append(float(lab.sum()))
    return np.array(exp), np.array(act)


def raster_series(reference_dir, raster_path, binarize):
    """(expected, actual) per chip, scoring a full-basin raster through windows."""
    import rasterio as rio
    from rasterio.windows import from_bounds
    exp, act = [], []
    with rio.open(raster_path) as src:
        for _, mask_path in chip_pairs(reference_dir):
            with rio.open(mask_path) as m:
                lab = m.read(1) > 0
                bounds = m.bounds
            win = from_bounds(*bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
            v = src.read(1, window=win).astype(np.float64)
            if v.shape != lab.shape:
                raise ValueError(f"window {v.shape} != chip {lab.shape}")
            exp.append(float((v > 0).sum() if binarize else v.sum()))
            act.append(float(lab.sum()))
    return np.array(exp), np.array(act)


def fit_stats(x, y):
    """OLS slope/intercept/R^2 plus Spearman rho."""
    if x.std() == 0 or y.std() == 0:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, rho=np.nan)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    rho = float(np.corrcoef(xr, yr)[0, 1])
    return dict(slope=float(slope), intercept=float(intercept), r2=float(r2), rho=rho)


def plot(panels, year, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.2, 8.6), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()

    # One shared square range across panels so the 1:1 line sits at 45 degrees
    # everywhere and the panels can be read against each other.
    hi = max(float(np.percentile(np.concatenate([p["x"] for p in panels]), 99.5)),
             float(np.percentile(np.concatenate([p["y"] for p in panels]), 99.5)))
    hi *= 1.05

    for ax, p in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.plot([0, hi], [0, hi], "--", color=ONE_TO_ONE, linewidth=1.3,
                zorder=2, label="1:1")
        ax.scatter(p["x"], p["y"], s=9, color=POINT, alpha=0.28,
                   linewidths=0, zorder=3)
        s = p["stats"]
        xs = np.array([0, hi])
        ax.plot(xs, s["slope"] * xs + s["intercept"], "-", color=FIT,
                linewidth=2.0, zorder=4)
        ax.set_title(p["name"], fontsize=11, color=INK_PRIMARY, loc="left", pad=8)
        ax.text(0.035, 0.955,
                f"R² = {s['r2']:.3f}\nρ = {s['rho']:.3f}\nslope = {s['slope']:.2f}",
                transform=ax.transAxes, fontsize=8.8, color=INK_SECONDARY,
                va="top", linespacing=1.6)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    for i, ax in enumerate(axes[:len(panels)]):
        if i % ncol == 0:
            ax.set_ylabel("actual burned pixels", fontsize=9.5, color=INK_SECONDARY)
        if i // ncol == nrow - 1:
            ax.set_xlabel("expected burned pixels", fontsize=9.5, color=INK_SECONDARY)

    fig.suptitle(f"Per-chip expected vs actual burned area — {year}",
                 fontsize=13.5, color=INK_PRIMARY, x=0.006, ha="left", y=0.995)
    fig.text(0.006, 0.005,
             "1,813 chips. Grey dashed = 1:1; orange = OLS fit. Model expectations are "
             "deflated for pos_weight 10; climatology and last-year burn are already on a "
             "natural scale.\nAxes are clipped at the 99.5th percentile, so a few extreme "
             "chips fall outside the frame but are included in every statistic. "
             "ρ is Spearman — prefer it to R² where the cloud is skewed.",
             fontsize=7.8, color=INK_SECONDARY, va="bottom")
    fig.tight_layout(rect=[0, 0.045, 1, 0.965])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"[scatter] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", required=True)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=DIR",
                    help="model chip dir for this year; repeatable")
    ap.add_argument("--label-dir", required=True,
                    help="directory of label_<year>.tif full-basin mosaics")
    ap.add_argument("--climatology", required=True)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    entries = []
    for item in args.model:
        if "=" not in item:
            raise SystemExit(f"--model needs NAME=DIR, got {item!r}")
        name, path = item.split("=", 1)
        entries.append((name.strip(), path))
    if not entries:
        raise SystemExit("need at least one --model")

    panels = []
    for name, path in entries:
        print(f"[scatter] {name}: {path}", flush=True)
        x, y = model_series(path, args.pos_weight)
        panels.append({"name": name, "x": x, "y": y, "stats": fit_stats(x, y)})

    ref = entries[0][1]
    print("[scatter] Climatology", flush=True)
    x, y = raster_series(ref, args.climatology, binarize=False)
    panels.append({"name": "Climatology", "x": x, "y": y, "stats": fit_stats(x, y)})

    prev = os.path.join(args.label_dir, f"label_{int(args.year) - 1}.tif")
    print(f"[scatter] Last-year burn ({int(args.year) - 1})", flush=True)
    x, y = raster_series(ref, prev, binarize=True)
    panels.append({"name": f"Last-year burn ({int(args.year) - 1})",
                   "x": x, "y": y, "stats": fit_stats(x, y)})

    import csv
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "model", "n_chips", "r2", "spearman_rho", "slope",
                    "intercept", "sum_expected", "sum_actual"])
        for p in panels:
            s = p["stats"]
            w.writerow([args.year, p["name"], len(p["x"]), f"{s['r2']:.6f}",
                        f"{s['rho']:.6f}", f"{s['slope']:.6f}",
                        f"{s['intercept']:.6f}", f"{p['x'].sum():.1f}",
                        f"{p['y'].sum():.1f}"])
    print(f"[scatter] wrote {args.out_csv}")

    plot(panels, args.year, args.out_png)


if __name__ == "__main__":
    main()
