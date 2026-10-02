"""Figure: one year's forecast expected burned fraction (A) and its difference from climatology (B).

Calibrated by default with the frozen Platt calibrator; --calibrator '' uses deflate + a level
factor from the model's own val years (never test years).
Usage: make_forecast_figure.py [--year 2026] [--risk_vmax 50 --diff_vmax 15]"""

import argparse
import os

import numpy as np

from aic_risk_modeling.eval.calibration import deflate, load_calibrator, to_prob
from aic_risk_modeling.eval.chips import basin_mask, block_nanmean, chip_pairs
from style import (CALIBRATOR, DIFF_CMAP, GRID, INK_PRIMARY, INK_SECONDARY, LABEL_DIR,
                   OUTSIDE, RISK_CMAP, SHP, SURFACE, save_figure)

PRED_ROOT = "out/cv/preds/factored_v3p_union4_monthlyattn_wide_yeargain/final_all"
BLOCK_MAP = 8
BLOCK_DIFF = 16    # ~7.4 km: smooths the k/13 steps of a 13-year pixel climatology


def level_factor(chip_dirs, pos_weight):
    """Pooled deflated-expected / actual burned pixels over the given chip dirs."""
    import rasterio as rio
    e = a = 0.0
    for d in chip_dirs:
        for o, mask in chip_pairs(d):
            with rio.open(o) as s:
                e += deflate(np.clip(s.read(1).astype(np.float64), 0, 1), pos_weight).sum()
            with rio.open(mask) as m:
                a += (m.read(1) > 0).sum()
    return e / a


def load(pred_path, label_dir, clim_years, shp, pos_weight, level, cal=None):
    import rasterio as rio

    with rio.open(pred_path) as s:
        q = s.read(1).astype(np.float64)
        tr, crs, shape = s.transform, s.crs, s.shape
    q = np.clip(q, 0, 1)
    pred = to_prob(q, pos_weight, level, cal)
    del q
    total = np.zeros(shape, np.float32)
    for y in clim_years:
        with rio.open(os.path.join(label_dir, f"label_{y}.tif")) as s:
            if s.transform != tr or s.shape != shape:
                raise ValueError(f"label_{y}.tif is not on the prediction grid")
            total += s.read(1) > 0
    clim = total / len(clim_years)
    inside, gdf = basin_mask(shp, crs, tr, shape)
    pred = np.where(inside, pred, np.nan)
    clim = np.where(inside, clim, np.nan)
    return pred, clim, tr, gdf


def plot(pred, clim, tr, gdf, out_png, year, clim_years, risk_vmax=None, diff_vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    risk = block_nanmean(pred, BLOCK_MAP) * 100
    diff = (block_nanmean(pred, BLOCK_DIFF) - block_nanmean(clim, BLOCK_DIFF)) * 100

    def extent(arr, b):
        h, w = arr.shape
        return [tr.c, tr.c + w * b * tr.a, tr.f + h * b * tr.e, tr.f]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 6.4), facecolor=SURFACE)
    vmax = risk_vmax if risk_vmax is not None else float(np.nanpercentile(risk, 98))
    im1 = a1.imshow(risk, extent=extent(risk, BLOCK_MAP), origin="upper", cmap=RISK_CMAP,
                    vmin=0, vmax=vmax, interpolation="nearest", zorder=2)
    dmax = diff_vmax if diff_vmax is not None else float(np.nanpercentile(np.abs(diff), 98))
    cmap = LinearSegmentedColormap.from_list("diff", DIFF_CMAP)
    im2 = a2.imshow(diff, extent=extent(diff, BLOCK_DIFF), origin="upper", cmap=cmap,
                    norm=TwoSlopeNorm(0, -dmax, dmax), interpolation="nearest", zorder=2)

    minx, miny, maxx, maxy = gdf.total_bounds
    for ax, im, label, letter in (
            (a1, im1, f"Predicted Burned Area, {year} (%)", "A"),
            (a2, im2, f"Difference from {clim_years[0]}–{clim_years[-1]} Mean "
                      "(Percentage Points)", "B")):
        ax.set_facecolor(OUTSIDE)
        gdf.boundary.plot(ax=ax, color=INK_SECONDARY, linewidth=0.7, zorder=3)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04,
                          extend="max" if letter == "A" else "both")
        cb.set_label(label, fontsize=10.5, color=INK_SECONDARY)
        cb.ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
        cb.outline.set_edgecolor(GRID)
        ax.text(0.0, 1.01, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=13, fontweight="bold", color=INK_PRIMARY)

    save_figure(fig, out_png, "forecast_fig", tight=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--pred", default=None, help="default <PRED_ROOT>/<year>/preds_out.tif")
    ap.add_argument("--level_chips", nargs="+", default=None,
                    help="with --calibrator '': val-year chip dirs for the level factor")
    ap.add_argument("--level_factor", type=float, default=None,
                    help="with --calibrator '': use this level factor")
    ap.add_argument("--calibrator", default=CALIBRATOR, metavar="NPZ",
                    help="frozen Platt calibrator; '' = deflate + level factor")
    ap.add_argument("--pos_weight", type=float, default=10.0)
    ap.add_argument("--label_dir", default=LABEL_DIR)
    ap.add_argument("--clim_years", default=None, help="default 2013-<year - 1>")
    ap.add_argument("--shp", default=SHP)
    ap.add_argument("--risk_vmax", type=float, default=None,
                    help="panel A max in %%; default 98th percentile")
    ap.add_argument("--diff_vmax", type=float, default=None,
                    help="panel B +/- limit in pp; default 98th percentile")
    ap.add_argument("--out_png", default=None, help="default out/figures/fig_forecast_<year>.png")
    a = ap.parse_args()

    if not a.calibrator and a.level_factor is None and not a.level_chips:
        ap.error("--calibrator '' needs --level_factor or --level_chips")
    cal = load_calibrator(a.calibrator) if a.calibrator else None
    if cal is not None:
        level = None
        print(f"[forecast_fig] calibrator {a.calibrator}")
    else:
        level = a.level_factor or level_factor(a.level_chips, a.pos_weight)
        print(f"[forecast_fig] level factor (deflated expected / actual) = {level:.3f}")
    lo, _, hi = (a.clim_years or f"2013-{a.year - 1}").partition("-")
    clim_years = list(range(int(lo), int(hi or lo) + 1))
    if max(clim_years) >= a.year:
        raise ValueError(f"climatology {clim_years[0]}-{clim_years[-1]} reaches the forecast year {a.year}")
    pred_path = a.pred or os.path.join(PRED_ROOT, str(a.year), "preds_out.tif")
    out_png = a.out_png or f"out/figures/fig_forecast_{a.year}.png"
    pred, clim, tr, gdf = load(pred_path, a.label_dir, clim_years, a.shp, a.pos_weight, level, cal)
    print(f"[forecast_fig] basin burned fraction: forecast {np.nanmean(pred):.4f}, "
          f"climatology {np.nanmean(clim):.4f} (ratio {np.nanmean(pred) / np.nanmean(clim):.2f})")
    plot(pred, clim, tr, gdf, out_png, a.year, clim_years, a.risk_vmax, a.diff_vmax)


if __name__ == "__main__":
    main()
