"""Figure: one year's fire-risk forecast (A) and its deviation from climatology (B).

  A  Basin map of the forecast expected burned fraction, per BLOCK_MAP-px block.
  B  Forecast minus climatology (mean union4 burn frequency over --clim_years,
     default 2013 .. --year minus 1, i.e. only years before the forecast year), per
     BLOCK_DIFF-px block, in percentage points: vermillion = more fire than normal,
     blue = less.

Model output is put on a burned-fraction scale in two steps:
  1. `deflate` inverts the weighted-BCE inflation (pos_weight), as in
     make_risk_figure_2024.py / scatter_expected_actual.py.
  2. A single LEVEL factor, expected / actual burned pixels pooled over the
     model's own evaluation years (--level_chips; for a CV fold these are its
     val years, never the 2024-25 test years), divides the deflated output. The
     deflated yeargain model under-predicts the level (~0.72 on fwdpair_2022), and
     uncorrected that bias would paint all of B "less fire than normal".
By default (--calibrator) both steps are replaced by the frozen Platt calibrator
fit on the CV fold-years 2018-23 (calibrated_year_totals.py --save-calibrator, the
one Fig 5 uses), applied to the raw output of the yeargain final_all model.
`--calibrator ''` restores the deflate + level path, which then needs an explicit
--level_factor or --level_chips (final_all has no val years to derive one from).
Both panels are masked to the RAISG outline and drawn in the grid's own MODIS
sinusoidal CRS (near-equatorial, so close to true shape).

    .venv/bin/python scripts/analysis/make_forecast_figure.py
    for y in 2024 2025 2026; do       # comparable set: shared colour scales
        .venv/bin/python scripts/analysis/make_forecast_figure.py --year $y --risk_vmax 50 --diff_vmax 15
    done
    .venv/bin/python scripts/analysis/make_forecast_figure.py --calibrator '' \
        --pred out/cv/preds/<arch>/fwdpair_2022/2026/preds_out.tif --level_factor 0.717
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scatter_expected_actual import (  # noqa: E402
    GRID, INK_PRIMARY, INK_SECONDARY, SURFACE, deflate, load_calibrator)

RISK_CMAP = "YlOrRd"
DIFF_CMAP = ["#0072b2", "#f7f7f7", "#d55e00"]   # less fire -> normal -> more fire
OUTSIDE = "#f2f1ee"

PRED_ROOT = "out/cv/preds/factored_v3p_union4_monthlyattn_wide_yeargain/final_all"
CALIBRATOR = "out/cv/calibrator_platt_cv2018_2023.npz"
LABEL_DIR = "out/label_mosaics_v3p_union4"
SHP = "../data/Limites_RAISG_2025/Lim_Raisg.shp"
BLOCK_MAP = 8      # ~3.7 km
BLOCK_DIFF = 16    # ~7.4 km: smooths the k/13 steps of a 13-year pixel climatology


def level_factor(chip_dirs, pos_weight):
    """Pooled deflated-expected / actual burned pixels over the given chip dirs."""
    import rasterio as rio
    e = a = 0.0
    for d in chip_dirs:
        outs = glob.glob(os.path.join(d, "out_*.tif"))
        if not outs:
            raise FileNotFoundError(f"no chips in {d}")
        for o in outs:
            with rio.open(o) as s:
                e += deflate(np.clip(s.read(1).astype(np.float64), 0, 1), pos_weight).sum()
            with rio.open(o.replace("out_", "mask_", 1)) as m:
                a += (m.read(1) > 0).sum()
    return e / a


def block_mean(arr, b):
    """Mean over b x b blocks, NaN-aware (a block is NaN only if all of it is)."""
    h, w = (arr.shape[0] // b) * b, (arr.shape[1] // b) * b
    v = arr[:h, :w].reshape(h // b, b, w // b, b)
    with warnings.catch_warnings():                 # all-NaN blocks outside the basin
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, axis=(1, 3))


def load(pred_path, label_dir, clim_years, shp, pos_weight, level, cal=None):
    import geopandas as gpd
    import rasterio as rio
    from rasterio.features import rasterize

    with rio.open(pred_path) as s:
        q = s.read(1).astype(np.float64)
        tr, crs, shape = s.transform, s.crs, s.shape
    q = np.clip(q, 0, 1)
    pred = cal(q) if cal is not None else deflate(q, pos_weight) / level
    del q
    total = np.zeros(shape, np.float32)
    for y in clim_years:
        with rio.open(os.path.join(label_dir, f"label_{y}.tif")) as s:
            if s.transform != tr or s.shape != shape:
                raise ValueError(f"label_{y}.tif is not on the prediction grid")
            total += s.read(1) > 0
    clim = total / len(clim_years)
    gdf = gpd.read_file(shp).to_crs(crs)
    inside = rasterize(((g, 1) for g in gdf.geometry), out_shape=shape, transform=tr,
                       fill=0, dtype="uint8") > 0
    pred = np.where(inside, pred, np.nan)
    clim = np.where(inside, clim, np.nan)
    return pred, clim, tr, gdf


def plot(pred, clim, tr, gdf, out_png, year, clim_years, risk_vmax=None, diff_vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    risk = block_mean(pred, BLOCK_MAP) * 100
    diff = (block_mean(pred, BLOCK_DIFF) - block_mean(clim, BLOCK_DIFF)) * 100

    def extent(arr, b):
        h, w = arr.shape
        return [tr.c, tr.c + w * b * tr.a, tr.f + h * b * tr.e, tr.f]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 6.4), facecolor=SURFACE)
    # auto (98th percentile) scales differ by year; pass both to compare years
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

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[forecast_fig] wrote {out_png} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--pred", default=None, help="default <PRED_ROOT>/<year>/preds_out.tif")
    ap.add_argument("--level_chips", nargs="+", default=None,
                    help="with --calibrator '': chip dirs (with labels) for the level "
                         "factor, from the --pred model's own val years; never test years")
    ap.add_argument("--level_factor", type=float, default=None,
                    help="with --calibrator '': skip the chip pass and use this factor")
    ap.add_argument("--calibrator", default=CALIBRATOR, metavar="NPZ",
                    help="frozen Platt calibrator; replaces deflate + level factor. "
                         "Pass '' to use --level_factor / --level_chips instead")
    ap.add_argument("--pos_weight", type=float, default=10.0)
    ap.add_argument("--label_dir", default=LABEL_DIR)
    ap.add_argument("--clim_years", default=None, help="default 2013-<year - 1>")
    ap.add_argument("--shp", default=SHP)
    ap.add_argument("--risk_vmax", type=float, default=None,
                    help="panel A colour-scale max in %%; default = this year's 98th percentile")
    ap.add_argument("--diff_vmax", type=float, default=None,
                    help="panel B +/- limit in pp; default = this year's 98th percentile of |diff|")
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
