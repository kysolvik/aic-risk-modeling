"""Per-year figure: calibrated risk (A) and per-group Shapley maps (B..F) with shared colour scales.

Reads make_shapley_figure.py's block-mean cache. Shapley values are relative to a synthetic
grid-average pixel and on the deflated scale; A is calibrated.
Usage: make_shapley_maps.py [--years 2024]"""

import argparse
import os

import numpy as np

from make_shapley_figure import ATTR_ROOT, BLOCK, CACHE, GROUPS, compute
from style import (CALIBRATOR, DIFF_CMAP, GRID, INK_PRIMARY, INK_SECONDARY, OUTSIDE,
                   RISK_CMAP, SHP, SURFACE, save_figure)


def load_cache(years, shp):
    import geopandas as gpd
    import rasterio as rio
    if os.path.exists(CACHE):
        data = dict(np.load(CACHE))
        missing = [y for y in years if f"blocks_{y}" not in data]
        if not missing:
            with rio.open(os.path.join(ATTR_ROOT, str(years[0]), "attr_shap.tif")) as s:
                return data, gpd.read_file(shp).to_crs(s.crs)
        print(f"[shapley_maps] cache lacks {missing}; rebuilding")
    data, gdf = compute(years, ATTR_ROOT, CALIBRATOR, shp)
    np.savez_compressed(CACHE, **data)
    return data, gdf


def plot_year(data, gdf, year, risk_vmax, shap_vmax, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    a, c, e, f = (data["transform"][i] for i in (0, 2, 4, 5))
    blocks = data[f"blocks_{year}"]
    h, w = blocks.shape[1:]
    extent = [c, c + w * BLOCK * a, f + h * BLOCK * e, f]
    minx, miny, maxx, maxy = gdf.total_bounds
    diff_cmap = LinearSegmentedColormap.from_list("diff", DIFF_CMAP)
    diff_norm = TwoSlopeNorm(0, -shap_vmax, shap_vmax)

    fig = plt.figure(figsize=(13.2, 9.4), facecolor=SURFACE)
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 0.035, 0.17, 1, 0.035],
                          hspace=0.12, wspace=0.05)
    axes = np.array([[fig.add_subplot(gs[r, k]) for k in range(3)] for r in (0, 3)])
    panels = [(blocks[0] * 100, "Burn Probability", RISK_CMAP, None)]
    panels += [(blocks[gi + 1] * 100, label, diff_cmap, diff_norm)
               for gi, (_, label, _) in enumerate(GROUPS)]
    ims = []
    for k, (ax, (arr, label, cmap, norm)) in enumerate(zip(axes.flat, panels)):
        ax.set_facecolor(OUTSIDE)
        kw = {"norm": norm} if norm is not None else {"vmin": 0, "vmax": risk_vmax}
        ims.append(ax.imshow(np.ma.masked_invalid(arr), extent=extent, origin="upper", cmap=cmap,
                             interpolation="nearest", zorder=2, **kw))
        gdf.boundary.plot(ax=ax, color=INK_SECONDARY, linewidth=0.6, zorder=3)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.text(0.0, 1.015, "ABCDEF"[k], transform=ax.transAxes, ha="left", va="bottom",
                fontsize=13, fontweight="bold", color=INK_PRIMARY)
        ax.text(0.065, 1.015, label, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, color=INK_PRIMARY)

    def colorbar(im, cax, label, extend):
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend)
        cb.set_label(label, fontsize=10.5, color=INK_SECONDARY)
        cb.ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
        cb.outline.set_edgecolor(GRID)

    colorbar(ims[0], fig.add_subplot(gs[1, 0]), "Calibrated Burn Probability (%)", "max")
    colorbar(ims[1], fig.add_subplot(gs[4, 1]), "Shapley Value (Percentage Points)", "both")

    save_figure(fig, out_png, "shapley_maps", tight=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--risk_vmax", type=float, default=None,
                    help="%%; default pooled 98th percentile")
    ap.add_argument("--shap_vmax", type=float, default=None,
                    help="pp; default pooled 98th percentile of |value|")
    ap.add_argument("--shp", default=SHP)
    ap.add_argument("--out_dir", default="out/figures")
    a = ap.parse_args()

    data, gdf = load_cache(a.years, a.shp)
    cached = [int(y) for y in data["years"]]
    stack = np.stack([data[f"blocks_{y}"] for y in cached])
    risk_vmax = a.risk_vmax or float(np.nanpercentile(stack[:, 0] * 100, 98))
    shap_vmax = a.shap_vmax or float(np.nanpercentile(np.abs(stack[:, 1:]) * 100, 98))
    print(f"[shapley_maps] shared scales over {cached}: risk 0-{risk_vmax:.1f}%, "
          f"Shapley +/-{shap_vmax:.1f} pp")
    for y in a.years:
        plot_year(data, gdf, y, risk_vmax, shap_vmax,
                  os.path.join(a.out_dir, f"fig_shapley_maps_{y}.png"))


if __name__ == "__main__":
    main()
