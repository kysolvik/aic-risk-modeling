"""Figure: which driver group raises fire risk where it is high (Shapley attribution).

One column per year (default 2023, 2024, 2025), from the yeargain final_all
Shapley mosaics (docker/download_cv_attr.sh -> out/cv/attr/<arch>/final_all/<y>/
attr_shap.tif; 5 driver groups, configs/attribution_drivers_v3p_yeargain.json).

  Top row     Dominant driver: per BLOCK-px block, the group with the largest mean
              Shapley value, drawn only where the block's mean calibrated burn
              probability is >= --threshold; lower-risk blocks are grey.
  Bottom row  Mean Shapley value of each group by calibrated-risk class, with the
              class's share of the basin in parentheses under each tick.

Why only high-risk blocks: Shapley values are relative to a synthetic grid-average
baseline pixel. For low-risk pixels (mostly intact forest) land use and fire
history carry large, opposite-signed values (+0.16 / -0.13 on 2024 pixels under
0.2% risk) that cancel -- an interaction split between the two groups, not "land
use raises risk in intact forest". The bottom row shows that offset rather than
mapping it.

Scales: attr_shap.tif bands are DEFLATED probabilities (pos_weight 10). Risk
classes use the frozen Platt calibrator (same as Figs 4, 5) applied to the
re-inflated risk band (inflate(band 1) == predict's preds_out.tif to 1e-6); the
Shapley values themselves stay on the deflated scale (Platt is nonlinear, so
there is no additive calibrated version), plotted in percentage points.
Year-level shifts (gamma, year gain) are not players: they sit in the baseline
band, so this is within-year spatial attribution.

    .venv/bin/python scripts/figures/make_shapley_figure.py
    .venv/bin/python scripts/figures/make_shapley_figure.py --from_cache   # restyle only
"""

import argparse
import os

import numpy as np

from aic_risk_modeling.eval.calibration import inflate, load_calibrator
from aic_risk_modeling.eval.chips import basin_mask, block_nanmean
from style import (CALIBRATOR, GRID, INK_PRIMARY, INK_SECONDARY, OUTSIDE, SHP, SURFACE,
                   save_figure)

ATTR_ROOT = "out/cv/attr/factored_v3p_union4_monthlyattn_wide_yeargain/final_all"
CACHE = "out/figures/fig_shapley_drivers_cache.npz"
POS_WEIGHT = 10.0
BLOCK = 8          # ~3.7 km, as the forecast map
# (band description, label, colour): Okabe-Ito; land use vs fire history (the two
# groups that dominate the map) get the most distinct pair. Palette validated
# all-pairs (dataviz validate_palette.js); worst CVD pair green/purple 7.6 is
# carried by the legend labels and terrain is almost never dominant.
GROUPS = [
    ("shapley_land_use_human", "Land Use + Human", "#0072b2"),
    ("shapley_fire_history", "Fire History", "#d55e00"),
    ("shapley_climate_weather", "Climate + Weather", "#e69f00"),
    ("shapley_vegetation", "Vegetation", "#009e73"),
    ("shapley_terrain_water", "Terrain + Water", "#cc79a7"),
]
LOW_RISK = "#d9d8d3"
CLASS_EDGES = [0.0, 0.01, 0.05, 0.20, 1.01]
CLASS_LABELS = ["< 1%", "1–5%", "5–20%", "≥ 20%"]


def year_summary(path, cal, inside):
    """Block means (risk + groups) and per-class group means for one year."""
    import rasterio as rio
    with rio.open(path) as s:
        desc = list(s.descriptions)
        idx = [desc.index(g[0]) + 1 for g in GROUPS]
        p = s.read(1).astype(np.float64)
        risk = np.where(inside, cal(inflate(p, POS_WEIGHT)), np.nan)
        del p
        cls = np.digitize(risk, CLASS_EDGES) - 1          # NaN -> len(edges)-1, dropped
        valid = inside & (cls >= 0) & (cls < len(CLASS_LABELS))
        cls_share = np.bincount(cls[valid], minlength=len(CLASS_LABELS)) / valid.sum()
        blocks = [block_nanmean(risk, BLOCK)]
        cls_mean = np.zeros((len(GROUPS), len(CLASS_LABELS)))
        for gi, bi in enumerate(idx):
            v = s.read(bi).astype(np.float64)
            cls_mean[gi] = (np.bincount(cls[valid], weights=v[valid], minlength=len(CLASS_LABELS))
                            / np.bincount(cls[valid], minlength=len(CLASS_LABELS)))
            blocks.append(block_nanmean(np.where(inside, v, np.nan), BLOCK))
            del v
    return np.stack(blocks).astype(np.float32), cls_mean, cls_share


def compute(years, attr_root, calibrator, shp):
    import rasterio as rio

    cal = load_calibrator(calibrator)
    first = os.path.join(attr_root, str(years[0]), "attr_shap.tif")
    with rio.open(first) as s:
        tr, crs, shape = s.transform, s.crs, s.shape
    inside, gdf = basin_mask(shp, crs, tr, shape)
    out = {"years": np.array(years), "transform": np.array(tr)[:6]}
    for y in years:
        path = os.path.join(attr_root, str(y), "attr_shap.tif")
        with rio.open(path) as s:
            if s.transform != tr or s.shape != shape:
                raise ValueError(f"{path} is not on the {years[0]} grid")
        blocks, cls_mean, cls_share = year_summary(path, cal, inside)
        out[f"blocks_{y}"], out[f"cls_mean_{y}"], out[f"cls_share_{y}"] = blocks, cls_mean, cls_share
        print(f"[shapley_fig] {y}: class shares " +
              ", ".join(f"{l} {v:.1%}" for l, v in zip(CLASS_LABELS, cls_share)))
    return out, gdf


def dominant_map(blocks, threshold):
    """0..G-1 = dominant group, G = low risk, NaN = outside the basin."""
    risk, shap = blocks[0], blocks[1:]
    dom = np.nanargmax(np.where(np.isfinite(shap), shap, -np.inf), axis=0).astype(np.float32)
    dom[risk < threshold] = len(GROUPS)
    dom[~np.isfinite(risk)] = np.nan
    return dom


def write_csv(data, years, threshold, path):
    import pandas as pd
    rows = []
    for y in years:
        dom = dominant_map(data[f"blocks_{y}"], threshold)
        hi = dom[np.isfinite(dom) & (dom < len(GROUPS))].astype(int)
        share = np.bincount(hi, minlength=len(GROUPS)) / max(hi.size, 1)
        for gi, (_, label, _) in enumerate(GROUPS):
            r = {"year": y, "group": label, "dominant_share_high_risk_blocks": share[gi]}
            for ci, cl in enumerate(CLASS_LABELS):
                r[f"mean_shap_pp_{cl}"] = data[f"cls_mean_{y}"][gi, ci] * 100
                r[f"area_share_{cl}"] = data[f"cls_share_{y}"][ci]
            rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.5g")
    print(f"[shapley_fig] wrote {path}")


def plot(data, gdf, years, threshold, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    tr = data["transform"]          # affine a, b, c, d, e, f
    a, c, e, f = tr[0], tr[2], tr[4], tr[5]
    n = len(years)
    fig = plt.figure(figsize=(4.3 * n, 8.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, n, height_ratios=[1.25, 1.0], hspace=0.26, wspace=0.08)
    cmap = ListedColormap([g[2] for g in GROUPS] + [LOW_RISK])
    minx, miny, maxx, maxy = gdf.total_bounds
    letters = "ABCDEFGHIJ"

    bar_axes = []
    ymax = max(np.abs(data[f"cls_mean_{y}"]).max() for y in years) * 100 * 1.08
    for j, y in enumerate(years):
        dom = dominant_map(data[f"blocks_{y}"], threshold)
        h, w = dom.shape
        ax = fig.add_subplot(gs[0, j])
        ax.set_facecolor(OUTSIDE)
        ax.imshow(np.ma.masked_invalid(dom), cmap=cmap, vmin=-0.5, vmax=len(GROUPS) + 0.5,
                  extent=[c, c + w * BLOCK * a, f + h * BLOCK * e, f], origin="upper",
                  interpolation="nearest", zorder=2)
        gdf.boundary.plot(ax=ax, color=INK_SECONDARY, linewidth=0.6, zorder=3)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.text(0.0, 1.015, letters[j], transform=ax.transAxes, ha="left", va="bottom",
                fontsize=13, fontweight="bold", color=INK_PRIMARY)
        ax.text(0.07, 1.015, str(y), transform=ax.transAxes, ha="left", va="bottom",
                fontsize=12, color=INK_PRIMARY)

        bx = fig.add_subplot(gs[1, j], sharey=bar_axes[0] if bar_axes else None)
        bar_axes.append(bx)
        means = data[f"cls_mean_{y}"] * 100
        shares = data[f"cls_share_{y}"]
        k = len(GROUPS)
        width = 0.8 / k
        x = np.arange(len(CLASS_LABELS))
        for gi, (_, label, col) in enumerate(GROUPS):
            bx.bar(x - 0.4 + width * (gi + 0.5), means[gi], width=width * 0.86, color=col,
                   label=label, zorder=2)
        bx.axhline(0, color=INK_SECONDARY, linewidth=0.8, zorder=3)
        bx.set_xticks(x)
        bx.set_xticklabels([f"{l}\n({s:.0%})" for l, s in zip(CLASS_LABELS, shares)],
                           fontsize=9, color=INK_SECONDARY)
        bx.set_ylim(-ymax, ymax)
        bx.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        bx.set_facecolor(SURFACE)
        for side in ("top", "right"):
            bx.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            bx.spines[side].set_color(GRID)
        bx.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
        bx.set_xlabel("Predicted Burn Probability (Share of Area)", fontsize=10, color=INK_SECONDARY)
        if j == 0:
            bx.set_ylabel("Mean Shapley Value (Percentage Points)", fontsize=10,
                          color=INK_SECONDARY)
        else:
            plt.setp(bx.get_yticklabels(), visible=False)
        bx.text(0.0, 1.03, letters[n + j], transform=bx.transAxes, ha="left", va="bottom",
                fontsize=13, fontweight="bold", color=INK_PRIMARY)

    handles = [Patch(facecolor=g[2], label=g[1]) for g in GROUPS]
    handles.append(Patch(facecolor=LOW_RISK, label=f"Burn Probability < {threshold:.0%}"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.502), ncol=len(handles),
               frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, handlelength=1.2,
               columnspacing=1.4)

    save_figure(fig, out_png, "shapley_fig", tight=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--attr_root", default=ATTR_ROOT)
    ap.add_argument("--calibrator", default=CALIBRATOR)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="calibrated burn probability above which a block gets a dominant driver")
    ap.add_argument("--shp", default=SHP)
    ap.add_argument("--out_png", default="out/figures/fig_shapley_drivers.png")
    ap.add_argument("--from_cache", action="store_true",
                    help=f"reuse {CACHE} (block means + class means) instead of reading the mosaics")
    a = ap.parse_args()

    if a.from_cache:
        import geopandas as gpd
        import rasterio as rio
        data = dict(np.load(CACHE))
        if list(data["years"]) != a.years:
            raise ValueError(f"cache has years {list(data['years'])}, asked for {a.years}")
        with rio.open(os.path.join(a.attr_root, str(a.years[0]), "attr_shap.tif")) as s:
            gdf = gpd.read_file(a.shp).to_crs(s.crs)
    else:
        data, gdf = compute(a.years, a.attr_root, a.calibrator, a.shp)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, **data)
    write_csv(data, a.years, a.threshold, os.path.splitext(a.out_png)[0] + ".csv")
    plot(data, gdf, a.years, a.threshold, a.out_png)


if __name__ == "__main__":
    main()
