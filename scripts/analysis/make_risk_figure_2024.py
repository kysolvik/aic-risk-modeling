"""Publication figure: factored_v1 2024 fire-risk at three zoom levels.

Three columns, left to right:

  A. Basin map of 2024 fire risk (factored_v1 output), RAISG outline, with the
     three example chips boxed.
  B. Per-chip expected-vs-actual burned pixels for factored_v1, last-year burn and
     climatology (the cloud behind the year-total headline), with the three example
     chips highlighted in every panel.
  C. Zoomed risk crops for the three example chips (under- / well- / over-predicted),
     each with the 2024 actual-burn footprint outlined.

Everything is computed on ONE calibrated scale -- deflated expected burned pixels
per chip -- reusing the per-predictor definitions in `scatter_expected_actual.py`:
models are deflated for the weighted-BCE pos_weight (10), climatology is a natural
burn-frequency sum, last-year burn is the previous year's 0/1 mask.

    .venv/bin/python scripts/analysis/make_risk_figure_2024.py

Inputs (all present locally, one 7296x6272 EPSG:4326 grid):
  out/baselines/factored_v1/2024_out.tif        risk mosaic (raw probability)
  out/baselines/factored_v1/2024/{out_,mask_}*  1,813 per-chip tiles (prob + label)
  out/label_mosaics/{label_2023,climatology_2013_2022}.tif
  ../data/Limites_RAISG_2025/Lim_Raisg.shp      basin outline
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Sibling import: the script's own directory is sys.path[0] when run directly, so
# the per-predictor math stays defined in exactly one place.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scatter_expected_actual import (  # noqa: E402
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    ONE_TO_ONE,
    POINT,
    SURFACE,
    chip_pairs,
    deflate,
    fit_stats,
)

# Okabe-Ito colourblind-safe trio, reused for each chip across A, B and C.
CHIP_COLORS = {"under": "#0072b2", "right": "#009e73", "over": "#d55e00"}
CHIP_LABELS = {"under": "Under-predicted", "right": "Well-predicted",
               "over": "Over-predicted"}
CHIP_ORDER = ["under", "right", "over"]  # top-to-bottom in panels B and C
RISK_CMAP = "YlOrRd"
BURN_CMAP = ["#f3ede2", "#7f0000"]  # unburned cream, burned dark red (YlOrRd top)


# --------------------------------------------------------------------------- #
# Per-chip table (single pass over the 1,813 tiles)
# --------------------------------------------------------------------------- #
def per_chip_table(chip_dir, clim_path, prev_label_path, pos_weight):
    """One row per chip: id, geographic bounds, actual + 3 expected burned-pixel counts."""
    import rasterio as rio

    pairs = chip_pairs(chip_dir)
    rows = []
    clim = rio.open(clim_path)
    prev = rio.open(prev_label_path)
    try:
        for i, (out_path, mask_path) in enumerate(pairs):
            chip_id = os.path.basename(out_path)[len("out_"):-len(".tif")]
            with rio.open(out_path) as s:
                q = np.clip(s.read(1).astype(np.float64), 0.0, 1.0)
            with rio.open(mask_path) as m:
                lab = m.read(1) > 0
                b = m.bounds
            exp_factored = float(deflate(q, pos_weight).sum())
            actual = float(lab.sum())
            exp_clim = _window_sum(clim, b, lab.shape, binarize=False)
            exp_lastyear = _window_sum(prev, b, lab.shape, binarize=True)
            rows.append(dict(
                chip_id=chip_id, left=b.left, bottom=b.bottom, right=b.right,
                top=b.top, lon=(b.left + b.right) / 2, lat=(b.bottom + b.top) / 2,
                actual=actual, exp_factored=exp_factored, exp_clim=exp_clim,
                exp_lastyear=exp_lastyear, out_path=out_path, mask_path=mask_path))
            if (i + 1) % 300 == 0:
                print(f"  ...{i + 1}/{len(pairs)} chips", flush=True)
    finally:
        clim.close()
        prev.close()
    return pd.DataFrame(rows)


def _window_sum(src, bounds, chip_shape, binarize):
    """Sum a full-basin raster over one chip's window (mirrors raster_series)."""
    from rasterio.windows import from_bounds
    win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
    v = src.read(1, window=win).astype(np.float64)
    if v.shape != chip_shape:
        raise ValueError(f"window {v.shape} != chip {chip_shape}")
    return float((v > 0).sum() if binarize else v.sum())


GRID_STEP = 0.64  # chip centres sit on a regular 0.64-degree lon/lat grid


def add_interior_flag(df):
    """Flag chips whose 8 grid neighbours all exist (i.e. not on the ragged basin edge)."""
    present = set(zip(df["lon"].round(2), df["lat"].round(2)))
    flags = []
    for lo, la in zip(df["lon"].round(2), df["lat"].round(2)):
        ok = all(
            (round(lo + dlo, 2), round(la + dla, 2)) in present
            for dlo in (-GRID_STEP, 0, GRID_STEP)
            for dla in (-GRID_STEP, 0, GRID_STEP)
            if not (dlo == 0 and dla == 0))
        flags.append(ok)
    df["interior"] = flags
    return df


def display_cap(df):
    """99.5th-percentile of expected+actual across predictors = the scatter axis top."""
    vals = np.concatenate([df["exp_factored"].values, df["exp_lastyear"].values,
                           df["exp_clim"].values, df["actual"].values])
    return float(np.percentile(vals, 99.5))


def select_chips(df, min_actual_pct, cap):
    """Under- / well- / over-predicted chips: real fire, interior, and on the scatter scale.

    On-scale (actual & expected <= the 99.5-pct axis cap) keeps every example visible in
    panel B; interior avoids chips clipped by the basin boundary.
    """
    cutoff = np.percentile(df["actual"], min_actual_pct)
    pool = df[(df["actual"] >= cutoff) & df["interior"]
              & (df["actual"] <= cap) & (df["exp_factored"] <= cap)].copy()
    if len(pool) < 3:
        raise RuntimeError("too few eligible chips; lower --min-actual-pct")
    pool["resid"] = pool["exp_factored"] - pool["actual"]
    picks = {
        "over": pool.loc[pool["resid"].idxmax()],
        "under": pool.loc[pool["resid"].idxmin()],
        "right": pool.loc[pool["resid"].abs().idxmin()],
    }
    if len({picks[k]["chip_id"] for k in picks}) != 3:
        raise RuntimeError("selected chips are not distinct; adjust --min-actual-pct")
    return picks


# --------------------------------------------------------------------------- #
# Raster reads for the maps
# --------------------------------------------------------------------------- #
def read_basin_map(map_path, shp_path, target_width, pos_weight):
    """Decimated, deflated risk map masked to the RAISG basin. Returns (arr, extent, vmax)."""
    import geopandas as gpd
    import rasterio as rio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds as tr_from_bounds

    with rio.open(map_path) as s:
        scale = target_width / s.width
        h, w = int(round(s.height * scale)), target_width
        q = s.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float64)
        b = s.bounds
    arr = deflate(np.clip(q, 0.0, 1.0), pos_weight)

    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    tr = tr_from_bounds(b.left, b.bottom, b.right, b.top, w, h)
    basin = rasterize(((g, 1) for g in gdf.geometry), out_shape=(h, w),
                      transform=tr, fill=0, dtype="uint8")
    arr = np.where(basin > 0, arr, np.nan)

    vmax = float(np.nanpercentile(arr, 98))
    extent = [b.left, b.right, b.bottom, b.top]
    return arr, extent, vmax, gdf


def read_crop(out_path, mask_path, pos_weight):
    """Full-res deflated risk crop + actual-burn mask for one chip. Returns (prob, burn, extent)."""
    import rasterio as rio
    with rio.open(out_path) as s:
        q = np.clip(s.read(1).astype(np.float64), 0.0, 1.0)
        b = s.bounds
    with rio.open(mask_path) as m:
        burn = (m.read(1) > 0).astype(float)
    prob = deflate(q, pos_weight)
    return prob, burn, [b.left, b.right, b.bottom, b.top]


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(df, picks, cap, map_arr, map_extent, vmax, gdf, pos_weight, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(13.6, 15.0), facecolor=SURFACE)

    # One outer frame divided into three connected panels: A (map) and B (scatter)
    # share the top row, split by a vertical divider at XB; C (chips) is the whole
    # bottom, split from the top by a horizontal divider at YB. All axes are placed
    # by hand so they line up to this grid.
    BX0, BX1 = 0.050, 0.968        # frame left / right
    BY0, BY1 = 0.035, 0.965        # frame bottom / top
    XB = 0.560                     # vertical divider (A | B), top row only
    YB = 0.595                     # horizontal divider (A,B above | C below)
    LW = 1.7

    # ---- Panel A: basin map + colorbar (top-left) --------------------------
    ax_map = fig.add_axes([0.098, YB + 0.050, 0.360, BY1 - (YB + 0.050) - 0.014])
    ax_map.set_facecolor("#f2f1ee")
    im = ax_map.imshow(map_arr, extent=map_extent, origin="upper", cmap=RISK_CMAP,
                       vmin=0, vmax=vmax, interpolation="nearest")
    gdf.boundary.plot(ax=ax_map, color=INK_SECONDARY, linewidth=0.7, zorder=3)
    for key in CHIP_ORDER:
        r = picks[key]
        c = CHIP_COLORS[key]
        ax_map.add_patch(Rectangle(
            (r["left"], r["bottom"]), r["right"] - r["left"], r["top"] - r["bottom"],
            fill=False, edgecolor=c, linewidth=2.4, zorder=5))
        ax_map.annotate(
            key[0].upper(), xy=(r["right"], r["top"]), xytext=(3, 3),
            textcoords="offset points", color="white", fontsize=11, fontweight="bold",
            ha="left", va="bottom", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.15", facecolor=c, edgecolor="white", lw=0.8))
    ax_map.set_xlabel("Longitude", fontsize=11, color=INK_SECONDARY)
    ax_map.set_ylabel("Latitude", fontsize=11, color=INK_SECONDARY)
    ax_map.tick_params(colors=INK_SECONDARY, labelsize=9.5)
    ax_map.set_aspect("equal")
    ax_map.set_anchor("N")  # push the aspect-shrunk map to the top of its cell
    minx, miny, maxx, maxy = gdf.total_bounds
    ax_map.set_xlim(minx, maxx)
    ax_map.set_ylim(miny, maxy)
    cax = fig.add_axes([XB - 0.074, YB + 0.070, 0.015, 0.28])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Calibrated burn probability (deflated)", fontsize=10,
                 color=INK_SECONDARY)
    cb.ax.tick_params(labelsize=9, colors=INK_SECONDARY)

    # ---- Panel B: single model scatter (top-right) -------------------------
    hi = cap * 1.05
    ax_b = fig.add_axes([XB + 0.072, YB + 0.052, BX1 - (XB + 0.072) - 0.012,
                         BY1 - (YB + 0.052) - 0.016])
    ax_b.set_facecolor(SURFACE)
    ax_b.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax_b.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax_b.spines[sp].set_color(GRID)
    x, y = df["exp_factored"].values, df["actual"].values
    ax_b.plot([0, hi], [0, hi], "--", color=ONE_TO_ONE, linewidth=1.3, zorder=2)
    ax_b.scatter(x, y, s=9, color=POINT, alpha=0.22, linewidths=0, zorder=3)
    s = fit_stats(x, y)
    for key in CHIP_ORDER:
        r = picks[key]
        ax_b.scatter([float(r["exp_factored"])], [float(r["actual"])], s=185,
                     facecolor=CHIP_COLORS[key], edgecolor="black", linewidth=1.5,
                     zorder=6)
    ax_b.text(0.96, 0.06, f"R² = {s['r2']:.3f}", transform=ax_b.transAxes,
              fontsize=12.5, color=INK_SECONDARY, va="bottom", ha="right")
    ax_b.set_xlim(0, hi)
    ax_b.set_ylim(0, hi)
    ax_b.set_aspect("equal")
    ax_b.set_anchor("N")
    ax_b.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    ax_b.set_xlabel("Expected burned pixels", fontsize=11.5, color=INK_SECONDARY)
    ax_b.set_ylabel("Actual burned pixels", fontsize=11.5, color=INK_SECONDARY)

    # ---- Panel C: 2 rows (predicted / actual) x 3 cols (chips) -------------
    burn_cmap = ListedColormap(BURN_CMAP)
    c_left, c_right = 0.092, 0.956
    c_top, c_bottom = YB - 0.044, BY0 + 0.012
    col_slot = (c_right - c_left) / 3.0
    row_slot = (c_top - c_bottom) / 2.0
    gx, gy = 0.010, 0.010  # inner gap so squares don't touch
    for ccol, key in enumerate(CHIP_ORDER):
        r = picks[key]
        c = CHIP_COLORS[key]
        prob, burn, ext = read_crop(r["out_path"], r["mask_path"], pos_weight)
        axes_col = []
        for rrow, (arr, cmap, vlim) in enumerate(
                [(prob, RISK_CMAP, vmax), (burn, burn_cmap, 1)]):
            rect = [c_left + ccol * col_slot + gx,
                    c_top - (rrow + 1) * row_slot + gy,
                    col_slot - 2 * gx, row_slot - 2 * gy]
            a = fig.add_axes(rect)
            a.imshow(arr, extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=vlim,
                     interpolation="nearest")
            for sp in a.spines.values():
                sp.set_color(c)
                sp.set_linewidth(2.6)
            a.set_xticks([])
            a.set_yticks([])
            a.set_anchor("N" if rrow == 0 else "S")
            axes_col.append(a)
        axes_col[0].set_title(f"{CHIP_LABELS[key]}   (Expected {r['exp_factored']:.0f} · "
                              f"Actual {r['actual']:.0f})", fontsize=11.5, color=c, pad=5)
        if ccol == 0:
            axes_col[0].set_ylabel("Predicted risk", fontsize=11.5, color=INK_SECONDARY)
            axes_col[1].set_ylabel("Actual burn", fontsize=11.5, color=INK_SECONDARY)

    # ---- outer frame + the two internal dividers ---------------------------
    fig.add_artist(Rectangle((BX0, BY0), BX1 - BX0, BY1 - BY0,
                             transform=fig.transFigure, fill=False,
                             edgecolor="black", linewidth=LW, zorder=20))
    for xy in ([[BX0, BX1], [YB, YB]],       # horizontal divider (full width)
               [[XB, XB], [YB, BY1]]):        # vertical divider (top row only)
        ln = Line2D(xy[0], xy[1], color="black", linewidth=LW, zorder=20)
        ln.set_transform(fig.transFigure)
        fig.add_artist(ln)
    for letter, (lx, ly) in [("A", (BX0, BY1)), ("B", (XB, BY1)), ("C", (BX0, YB))]:
        fig.text(lx + 0.007, ly - 0.009, letter, fontsize=17, fontweight="bold",
                 color=INK_PRIMARY, ha="left", va="top", zorder=21,
                 bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                           edgecolor="none"))

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor=SURFACE)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", facecolor=SURFACE)
    plt.close(fig)
    print(f"[figure] wrote {out_png} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip-dir", default="out/baselines/factored_v1/2024")
    ap.add_argument("--map", default="out/baselines/factored_v1/2024_out.tif")
    ap.add_argument("--climatology",
                    default="out/label_mosaics/climatology_2013_2022.tif")
    ap.add_argument("--label-dir", default="out/label_mosaics")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--shp", default="../data/Limites_RAISG_2025/Lim_Raisg.shp")
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--min-actual-pct", type=float, default=75.0,
                    help="only chips at/above this actual-burn percentile are eligible")
    ap.add_argument("--map-width", type=int, default=1600,
                    help="decimated width for the panel-A map read")
    ap.add_argument("--out_csv", default="out/figures/fig2024_per_chip.csv")
    ap.add_argument("--out_png", default="out/figures/fig_factored_v1_2024.png")
    ap.add_argument("--from_csv", default=None,
                    help="load the per-chip table from this CSV to skip the slow "
                         "recompute (chip paths are rebuilt from --chip-dir); use "
                         "when iterating on the figure layout only")
    args = ap.parse_args()

    prev_label = os.path.join(args.label_dir, f"label_{args.year - 1}.tif")
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        df["out_path"] = [os.path.join(args.chip_dir, f"out_{c}.tif")
                          for c in df["chip_id"]]
        df["mask_path"] = [os.path.join(args.chip_dir, f"mask_{c}.tif")
                           for c in df["chip_id"]]
        if "interior" not in df.columns:
            add_interior_flag(df)
        print(f"[figure] loaded {len(df)} chips from {args.from_csv}")
    else:
        print("[figure] building per-chip table...", flush=True)
        df = per_chip_table(args.chip_dir, args.climatology, prev_label, args.pos_weight)
        add_interior_flag(df)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        df.drop(columns=["out_path", "mask_path"]).to_csv(args.out_csv, index=False)
        print(f"[figure] wrote {args.out_csv} ({len(df)} chips, "
              f"{int(df['interior'].sum())} interior)")

    cap = display_cap(df)
    picks = select_chips(df, args.min_actual_pct, cap)
    for key in CHIP_ORDER:
        r = picks[key]
        print(f"  {key:6s} {r['chip_id']}  exp={r['exp_factored']:.0f} "
              f"act={r['actual']:.0f} resid={r['exp_factored'] - r['actual']:+.0f}")

    print("[figure] reading basin map...", flush=True)
    map_arr, extent, vmax, gdf = read_basin_map(
        args.map, args.shp, args.map_width, args.pos_weight)
    print(f"[figure] map vmax(98pct)={vmax:.3f}", flush=True)

    build_figure(df, picks, cap, map_arr, extent, vmax, gdf,
                 args.pos_weight, args.out_png)


if __name__ == "__main__":
    main()
