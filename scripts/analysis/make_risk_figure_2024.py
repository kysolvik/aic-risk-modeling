"""Publication figure (Fig 4): one year's fire risk, three panels (default) or a 2x2 frame.

`--layout 3panel` (default): A risk map | B per-chip scatter on top, C the three
example chips (columns) x [Predicted risk, Actual burn] (rows) below.
`--layout 2x2`: the four panels described below (adds the per-tile error map).

v3p version: defaults to the yeargain final_all model (trained 2013-2023) on the
2024 write-once test year, with the frozen Platt calibrator fit on the CV
fold-years 2018-23 (the one Fig 5 uses) applied to the raw output, and the
2013-2023 (final_all train years) burn-frequency climatology. `--calibrator ''`
falls back to the old deflate / --level-factor path (deflated expected / actual on
years OTHER than --year; e.g. 0.717 = fwdpair_2022 pooled over 2022-23), which then
needs an explicit --level-factor. Maps are drawn in the grid's native MODIS
sinusoidal CRS, without lon/lat axes.


  A (top-left)     Basin map of 2024 fire risk (factored_v1 output), RAISG outline,
                   with the three example chips boxed.
  B (top-right)    Per-tile normalized prediction error across the basin: signed
                   (expected - actual) / (expected + actual) on a diverging scale
                   (over-predicted warm, under-predicted cool), same three example
                   chips boxed. Tiles below the activity floor are drawn neutral.
  C (bottom-left)  Per-chip expected-vs-actual burned pixels for factored_v1 (the
                   cloud behind the year-total headline), the three example chips
                   highlighted.
  D (bottom-right) Zoomed crops for the three example chips (rows: under- / well- /
                   over-predicted), each row showing predicted risk and the 2024
                   actual-burn footprint side by side.

Everything is computed on ONE calibrated scale -- deflated expected burned pixels
per chip -- reusing the per-predictor definitions in `scatter_expected_actual.py`:
models are deflated for the weighted-BCE pos_weight (10).

    .venv/bin/python scripts/analysis/make_risk_figure_2024.py

Original v2 inputs (factored_v1 2024, one 7296x6272 EPSG:4326 grid):
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
    load_calibrator,
    fit_stats,
)

# Okabe-Ito colourblind-safe trio, reused for each chip across A, B and C.
CHIP_COLORS = {"under": "#0072b2", "right": "#009e73", "over": "#d55e00"}
CHIP_LABELS = {"under": "Under-predicted", "right": "Well-predicted",
               "over": "Over-predicted"}
CHIP_ORDER = ["under", "right", "over"]  # top-to-bottom in panels B and C
RISK_CMAP = "YlOrRd"
BURN_CMAP = ["#f3ede2", "#7f0000"]  # unburned cream, burned dark red (YlOrRd top)
# Diverging per-tile error map (panel B): under-predicted = blue, on-target = white,
# over-predicted = vermillion -- same over/under semantics as CHIP_COLORS.
ERROR_CMAP = ["#0072b2", "#f7f7f7", "#d55e00"]  # under (blue) -> white -> over (vermillion)
ERROR_NEUTRAL = "#dedcd6"  # tiles below the activity floor (no reliable error signal)


# --------------------------------------------------------------------------- #
# Per-chip table (single pass over the 1,813 tiles)
# --------------------------------------------------------------------------- #
def to_prob(q, pos_weight, level=1.0, cal=None):
    """Model score -> burn probability: the frozen calibrator if given, else deflate / level."""
    return cal(q) if cal is not None else deflate(q, pos_weight) / level


def per_chip_table(chip_dir, clim_path, prev_label_path, pos_weight, level=1.0, cal=None):
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
            exp_factored = float(to_prob(q, pos_weight, level, cal).sum())
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


def add_interior_flag(df):
    """Flag chips whose 8 grid neighbours all exist (i.e. not on the ragged basin edge).

    Chips tile a regular grid with step = chip width, in whatever CRS the bounds are
    in (0.64 deg on the v2 grid, ~59.3 km on the v3 sinusoidal grid).
    """
    step = float(np.median(df["right"] - df["left"]))
    ix = np.rint(df["lon"] / step).astype(int)
    iy = np.rint(df["lat"] / step).astype(int)
    present = set(zip(ix, iy))
    df["interior"] = [all((i + di, j + dj) in present
                          for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0))
                      for i, j in zip(ix, iy)]
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
def read_basin_map(map_path, shp_path, target_width, pos_weight, level=1.0, cal=None):
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
        crs = s.crs
    arr = to_prob(np.clip(q, 0.0, 1.0), pos_weight, level, cal)

    gdf = gpd.read_file(shp_path).to_crs(crs)  # draw in the raster's own CRS
    tr = tr_from_bounds(b.left, b.bottom, b.right, b.top, w, h)
    basin = rasterize(((g, 1) for g in gdf.geometry), out_shape=(h, w),
                      transform=tr, fill=0, dtype="uint8")
    arr = np.where(basin > 0, arr, np.nan)

    vmax = float(np.nanpercentile(arr, 98))
    extent = [b.left, b.right, b.bottom, b.top]
    return arr, extent, vmax, gdf


def read_crop(out_path, mask_path, pos_weight, level=1.0, cal=None):
    """Full-res deflated risk crop + actual-burn mask for one chip. Returns (prob, burn, extent)."""
    import rasterio as rio
    with rio.open(out_path) as s:
        q = np.clip(s.read(1).astype(np.float64), 0.0, 1.0)
        b = s.bounds
    with rio.open(mask_path) as m:
        burn = (m.read(1) > 0).astype(float)
    prob = to_prob(q, pos_weight, level, cal)
    return prob, burn, [b.left, b.right, b.bottom, b.top]


def draw_error_map(ax, df, gdf, cmap, norm, picks):
    """Per-tile normalized-error choropleth on the RAISG basin outline.

    Each 0.64-degree tile is a filled rectangle coloured by its signed error
    ``df['err_value']`` (positive = over-predicted). Tiles below the activity
    floor (``df['err_active']`` False) are drawn neutral -- their error is not
    meaningful when almost nothing burned or was predicted. Returns a ScalarMappable
    for the colorbar.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    ax.set_facecolor("#f2f1ee")
    for _, r in df.iterrows():
        if bool(r["err_active"]) and np.isfinite(r["err_value"]):
            fc = cmap(norm(float(r["err_value"])))
        else:
            fc = ERROR_NEUTRAL
        ax.add_patch(Rectangle(
            (r["left"], r["bottom"]), r["right"] - r["left"], r["top"] - r["bottom"],
            facecolor=fc, edgecolor="none", linewidth=0, zorder=2))
    gdf.boundary.plot(ax=ax, color=INK_SECONDARY, linewidth=0.7, zorder=4)
    for key in CHIP_ORDER:
        r = picks[key]
        c = CHIP_COLORS[key]
        ax.add_patch(Rectangle(
            (r["left"], r["bottom"]), r["right"] - r["left"], r["top"] - r["bottom"],
            fill=False, edgecolor=c, linewidth=2.2, zorder=5))
        # Same U/R/O circle markers as panel A, so the example chips stay locatable
        # even where the box edge blends into a like-coloured error patch.
        ax.annotate(
            key[0].upper(), xy=(r["right"], r["top"]), xytext=(3, 3),
            textcoords="offset points", color="white", fontsize=11, fontweight="bold",
            ha="left", va="bottom", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.15", facecolor=c, edgecolor="white", lw=0.8))
    ax.set_aspect("equal")
    ax.set_anchor("S")
    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(df, picks, cap, map_arr, map_extent, vmax, gdf, pos_weight, out_png,
                 error_mode="raw", level=1.0, cal=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    fig = plt.figure(figsize=(13.6, 12.6), facecolor=SURFACE)

    # One outer frame split into a clean 2x2 by a full-width horizontal divider (YM)
    # and a full-height vertical divider (XM): A risk map | B error map on top,
    # C scatter | D chip grid below. All axes are placed by hand to line up to it.
    BX0, BX1 = 0.050, 0.968        # frame left / right
    BY0, BY1 = 0.035, 0.965        # frame bottom / top
    XM = 0.508                     # vertical divider (columns): A|B and C|D
    YM = 0.500                     # horizontal divider (rows): A,B above | C,D below
    LW = 1.7

    # Diverging error colormap + symmetric norm from the active tiles only.
    err_cmap = LinearSegmentedColormap.from_list("err", ERROR_CMAP)
    active = df["err_active"].to_numpy().astype(bool)
    ev = df["err_value"].to_numpy()
    vlim = float(np.nanpercentile(np.abs(ev[active]), 95)) if active.any() else 1.0
    vlim = max(vlim, 1.0 if error_mode == "raw" else 0.1)
    err_norm = Normalize(vmin=-vlim, vmax=vlim)

    # Map cells: the map spans the cell width (no lon/lat axes on the sinusoidal
    # grid) with a horizontal colorbar underneath.
    def map_rect(x0, x1):
        return [x0 + 0.018, YM + 0.092, (x1 - x0) - 0.036, (BY1 - YM) - 0.092 - 0.034]

    def cbar_rect(x0, x1):
        return [x0 + 0.08, YM + 0.058, (x1 - x0) - 0.16, 0.011]

    # ---- Panel A: basin risk map + colorbar (top-left) ---------------------
    ax_map = fig.add_axes(map_rect(BX0, XM))
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
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_aspect("equal")
    ax_map.set_anchor("S")  # sit the aspect-shrunk map just above its colorbar
    minx, miny, maxx, maxy = gdf.total_bounds
    ax_map.set_xlim(minx, maxx)
    ax_map.set_ylim(miny, maxy)
    cax_a = fig.add_axes(cbar_rect(BX0, XM))
    cb = fig.colorbar(im, cax=cax_a, orientation="horizontal", extend="max")
    cb.set_label("Predicted burned area (%)", fontsize=10,
                 color=INK_SECONDARY)
    cb.ax.tick_params(labelsize=9, colors=INK_SECONDARY)
    # data stay probabilities; ticks in percent (0.5 -> "50"), matching the forecast maps
    cb.formatter = FuncFormatter(lambda v, _: f"{v * 100:.0f}")
    cb.update_ticks()

    # ---- Panel B: per-tile normalized error map (top-right) ----------------
    ax_err = fig.add_axes(map_rect(XM, BX1))
    err_sm = draw_error_map(ax_err, df, gdf, err_cmap, err_norm, picks)
    ax_err.set_xticks([])
    ax_err.set_yticks([])
    cax_b = fig.add_axes(cbar_rect(XM, BX1))
    cb2 = fig.colorbar(err_sm, cax=cax_b, orientation="horizontal")
    err_label = ("Over/under-prediction (pixels)  (over +, under −)"
                 if error_mode == "raw"
                 else "Normalized error  (over +, under −)")
    cb2.set_label(err_label, fontsize=10, color=INK_SECONDARY)
    cb2.ax.tick_params(labelsize=9, colors=INK_SECONDARY)

    # ---- Panel C: single-model scatter (bottom-left) -----------------------
    hi = cap * 1.05
    ax_c = fig.add_axes([BX0 + 0.064, BY0 + 0.058,
                         (XM - BX0) - 0.064 - 0.030,
                         (YM - BY0) - 0.058 - 0.034])
    ax_c.set_facecolor(SURFACE)
    ax_c.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax_c.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_c.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax_c.spines[sp].set_color(GRID)
    x, y = df["exp_factored"].values, df["actual"].values
    ax_c.plot([0, hi], [0, hi], "--", color=ONE_TO_ONE, linewidth=1.3, zorder=2)
    ax_c.scatter(x, y, s=9, color=POINT, alpha=0.22, linewidths=0, zorder=3)
    s = fit_stats(x, y)
    for key in CHIP_ORDER:
        r = picks[key]
        ax_c.scatter([float(r["exp_factored"])], [float(r["actual"])], s=185,
                     facecolor=CHIP_COLORS[key], edgecolor="black", linewidth=1.5,
                     zorder=6)
    ax_c.text(0.96, 0.06, f"R² = {s['r2']:.3f}", transform=ax_c.transAxes,
              fontsize=12.5, color=INK_SECONDARY, va="bottom", ha="right")
    ax_c.set_xlim(0, hi)
    ax_c.set_ylim(0, hi)
    ax_c.set_aspect("equal")
    ax_c.set_anchor("N")
    ax_c.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    ax_c.set_xlabel("Expected burned pixels", fontsize=11.5, color=INK_SECONDARY)
    ax_c.set_ylabel("Actual burned pixels", fontsize=11.5, color=INK_SECONDARY)

    # ---- Panel D: chips as rows (under/well/over) x [Predicted, Actual] -----
    burn_cmap = ListedColormap(BURN_CMAP)
    d_left = XM + 0.126         # room for the per-chip row labels
    d_right = BX1 - 0.016
    d_top = YM - 0.040          # room for the letter + column headers
    d_bottom = BY0 + 0.016
    col_slot = (d_right - d_left) / 2.0
    row_slot = (d_top - d_bottom) / 3.0
    gx, gy = 0.008, 0.012  # inner gap so squares don't touch
    col_names = ["Predicted risk", "Actual burn"]
    for drow, key in enumerate(CHIP_ORDER):
        r = picks[key]
        c = CHIP_COLORS[key]
        prob, burn, ext = read_crop(r["out_path"], r["mask_path"], pos_weight, level, cal)
        for dcol, (arr, cmap, vlim) in enumerate(
                [(prob, RISK_CMAP, vmax), (burn, burn_cmap, 1)]):
            rect = [d_left + dcol * col_slot + gx,
                    d_top - (drow + 1) * row_slot + gy,
                    col_slot - 2 * gx, row_slot - 2 * gy]
            a = fig.add_axes(rect)
            a.imshow(arr, extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=vlim,
                     interpolation="nearest")
            for sp in a.spines.values():
                sp.set_color(c)
                sp.set_linewidth(2.6)
            a.set_xticks([])
            a.set_yticks([])
            if drow == 0:
                a.set_title(col_names[dcol], fontsize=11.5, color=INK_SECONDARY, pad=6)
        # Per-chip row label to the left, vertically centred on the row.
        ry = d_top - (drow + 0.5) * row_slot
        fig.text(d_left - 0.012, ry,
                 f"{CHIP_LABELS[key]}\nExpected {r['exp_factored']:.0f}\n"
                 f"Actual {r['actual']:.0f}",
                 fontsize=10.5, color=c, fontweight="bold", ha="right", va="center")

    # ---- outer frame + the cross divider -----------------------------------
    fig.add_artist(Rectangle((BX0, BY0), BX1 - BX0, BY1 - BY0,
                             transform=fig.transFigure, fill=False,
                             edgecolor="black", linewidth=LW, zorder=20))
    for xy in ([[BX0, BX1], [YM, YM]],       # horizontal divider (full width)
               [[XM, XM], [BY0, BY1]]):       # vertical divider (full height)
        ln = Line2D(xy[0], xy[1], color="black", linewidth=LW, zorder=20)
        ln.set_transform(fig.transFigure)
        fig.add_artist(ln)
    for letter, (lx, ly) in [("A", (BX0, BY1)), ("B", (XM, BY1)),
                             ("C", (BX0, YM)), ("D", (XM, YM))]:
        fig.text(lx + 0.007, ly - 0.009, letter, fontsize=17, fontweight="bold",
                 color=INK_PRIMARY, ha="left", va="top", zorder=21,
                 bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                           edgecolor="none"))

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor=SURFACE)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", facecolor=SURFACE)
    plt.close(fig)
    print(f"[figure] wrote {out_png} (+ .pdf)")


def build_figure_3panel(df, picks, cap, map_arr, map_extent, vmax, gdf, pos_weight,
                        out_png, level=1.0, cal=None):
    """Three-panel layout (the original 9/10 design): A risk map | B scatter on top,
    C = the three example chips as columns x [Predicted, Actual] rows below. No error map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    fig = plt.figure(figsize=(13.6, 15.0), facecolor=SURFACE)

    # One outer frame divided into three connected panels: A (map) and B (scatter)
    # share the top row, split by a vertical divider at XB; C (chips) is the whole
    # bottom, split from the top by a horizontal divider at YB.
    BX0, BX1 = 0.050, 0.968        # frame left / right
    BY0, BY1 = 0.035, 0.965        # frame bottom / top
    XB = 0.560                     # vertical divider (A | B), top row only
    YB = 0.595                     # horizontal divider (A,B above | C below)
    LW = 1.7

    # ---- Panel A: basin map + vertical colorbar (top-left) -----------------
    # No lon/lat axes on the sinusoidal grid, so the map fills the cell up to the colorbar.
    ax_map = fig.add_axes([BX0 + 0.022, YB + 0.022, (XB - 0.105) - (BX0 + 0.022),
                           BY1 - (YB + 0.022) - 0.030])
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
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_aspect("equal")
    ax_map.set_anchor("C")
    minx, miny, maxx, maxy = gdf.total_bounds
    ax_map.set_xlim(minx, maxx)
    ax_map.set_ylim(miny, maxy)
    cax = fig.add_axes([XB - 0.082, YB + 0.070, 0.015, 0.28])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("Predicted burned area (%)", fontsize=10, color=INK_SECONDARY)
    cb.ax.tick_params(labelsize=9, colors=INK_SECONDARY)
    # data stay probabilities; ticks in percent (0.5 -> "50"), matching the forecast maps
    cb.formatter = FuncFormatter(lambda v, _: f"{v * 100:.0f}")
    cb.update_ticks()

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
        prob, burn, ext = read_crop(r["out_path"], r["mask_path"], pos_weight, level, cal)
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
    ap.add_argument("--pred-root", default="out/cv/preds/"
                    "factored_v3p_union4_monthlyattn_wide_yeargain/final_all",
                    help="fold dir; chips + map default to <pred-root>/<year>/...")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--chip-dir", default=None, help="default <pred-root>/<year>/chips")
    ap.add_argument("--map", default=None, help="default <pred-root>/<year>/preds_out.tif")
    ap.add_argument("--climatology",
                    default="out/label_mosaics_v3p_union4/climatology_2013_2023.tif")
    ap.add_argument("--label-dir", default="out/label_mosaics_v3p_union4")
    ap.add_argument("--level-factor", type=float, default=None,
                    help="with --calibrator '': divide the deflated output by this "
                         "(deflated expected / actual on years other than --year; "
                         "0.717 = fwdpair_2022 on 2022-23, 0.774 = on 2022 alone)")
    ap.add_argument("--calibrator", default="out/cv/calibrator_platt_cv2018_2023.npz",
                    metavar="NPZ",
                    help="frozen Platt calibrator (calibrated_year_totals.py "
                         "--save-calibrator); replaces deflate + --level-factor. "
                         "Pass '' to use --level-factor instead")
    ap.add_argument("--shp", default="../data/Limites_RAISG_2025/Lim_Raisg.shp")
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--min-actual-pct", type=float, default=75.0,
                    help="only chips at/above this actual-burn percentile are eligible")
    ap.add_argument("--map-vmax", type=float, default=None,
                    help="colour-scale max (probability) for the basin map and chip "
                         "risk panels; default = this year's 98th percentile. Pass the "
                         "same value for every year in a set (manuscript: 0.5)")
    ap.add_argument("--map-width", type=int, default=1600,
                    help="decimated width for the panel-A map read")
    ap.add_argument("--out_csv", default=None,
                    help="default out/figures/fig_tiles_<year>_per_chip.csv")
    ap.add_argument("--out_png", default=None, help="default out/figures/fig_tiles_<year>.png")
    ap.add_argument("--from_csv", default=None,
                    help="load the per-chip table from this CSV to skip the slow "
                         "recompute (chip paths are rebuilt from --chip-dir); use "
                         "when iterating on the figure layout only")
    ap.add_argument("--layout", choices=["3panel", "2x2"], default="3panel",
                    help="3panel = A map | B scatter over C chip columns (no error "
                         "map); 2x2 = A map | B error map over C scatter | D chips")
    ap.add_argument("--error-mode", choices=["raw", "normalized"], default="normalized",
                    help="panel B tile colour: 'raw' = signed over/under burned "
                         "pixels (expected - actual); 'normalized' = that residual "
                         "divided by (expected + actual), bounded [-1, 1]")
    ap.add_argument("--resid-activity-min", type=float, default=75.0,
                    help="panel B: tiles whose expected+actual burned pixels fall "
                         "below this carry no reliable error signal and are drawn "
                         "neutral instead of coloured. Normalized error is unstable "
                         "at low counts, so this floor matters most in that mode")
    args = ap.parse_args()
    args.chip_dir = args.chip_dir or os.path.join(args.pred_root, str(args.year), "chips")
    args.map = args.map or os.path.join(args.pred_root, str(args.year), "preds_out.tif")
    args.out_csv = args.out_csv or f"out/figures/fig_tiles_{args.year}_per_chip.csv"
    args.out_png = args.out_png or f"out/figures/fig_tiles_{args.year}.png"
    if not args.calibrator and args.level_factor is None:
        ap.error("--calibrator '' needs an explicit --level-factor")
    cal = load_calibrator(args.calibrator) if args.calibrator else None
    print(f"[figure] {args.chip_dir}, "
          + (f"calibrator {args.calibrator}" if cal else f"level factor {args.level_factor}"))

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
        df = per_chip_table(args.chip_dir, args.climatology, prev_label, args.pos_weight,
                            args.level_factor, cal)
        add_interior_flag(df)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        df.drop(columns=["out_path", "mask_path"]).to_csv(args.out_csv, index=False)
        print(f"[figure] wrote {args.out_csv} ({len(df)} chips, "
              f"{int(df['interior'].sum())} interior)")

    # Per-tile signed error for panel B, positive = over-predicted. Two modes:
    #   raw         expected - actual, in burned pixels (unbounded)
    #   normalized  (expected - actual) / (expected + actual), bounded [-1, 1]
    # Tiles below the activity floor carry no reliable signal and are flagged for
    # neutral fill.
    resid = (df["exp_factored"] - df["actual"]).to_numpy()
    denom = (df["exp_factored"] + df["actual"]).to_numpy()
    if args.error_mode == "raw":
        df["err_value"] = resid
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            df["err_value"] = resid / denom
    df["err_active"] = denom >= args.resid_activity_min

    cap = display_cap(df)
    picks = select_chips(df, args.min_actual_pct, cap)
    for key in CHIP_ORDER:
        r = picks[key]
        print(f"  {key:6s} {r['chip_id']}  exp={r['exp_factored']:.0f} "
              f"act={r['actual']:.0f} resid={r['exp_factored'] - r['actual']:+.0f}")

    print("[figure] reading basin map...", flush=True)
    map_arr, extent, vmax, gdf = read_basin_map(
        args.map, args.shp, args.map_width, args.pos_weight, args.level_factor, cal)
    if args.map_vmax is not None:       # fixed scale so years compare (default: 98th pct)
        vmax = args.map_vmax
    print(f"[figure] map + chip colour max={vmax:.3f}", flush=True)

    if args.layout == "3panel":
        build_figure_3panel(df, picks, cap, map_arr, extent, vmax, gdf,
                            args.pos_weight, args.out_png,
                            level=args.level_factor, cal=cal)
    else:
        build_figure(df, picks, cap, map_arr, extent, vmax, gdf,
                     args.pos_weight, args.out_png, error_mode=args.error_mode,
                     level=args.level_factor, cal=cal)


if __name__ == "__main__":
    main()
