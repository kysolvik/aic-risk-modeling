"""Fig 4: one year's calibrated fire risk as 3 panels (map, per-chip scatter, example chips) or a 2x2.

Defaults: yeargain final_all on 2024 with the frozen Platt calibrator; everything is on one
scale, expected burned pixels per chip.
Usage: make_risk_figure_2024.py [--layout 3panel|2x2] [--from_csv CSV]"""

import argparse
import os

import numpy as np
import pandas as pd

from aic_risk_modeling.eval.calibration import load_calibrator, to_prob
from aic_risk_modeling.eval.chips import basin_mask, chip_pairs, read_window
from aic_risk_modeling.eval.metrics import fit_stats
from style import (CALIBRATOR, INK_PRIMARY, INK_SECONDARY, LABEL_DIR, RISK_CMAP, SHP,
                   SURFACE, save_figure, style_axes)

POINT = "#2a78d6"
ONE_TO_ONE = "#8a8880"

CHIP_COLORS = {"under": "#0072b2", "right": "#009e73", "over": "#d55e00"}
CHIP_LABELS = {"under": "Under-predicted", "right": "Well-predicted",
               "over": "Over-predicted"}
CHIP_ORDER = ["under", "right", "over"]
BURN_CMAP = ["#f3ede2", "#7f0000"]
ERROR_CMAP = ["#0072b2", "#f7f7f7", "#d55e00"]
ERROR_NEUTRAL = "#dedcd6"


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
    v = read_window(src, bounds, chip_shape).astype(np.float64)
    return float((v > 0).sum() if binarize else v.sum())


def add_interior_flag(df):
    """Flag chips whose 8 grid neighbours all exist (not on the basin edge)."""
    step = float(np.median(df["right"] - df["left"]))
    ix = np.rint(df["lon"] / step).astype(int)
    iy = np.rint(df["lat"] / step).astype(int)
    present = set(zip(ix, iy))
    df["interior"] = [all((i + di, j + dj) in present
                          for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0))
                      for i, j in zip(ix, iy)]
    return df


def display_cap(df):
    """Scatter axis top: 99.5th percentile of expected+actual."""
    vals = np.concatenate([df["exp_factored"].values, df["exp_lastyear"].values,
                           df["exp_clim"].values, df["actual"].values])
    return float(np.percentile(vals, 99.5))


def select_chips(df, min_actual_pct, cap):
    """Pick under- / well- / over-predicted example chips: real fire, interior, on-scale."""
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


def read_basin_map(map_path, shp_path, target_width, pos_weight, level=1.0, cal=None):
    """Decimated calibrated risk map masked to the basin -> (arr, extent, vmax)."""
    import rasterio as rio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds as tr_from_bounds

    with rio.open(map_path) as s:
        scale = target_width / s.width
        h, w = int(round(s.height * scale)), target_width
        q = s.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float64)
        b = s.bounds
        crs = s.crs
    arr = to_prob(np.clip(q, 0.0, 1.0), pos_weight, level, cal)

    tr = tr_from_bounds(b.left, b.bottom, b.right, b.top, w, h)
    inside, gdf = basin_mask(shp_path, crs, tr, (h, w))
    arr = np.where(inside, arr, np.nan)

    vmax = float(np.nanpercentile(arr, 98))
    extent = [b.left, b.right, b.bottom, b.top]
    return arr, extent, vmax, gdf


def read_crop(out_path, mask_path, pos_weight, level=1.0, cal=None):
    """Full-res calibrated risk crop + burn mask for one chip -> (prob, burn, extent)."""
    import rasterio as rio
    with rio.open(out_path) as s:
        q = np.clip(s.read(1).astype(np.float64), 0.0, 1.0)
        b = s.bounds
    with rio.open(mask_path) as m:
        burn = (m.read(1) > 0).astype(float)
    prob = to_prob(q, pos_weight, level, cal)
    return prob, burn, [b.left, b.right, b.bottom, b.top]


def draw_error_map(ax, df, gdf, cmap, norm, picks):
    """Per-tile signed-error choropleth (positive = over-predicted; inactive tiles neutral)."""
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
        # U/R/O markers keep the chips locatable where the box blends into the map.
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

    BX0, BX1 = 0.050, 0.968
    BY0, BY1 = 0.035, 0.965
    XM = 0.508
    YM = 0.500
    LW = 1.7

    err_cmap = LinearSegmentedColormap.from_list("err", ERROR_CMAP)
    active = df["err_active"].to_numpy().astype(bool)
    ev = df["err_value"].to_numpy()
    vlim = float(np.nanpercentile(np.abs(ev[active]), 95)) if active.any() else 1.0
    vlim = max(vlim, 1.0 if error_mode == "raw" else 0.1)
    err_norm = Normalize(vmin=-vlim, vmax=vlim)

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
    ax_map.set_anchor("S")
    minx, miny, maxx, maxy = gdf.total_bounds
    ax_map.set_xlim(minx, maxx)
    ax_map.set_ylim(miny, maxy)
    cax_a = fig.add_axes(cbar_rect(BX0, XM))
    cb = fig.colorbar(im, cax=cax_a, orientation="horizontal", extend="max")
    cb.set_label("Predicted burned area (%)", fontsize=10,
                 color=INK_SECONDARY)
    cb.ax.tick_params(labelsize=9, colors=INK_SECONDARY)
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
    style_axes(ax_c)
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
    d_left = XM + 0.126
    d_right = BX1 - 0.016
    d_top = YM - 0.040
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
        ry = d_top - (drow + 0.5) * row_slot
        fig.text(d_left - 0.012, ry,
                 f"{CHIP_LABELS[key]}\nExpected {r['exp_factored']:.0f}\n"
                 f"Actual {r['actual']:.0f}",
                 fontsize=10.5, color=c, fontweight="bold", ha="right", va="center")

    fig.add_artist(Rectangle((BX0, BY0), BX1 - BX0, BY1 - BY0,
                             transform=fig.transFigure, fill=False,
                             edgecolor="black", linewidth=LW, zorder=20))
    for xy in ([[BX0, BX1], [YM, YM]],
               [[XM, XM], [BY0, BY1]]):
        ln = Line2D(xy[0], xy[1], color="black", linewidth=LW, zorder=20)
        ln.set_transform(fig.transFigure)
        fig.add_artist(ln)
    for letter, (lx, ly) in [("A", (BX0, BY1)), ("B", (XM, BY1)),
                             ("C", (BX0, YM)), ("D", (XM, YM))]:
        fig.text(lx + 0.007, ly - 0.009, letter, fontsize=17, fontweight="bold",
                 color=INK_PRIMARY, ha="left", va="top", zorder=21,
                 bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                           edgecolor="none"))

    save_figure(fig, out_png, "figure")
    plt.close(fig)


def build_figure_3panel(df, picks, cap, map_arr, map_extent, vmax, gdf, pos_weight,
                        out_png, level=1.0, cal=None):
    """Three-panel layout: A map | B scatter on top, C example chips below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    fig = plt.figure(figsize=(13.6, 15.0), facecolor=SURFACE)

    BX0, BX1 = 0.050, 0.968
    BY0, BY1 = 0.035, 0.965
    XB = 0.560
    YB = 0.595
    LW = 1.7

    # ---- Panel A: basin map + vertical colorbar (top-left) -----------------
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
    cb.formatter = FuncFormatter(lambda v, _: f"{v * 100:.0f}")
    cb.update_ticks()

    # ---- Panel B: single model scatter (top-right) -------------------------
    hi = cap * 1.05
    ax_b = fig.add_axes([XB + 0.072, YB + 0.052, BX1 - (XB + 0.072) - 0.012,
                         BY1 - (YB + 0.052) - 0.016])
    style_axes(ax_b)
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

    fig.add_artist(Rectangle((BX0, BY0), BX1 - BX0, BY1 - BY0,
                             transform=fig.transFigure, fill=False,
                             edgecolor="black", linewidth=LW, zorder=20))
    for xy in ([[BX0, BX1], [YB, YB]],
               [[XB, XB], [YB, BY1]]):
        ln = Line2D(xy[0], xy[1], color="black", linewidth=LW, zorder=20)
        ln.set_transform(fig.transFigure)
        fig.add_artist(ln)
    for letter, (lx, ly) in [("A", (BX0, BY1)), ("B", (XB, BY1)), ("C", (BX0, YB))]:
        fig.text(lx + 0.007, ly - 0.009, letter, fontsize=17, fontweight="bold",
                 color=INK_PRIMARY, ha="left", va="top", zorder=21,
                 bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                           edgecolor="none"))

    save_figure(fig, out_png, "figure")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", default="out/cv/preds/"
                    "factored_v3p_union4_monthlyattn_wide_yeargain/final_all",
                    help="fold dir with <year>/ chips and map")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--chip-dir", default=None, help="default <pred-root>/<year>/chips")
    ap.add_argument("--map", default=None, help="default <pred-root>/<year>/preds_out.tif")
    ap.add_argument("--climatology",
                    default=f"{LABEL_DIR}/climatology_2013_2023.tif")
    ap.add_argument("--label-dir", default=LABEL_DIR)
    ap.add_argument("--level-factor", type=float, default=None,
                    help="with --calibrator '': deflated expected / actual level")
    ap.add_argument("--calibrator", default=CALIBRATOR,
                    metavar="NPZ",
                    help="frozen Platt calibrator; '' = --level-factor")
    ap.add_argument("--shp", default=SHP)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--min-actual-pct", type=float, default=75.0,
                    help="min actual-burn percentile for example chips")
    ap.add_argument("--map-vmax", type=float, default=None,
                    help="risk colour max; default 98th percentile")
    ap.add_argument("--map-width", type=int, default=1600,
                    help="decimated width of the panel-A map")
    ap.add_argument("--out_csv", default=None,
                    help="default out/figures/fig_tiles_<year>_per_chip.csv")
    ap.add_argument("--out_png", default=None, help="default out/figures/fig_tiles_<year>.png")
    ap.add_argument("--from_csv", default=None,
                    help="reuse a saved per-chip table (layout iteration)")
    ap.add_argument("--layout", choices=["3panel", "2x2"], default="3panel")
    ap.add_argument("--error-mode", choices=["raw", "normalized"], default="normalized",
                    help="error: expected - actual, or / (expected + actual)")
    ap.add_argument("--resid-activity-min", type=float, default=75.0,
                    help="panel B: expected+actual below this is drawn neutral")
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
    if args.map_vmax is not None:
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
