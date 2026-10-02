"""Figure: previous-year October SOI vs basin MCD64 burned area, as time series (A, B) and a scatter (C).

Illustrates the ENSO signal behind gamma (which uses the Oct-Dec mean); extreme years highlighted.
Usage: make_soi_burn_figure.py [--from_csv CSV]"""

import argparse
import os

import numpy as np
import pandas as pd

from style import INK_PRIMARY, INK_SECONDARY, SURFACE, save_figure, style_axes

LINE = "#52514e"
HIGH = "#d55e00"
LOW = "#0072b2"
HIGH_BAND = "#f3d9c6"
LOW_BAND = "#d3e3ef"

SOI_MONTHS = (10,)
SOI_LABEL = "October SOI"
POLY_ORDER = 1
N_EXTREME = 3
PANEL = "out/target_panel/panel.parquet"
PIXEL_KM2 = 0.463312716528 ** 2


def build_table(panel, first, last):
    from aic_risk_modeling.preprocess.climate_indices import download_clim_indices
    d = pd.read_parquet(panel, columns=["year", "burn_bd"])
    burn = d.groupby("year")["burn_bd"].sum().loc[first:last]
    # one spare leading year: NOAA SOI lacks Jan 2000, and an edge gap is not filled
    soi = download_clim_indices("soi", first - 2, last - 1)["metric"]
    sel = soi[soi.index.month.isin(SOI_MONTHS)]
    n = sel.groupby(sel.index.year).size()
    if (n != len(SOI_MONTHS)).any():
        raise ValueError(f"incomplete {SOI_LABEL}: {n[n != len(SOI_MONTHS)].to_dict()}")
    so = sel.groupby(sel.index.year).mean()
    so.index = so.index + 1                         # months of Y-1 -> fire year Y
    t = pd.DataFrame({"burned_km2": burn * PIXEL_KM2, "soi_prev": so}).loc[first:last]
    if t.isna().any().any():
        raise ValueError(f"missing values:\n{t[t.isna().any(axis=1)]}")
    return t.rename_axis("year")


def _style(ax, grid_axis="y"):
    style_axes(ax, grid_axis)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)


def _letter(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=13, fontweight="bold", color=INK_PRIMARY)


def plot(t, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr

    years = t.index.to_numpy()
    burn = t.burned_km2.to_numpy() / 1e3
    soi = t.soi_prev.to_numpy()
    ranked = t.burned_km2.sort_values()
    high, low = set(ranked.index[-N_EXTREME:]), set(ranked.index[:N_EXTREME])
    colors = np.array([HIGH if y in high else LOW if y in low else LINE for y in years])

    fig = plt.figure(figsize=(13.0, 5.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1.0], hspace=0.14, wspace=0.2)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    a3 = fig.add_subplot(gs[:, 1])

    for ax in (a1, a2):
        _style(ax)
        for y in years:
            if y in high or y in low:
                ax.axvspan(y - 0.42, y + 0.42, color=HIGH_BAND if y in high else LOW_BAND,
                           linewidth=0, zorder=1)

    a1.plot(years, burn, color=LINE, linewidth=2.0, zorder=3)
    a1.scatter(years, burn, s=34, color=colors, edgecolor=SURFACE, linewidth=0.8, zorder=4)
    a1.set_ylim(0, burn.max() * 1.08)
    a1.set_ylabel("Burned Area\n(Thousand km²)", fontsize=10.5, color=INK_SECONDARY)
    a1.tick_params(labelbottom=False)

    a2.axhline(0, color=INK_SECONDARY, linewidth=0.8, zorder=2)
    a2.plot(years, soi, color=LINE, linewidth=1.6, zorder=3)
    a2.scatter(years, soi, s=34, color=colors, edgecolor=SURFACE, linewidth=0.8, zorder=4)
    a2.set_ylabel(f"{SOI_LABEL},\nPrevious Year", fontsize=10.5, color=INK_SECONDARY)
    a2.set_xlabel("Fire Year", fontsize=10.5, color=INK_SECONDARY)
    a2.set_xticks(years[::2])
    a2.set_xlim(years[0] - 0.6, years[-1] + 0.6)

    _style(a3, grid_axis="both")
    a3.axvline(0, color=INK_SECONDARY, linewidth=0.8, zorder=2)
    xs = np.linspace(soi.min(), soi.max(), 200)
    trend = np.polyval(np.polyfit(soi, burn, POLY_ORDER), xs)
    if not (np.all(np.diff(trend) <= 0) or np.all(np.diff(trend) >= 0)):
        print(f"[soi_burn] WARNING: order-{POLY_ORDER} trend is not monotonic over the data")
    a3.plot(xs, trend, color=INK_SECONDARY, linewidth=1.4, linestyle="--", zorder=3)
    rho, pval = spearmanr(soi, burn)
    ptxt = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
    a3.text(0.03, 0.04, f"Spearman ρ = {rho:.2f} ({ptxt})".replace("-", "−"),
            transform=a3.transAxes, ha="left", va="bottom", fontsize=9.5,
            color=INK_SECONDARY, zorder=6,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0))
    order = np.argsort([c != LINE for c in colors], kind="stable")
    a3.scatter(soi[order], burn[order], s=46, color=colors[order],
               edgecolor=SURFACE, linewidth=0.8, zorder=4)
    # Label each extreme on the first side whose text box holds no other point.
    w, h = 0.14 * np.ptp(soi), 0.035 * burn.max()
    sides = [((6, 0), "left", "center", (0.01 * w, w), (-h / 2, h / 2)),
             ((-6, 0), "right", "center", (-w, -0.01 * w), (-h / 2, h / 2)),
             ((0, -7), "center", "top", (-w / 2, w / 2), (-1.6 * h, -0.3 * h)),
             ((0, 7), "center", "bottom", (-w / 2, w / 2), (0.3 * h, 1.6 * h))]
    for y, x, v in zip(years, soi, burn):
        if y not in high and y not in low:
            continue
        others = (years != y)
        for off, ha, va, (x0, x1), (y0, y1) in sides:
            inside = ((soi > x + x0) & (soi < x + x1) & (burn > v + y0) & (burn < v + y1))
            if not np.any(inside & others):
                break
        a3.annotate(str(y), (x, v), xytext=off, textcoords="offset points", ha=ha, va=va,
                    fontsize=8.5, color=HIGH if y in high else LOW, zorder=5)
    a3.set_ylim(0, burn.max() * 1.08)
    a3.set_xlabel(f"{SOI_LABEL}, Previous Year", fontsize=10.5, color=INK_SECONDARY)
    a3.set_ylabel("Burned Area (Thousand km²)", fontsize=10.5, color=INK_SECONDARY)
    a3.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=HIGH, markersize=7,
               label=f"Top {N_EXTREME} Fire Years"),
        Line2D([], [], marker="o", linestyle="", color=LOW, markersize=7,
               label=f"Bottom {N_EXTREME} Fire Years")],
        loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    for ax, s in ((a1, "A"), (a2, "B"), (a3, "C")):
        _letter(ax, s)
    fig.align_ylabels([a1, a2])
    save_figure(fig, out_png, "soi_burn", tight=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--years", default="2001-2025")
    ap.add_argument("--from_csv", default=None)
    ap.add_argument("--out_png", default="out/figures/fig_soi_burn.png")
    a = ap.parse_args()
    if a.from_csv:
        t = pd.read_csv(a.from_csv, index_col="year")
    else:
        first, _, last = a.years.partition("-")
        t = build_table(a.panel, int(first), int(last or first))
        t.to_csv(os.path.splitext(a.out_png)[0] + ".csv")
    from scipy.stats import spearmanr
    rho, pval = spearmanr(t.soi_prev, t.burned_km2)
    print(t.round(2).to_string())
    print(f"[soi_burn] Spearman rho({SOI_LABEL} Y-1, burned area Y) = {rho:.3f}, "
          f"p = {pval:.2g}, n = {len(t)}")
    plot(t, a.out_png)


if __name__ == "__main__":
    main()
