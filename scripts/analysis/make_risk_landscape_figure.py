"""Conceptual figure: the fire-risk-model landscape (resolution x time horizon).

A single data-free schematic that situates three classes of fire-risk model on a
log-log plane of spatial resolution (y) against forecast / assessment horizon (x):

  - Short-term operational   -- weather-driven & real-time burn detection (<1 week)
  - Medium-term strategic     -- THIS STUDY (1 month - 1 year, ~0.5-50 km), highlighted
  - Long-term climatic        -- decadal climate-driven risk assessment

Each class is drawn as a shaded zone with a few representative systems plotted as
labelled points. Zone extents and example systems live in the ZONES / POINTS dicts
at the top so they can be renamed / repositioned without touching the plot code.

    .venv/bin/python scripts/analysis/make_risk_landscape_figure.py

House style is copied from make_risk_figure_2024.py (no shared plotting module):
manuscript figure -> PNG @ 300 dpi + companion PDF in out/figures/.
"""

import argparse
import os

import numpy as np

# --------------------------------------------------------------------------- #
# House style (copied per-script; see make_risk_figure_2024.py)
# --------------------------------------------------------------------------- #
INK_PRIMARY = "#0b0b0b"    # titles / primary text
INK_SECONDARY = "#52514e"  # axis labels, ticks, captions
SURFACE = "#fcfcfb"        # near-white figure + axes background
GRID = "#e4e3de"           # gridlines and left/bottom spines

# Okabe-Ito colourblind-safe trio, one per model class.
C_SHORT = "#0072b2"   # blue
C_MEDIUM = "#009e73"  # green (this study, highlighted)
C_LONG = "#d55e00"    # vermillion

# --------------------------------------------------------------------------- #
# Editable content: axes are internal in (days, metres); tick labels are set below.
# --------------------------------------------------------------------------- #
DAY, WEEK, MONTH, SEASON, YEAR, DECADE = 1.0, 7.0, 30.0, 91.0, 365.0, 3650.0
KM = 1000.0

# Zone extents as (x0, x1) days and (y0, y1) metres; title_xy / desc_xy in data coords.
ZONES = {
    "short": dict(
        x=(0.5, 9.0), y=(100.0, 28.0 * KM), color=C_SHORT,
        title="Short-term\noperational",
        desc="Weather-driven &\nreal-time detection",
        title_xy=(0.62, 26.0 * KM), desc_xy=(0.62, 6.5 * KM),
    ),
    "medium": dict(
        x=(MONTH, YEAR), y=(0.5 * KM, 50.0 * KM), color=C_MEDIUM, highlight=True,
        title="Medium-term\nstrategic",
        desc="",
        title_xy=(34.0, 46.0 * KM), desc_xy=(34.0, 14.0 * KM),
    ),
    "long": dict(
        x=(2 * YEAR, 9000.0), y=(10.0 * KM, 130.0 * KM), color=C_LONG,
        title="Long-term\nclimatic",
        desc="Climate-driven\nrisk assessment",
        title_xy=(820.0, 120.0 * KM), desc_xy=(820.0, 40.0 * KM),
    ),
}

# Representative systems: markers + labels (xytext in data coords). `star` = the study.
# `leader=True` draws a thin leader line from the label to the marker.
POINTS = [
    dict(zone="short", label="VIIRS / MODIS\nactive-fire detection",
         x=0.8, y=0.40 * KM, tx=1.7, ty=0.40 * KM, ha="left", va="center"),
    dict(zone="short", label="Fire Weather Index /\nECMWF fire forecast",
         x=6.5, y=14.0 * KM, tx=6.5, ty=3.6 * KM, ha="center", va="top", leader=True),
    dict(zone="medium", label="This study", star=True,
         x=YEAR, y=0.5 * KM, tx=300.0, ty=0.62 * KM, ha="right", va="center"),
    dict(zone="medium", label="Seasonal fire-\npotential outlooks",
         x=250.0, y=26.0 * KM, tx=250.0, ty=6.5 * KM, ha="center", va="top",
         leader=True),
    dict(zone="long", label="Climate fire\nprojections (CMIP)",
         x=3200.0, y=60.0 * KM, tx=3200.0, ty=24.0 * KM, ha="center", va="top",
         leader=True),
]

# Axis ticks (position -> label).
XTICKS = [(DAY, "1 day"), (WEEK, "1 week"), (MONTH, "1 month"),
          (SEASON, "1 season"), (YEAR, "1 year"), (DECADE, "decade")]
YTICKS = [(10.0, "10 m"), (100.0, "100 m"), (1.0 * KM, "1 km"),
          (10.0 * KM, "10 km"), (100.0 * KM, "100 km\n(~1°)")]

XLIM = (0.4, 11000.0)
YLIM = (7.0, 160.0 * KM)


def _axfrac(x, y):
    """Map (days, metres) to axes-fraction coords for the given log limits."""
    lx0, lx1 = np.log10(XLIM[0]), np.log10(XLIM[1])
    ly0, ly1 = np.log10(YLIM[0]), np.log10(YLIM[1])
    fx = (np.log10(x) - lx0) / (lx1 - lx0)
    fy = (np.log10(y) - ly0) / (ly1 - ly0)
    return fx, fy


def plot(out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.6, 7.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

    # --- zones (rounded rectangles drawn in axes-fraction space for clean corners) ---
    for key in ("short", "long", "medium"):  # draw medium last so it sits on top
        z = ZONES[key]
        hi = z.get("highlight", False)
        (x0, x1), (y0, y1) = z["x"], z["y"]
        fx0, fy0 = _axfrac(x0, y0)
        fx1, fy1 = _axfrac(x1, y1)
        box = FancyBboxPatch(
            (fx0, fy0), fx1 - fx0, fy1 - fy0,
            boxstyle="round,pad=0.0,rounding_size=0.028",
            transform=ax.transAxes, mutation_aspect=1.0,
            facecolor=z["color"], edgecolor=z["color"],
            alpha=0.16 if hi else 0.10,
            linewidth=0, zorder=1,
        )
        ax.add_patch(box)
        edge = FancyBboxPatch(
            (fx0, fy0), fx1 - fx0, fy1 - fy0,
            boxstyle="round,pad=0.0,rounding_size=0.028",
            transform=ax.transAxes, mutation_aspect=1.0,
            facecolor="none", edgecolor=z["color"],
            linewidth=2.6 if hi else 1.4,
            linestyle="solid", zorder=2,
        )
        ax.add_patch(edge)

        # zone title + descriptor (data coords, anchored top-left inside the box)
        tx, ty = z["title_xy"]
        ax.text(tx, ty, z["title"], color=z["color"], fontsize=11.5,
                fontweight="bold", ha="left", va="top", zorder=5,
                linespacing=1.05)
        if z["desc"]:
            dx, dy = z["desc_xy"]
            ax.text(dx, dy, z["desc"], color=z["color"] if hi else INK_SECONDARY,
                    fontsize=9.5 if hi else 9.0, fontweight="bold" if hi else "normal",
                    ha="left", va="top", zorder=5, linespacing=1.05)

    # --- representative systems ---
    for p in POINTS:
        color = ZONES[p["zone"]]["color"]
        star = p.get("star", False)
        if star:
            ax.scatter([p["x"]], [p["y"]], marker="*", s=460, color=color,
                       edgecolor="white", linewidth=1.2, zorder=6)
        else:
            ax.scatter([p["x"]], [p["y"]], marker="o", s=78, color=color,
                       edgecolor=SURFACE, linewidth=1.1, zorder=6)
        arrow = (dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.7,
                      shrinkA=2, shrinkB=6, alpha=0.7)
                 if p.get("leader", False) else None)
        ax.annotate(
            p["label"], xy=(p["x"], p["y"]), xytext=(p["tx"], p["ty"]),
            color=INK_PRIMARY if star else INK_SECONDARY,
            fontsize=9.2 if star else 8.5,
            fontweight="bold" if star else "normal",
            ha=p["ha"], va=p["va"], zorder=6, linespacing=1.05,
            arrowprops=arrow,
        )

    # --- axes cosmetics ---
    ax.set_xticks([t for t, _ in XTICKS])
    ax.set_xticklabels([lab for _, lab in XTICKS])
    ax.set_yticks([t for t, _ in YTICKS])
    ax.set_yticklabels([lab for _, lab in YTICKS])
    ax.minorticks_off()
    ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=9.5)

    ax.set_xlabel("Forecast / assessment horizon  → longer", color=INK_SECONDARY,
                  fontsize=11.5, labelpad=8)
    ax.set_ylabel("Spatial resolution  → coarser", color=INK_SECONDARY,
                  fontsize=11.5, labelpad=8)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)

    fig.tight_layout(rect=[0.005, 0.01, 0.995, 0.99])

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor=SURFACE)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", facecolor=SURFACE)
    plt.close(fig)
    print(f"[risk_landscape] wrote {out_png}")
    print(f"[risk_landscape] wrote {os.path.splitext(out_png)[0] + '.pdf'}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_png", default="out/figures/fig_risk_landscape.png")
    args = ap.parse_args()
    plot(args.out_png)


if __name__ == "__main__":
    main()
