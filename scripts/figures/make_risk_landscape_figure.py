"""Conceptual figure: fire-risk model classes on a resolution x time-horizon plane (no data).

Zones and example systems are edited in the ZONES / POINTS dicts.
Usage: make_risk_landscape_figure.py"""

import argparse

import numpy as np

from style import GRID, INK_PRIMARY, INK_SECONDARY, SURFACE, save_figure

# Okabe-Ito, one per model class.
C_SHORT = "#0072b2"
C_MEDIUM = "#009e73"
C_LONG = "#d55e00"

DAY, WEEK, MONTH, SEASON, YEAR, DECADE = 1.0, 7.0, 30.0, 91.0, 365.0, 3650.0
KM = 1000.0

ZONES = {
    "short": dict(
        x=(0.5, 9.0), y=(100.0, 28.0 * KM), color=C_SHORT,
        title="Short-term\noperational",
        desc="Weather-driven &\nreal-time detection",
    ),
    "medium": dict(
        x=(MONTH, YEAR), y=(0.5 * KM, 50.0 * KM), color=C_MEDIUM, highlight=True,
        title="Medium-term\nstrategic",
        desc="Climate & human\nactivity patterns",
    ),
    "long": dict(
        x=(2 * YEAR, 9000.0), y=(10.0 * KM, 130.0 * KM), color=C_LONG,
        title="Long-term\noutlooks",
        desc="Climate-driven\nrisk assessment",
    ),
}
TITLE_INSET = (9, -7)
DESC_GAP = 3

POINTS = [
    dict(zone="short", label="VIIRS / MODIS\nactive-fire detection",
         x=0.8, y=0.40 * KM, dx=9, dy=0, ha="left", va="center"),
    dict(zone="short", label="Fire Weather Index /\nECMWF fire forecast",
         x=4.0, y=7.0 * KM, dx=0, dy=-30, ha="center", va="top", leader=True),
    dict(zone="medium", label="This study", star=True,
         x=YEAR, y=0.5 * KM, dx=-12, dy=13, ha="right", va="center"),
    dict(zone="medium", label="Seasonal fire\npotential",
         x=200.0, y=10.0 * KM, dx=0, dy=-30, ha="center", va="top", leader=True),
    dict(zone="long", label="Climate fire\nprojections (CMIP)",
         x=4200.0, y=30.0 * KM, dx=0, dy=-9, ha="center", va="top"),
]

XTICKS = [(DAY, "1 day"), (WEEK, "1 week"), (MONTH, "1 month"),
          (SEASON, "1 season"), (YEAR, "1 year"), (DECADE, "decade")]
YTICKS = [(10.0, "10 m"), (100.0, "100 m"), (1.0 * KM, "1 km"),
          (10.0 * KM, "10 km"), (100.0 * KM, "100 km\n(~1°)")]

XLIM = (0.4, 11000.0)
YLIM = (7.0, 160.0 * KM)


def _axfrac(x, y):
    lx0, lx1 = np.log10(XLIM[0]), np.log10(XLIM[1])
    ly0, ly1 = np.log10(YLIM[0]), np.log10(YLIM[1])
    fx = (np.log10(x) - lx0) / (lx1 - lx0)
    fy = (np.log10(y) - ly0) / (ly1 - ly0)
    return fx, fy


def check_layout(fig, ax, texts, boxes, pad_px=6):
    """Warn about text outside its zone box, touching another zone, overlapping text, or crossed by a leader."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    disp = {k: Bbox.from_extents(*ax.transAxes.transform([(b[0], b[1]), (b[2], b[3])]).ravel())
            for k, b in boxes.items()}
    ext = [(z, t, Text.get_window_extent(t, r)) for z, t in texts]
    leaders = [(t, t.arrow_patch.get_window_extent(r)) for _, t in texts
               if getattr(t, "arrow_patch", None) is not None]
    bad = 0
    for z, t, e in ext:
        name = t.get_text().replace("\n", " ")
        own = disp[z].padded(-pad_px)
        if not (own.x0 <= e.x0 and e.x1 <= own.x1 and own.y0 <= e.y0 and e.y1 <= own.y1):
            print(f"[risk_landscape] LAYOUT: '{name}' crosses its {z} box edge")
            bad += 1
        for k, b in disp.items():
            if k != z and b.padded(pad_px).overlaps(e):
                print(f"[risk_landscape] LAYOUT: '{name}' touches the {k} box")
                bad += 1
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            if ext[i][2].overlaps(ext[j][2]):
                print(f"[risk_landscape] LAYOUT: '{ext[i][1].get_text()!r}' overlaps "
                      f"'{ext[j][1].get_text()!r}'")
                bad += 1
    for owner, le in leaders:
        for _, t, e in ext:
            if t is not owner and le.overlaps(e):
                print(f"[risk_landscape] LAYOUT: leader of {owner.get_text()!r} crosses "
                      f"{t.get_text()!r}")
                bad += 1
    print(f"[risk_landscape] layout check: {'OK' if not bad else f'{bad} problem(s)'}")


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

    texts, boxes = [], {}

    for key in ("short", "long", "medium"):
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

        t = ax.annotate(z["title"], xy=(x0, y1), xytext=TITLE_INSET,
                        textcoords="offset points", color=z["color"], fontsize=11.5,
                        fontweight="bold", ha="left", va="top", zorder=5,
                        linespacing=1.05)
        texts.append((key, t))
        if z["desc"]:
            d = ax.annotate(z["desc"], xy=(0, 0), xycoords=t, xytext=(0, -DESC_GAP),
                            textcoords="offset points",
                            color=INK_SECONDARY,
                            fontsize=9.0,
                            fontweight="normal",
                            ha="left", va="top", zorder=5, linespacing=1.05)
            texts.append((key, d))
        boxes[key] = (fx0, fy0, fx1, fy1)

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
        a = ax.annotate(
            p["label"], xy=(p["x"], p["y"]), xytext=(p["dx"], p["dy"]),
            textcoords="offset points",
            color=INK_PRIMARY if star else INK_SECONDARY,
            fontsize=9.2 if star else 8.5,
            fontweight="bold" if star else "normal",
            ha=p["ha"], va=p["va"], zorder=6, linespacing=1.05,
            arrowprops=arrow,
        )
        texts.append((p["zone"], a))

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
    check_layout(fig, ax, texts, boxes)

    save_figure(fig, out_png, "risk_landscape")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_png", default="out/figures/fig_risk_landscape.png")
    args = ap.parse_args()
    plot(args.out_png)


if __name__ == "__main__":
    main()
