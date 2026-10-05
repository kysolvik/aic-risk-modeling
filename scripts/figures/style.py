"""House style shared by the manuscript figure scripts."""

import os

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e4e3de"
OUTSIDE = "#f2f1ee"          # map background outside the basin
RISK_CMAP = "YlOrRd"
DIFF_CMAP = ["#0072b2", "#f7f7f7", "#d55e00"]   # below -> zero -> above

SHP = "../data/Limites_RAISG_2025/Lim_Raisg.shp"
CALIBRATOR = "out/cv/calibrator_platt_cv2018_2023.npz"
LABEL_DIR = "out/label_mosaics_v3p_union4"


def style_axes(ax, grid_axis="both"):
    """Surface background, light grid behind the data, left/bottom spines only."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def save_figure(fig, out_png, tag, tight=False):
    """Write `out_png` at 300 dpi plus a matching .pdf."""
    kw = {"facecolor": SURFACE}
    if tight:
        kw["bbox_inches"] = "tight"
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=300, **kw)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", **kw)
    print(f"[{tag}] wrote {out_png} (+ .pdf)")
