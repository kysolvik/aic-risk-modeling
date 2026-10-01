"""Publication figure (Fig 3): model accuracy across the spatial pyramid.

PR-AUC for the five v3p CV
architectures plus free baselines, scored on the SAME chip population: one CV
fold's eval years (default fwdpair_2022 -> 2022 + 2023; model trained 2013-2021),
pooled within chips from 1 px (~0.46 km) to 128 px (~59 km). Labels are the
union4 target; Burn frequency (climatology) is the fold's pixel-wise burn frequency over its
train years (as cv_collect_results.Climatology), Last-year burn the previous
year's union4 label. Every series is given its own colour AND line
pattern / marker so they stay distinguishable where the curves overlap;
Burn frequency and Last-year burn are drawn in shades of grey as free baselines.

Both panels use the mean-pooled score (the block's expected burned fraction) by
default; `--score_pool max` pools every series (models and baselines) by the
block MAX instead ("highest-risk pixel in the block"). A block is positive if
any pixel in it burned. PR-AUC is a threshold-free ranking
metric; Cohen's kappa is chance-corrected agreement, reported as the best value
over a swept threshold (see `pyramid_compare._best_kappa` for why the threshold
is swept, not fixed). Read the curves ACROSS models at a fixed scale, not along
the x axis -- block prevalence rises with block size, so PR-AUC rises with it.

The per-block metrics are computed with `pyramid_compare.model_levels` /
`baseline_levels` (reused, single source of truth) and cached to a CSV; pass
`--from_csv` to restyle without the ~minutes-long recompute.

    .venv/bin/python scripts/analysis/make_pyramid_figure.py
    .venv/bin/python scripts/analysis/make_pyramid_figure.py --from_csv out/figures/fig_pyramid_fwdpair_2022.csv
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyramid_compare import (  # noqa: E402
    DEFAULT_BLOCKS, GRID, INK_SECONDARY, SURFACE,
    baseline_levels, model_levels, write_csv,
)

KM_PER_PIXEL = 0.463312716528  # v3 MODIS sinusoidal grid (pyramid_compare's is the v2 0.005 deg)

# --------------------------------------------------------------------------- #
# Series: (name, prediction dir, colour, linestyle, marker). Models first
# (Okabe-Ito colours + distinct patterns), then the two grey baselines.
# --------------------------------------------------------------------------- #
# The "path" is the CV arch; chips are read from
# <preds_root>/<arch>/<fold>/<year>/chips/ for each eval year.
MODELS = [
    ("Factored",  "factored_v3p_union4_monthlyattn_wide_yeargain", "#0072b2", "-",  "o"),
    ("U-Net",     "unet_v3p_union4",     "#e69f00", ":",               "D"),
    ("ViT",       "vit_test_v3p_union4", "#cc79a7", (0, (3, 1, 1, 1)), "v"),
    ("MLP",       "mlp_v3p_union4_flat", "#d55e00", "--",              "s"),
    ("LSTM",      "lstm_v3p_union4",     "#009e73", "-.",              "^"),
]
BASELINES = [
    # (name, kind, colour, linestyle, marker) -- shades of grey.
    ("Burn frequency", "climatology", "#4d4d4d", (0, (4, 2)), "x"),
    ("Last-year burn", "last_year",   "#9a988f", (0, (1, 1.5)), "P"),
]
# No-skill floor derived from the labels: PR-AUC = block prevalence, ROC-AUC = 0.5.
RANDOM = ("Random", "#8a8880", "--", "")  # (name, colour, linestyle, marker)

DEFAULT_LABEL_DIR = "out/label_mosaics_v3p_union4"
DEFAULT_PREDS_ROOT = "out/cv/preds"

# Metric -> (axis label, y-limits), one panel each. PR-AUC: a threshold-free
# ranking metric on the mean-pooled score. Cohen's kappa (best over swept
# thresholds, still computed into the CSV) was a second panel until 9/30 --
# add ("kappa", "Cohen's κ", (0.0, 0.9)) back to plot it.
PANELS = [("pr_auc", "PR-AUC", (0.0, 1.0))]


def _style_for(name):
    for nm, _p, col, ls, mk in MODELS:
        if nm == name:
            return col, ls, mk, True
    for nm, _k, col, ls, mk in BASELINES:
        if nm == name:
            return col, ls, mk, False
    if name == RANDOM[0]:
        return RANDOM[1], RANDOM[2], RANDOM[3], False
    return "#b6b4ab", "-", "o", False


def _append_random(series):
    """Append the no-skill series from any real series' per-block prevalence."""
    base = series[0]["levels"]
    # No-skill floor: PR-AUC = block prevalence, ROC-AUC = 0.5, kappa = 0.
    series.append({"name": RANDOM[0],
                   "levels": [{"pr_auc": lvl["prevalence"], "roc_auc": 0.5,
                               "kappa": 0.0}
                              for lvl in base]})


def plot(series, blocks, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PANELS), figsize=(1.0 + 6.4 * len(PANELS), 6.4),
                             facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    xs = np.arange(len(blocks))
    for ax, (metric, ylabel, ylim) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        for s in series:
            col, ls, mk, is_model = _style_for(s["name"])
            is_random = s["name"] == RANDOM[0]
            y = [lvl[metric] for lvl in s["levels"]]
            ax.plot(xs, y, marker=mk, color=col,
                    linewidth=1.5 if is_random else 2.0, linestyle=ls,
                    markersize=6.0, markeredgecolor=SURFACE,
                    markeredgewidth=0.6, label=s["name"],
                    zorder=6 if is_model else (2 if is_random else 4))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b}\n{b * KM_PER_PIXEL:.3g} km" for b in blocks],
                           fontsize=8.5, color=INK_SECONDARY)
        ax.set_xlabel("Block size (pixels per side / km)", fontsize=10.5,
                      color=INK_SECONDARY)
        ax.set_ylabel(ylabel, fontsize=11.5, color=INK_SECONDARY)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
        ax.set_xlim(-0.35, len(blocks) - 1 + 0.35)
        ax.set_ylim(*ylim)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, labelcolor=INK_SECONDARY,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=2.6)
    fig.tight_layout(rect=[0, 0.1, 1, 1.0])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor=SURFACE)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", facecolor=SURFACE)
    plt.close(fig)
    print(f"[pyramid_fig] wrote {out_png}")
    print(f"[pyramid_fig] wrote {os.path.splitext(out_png)[0] + '.pdf'}")


def build_climatology(label_dir, years, out_path):
    """Pixel-wise burn frequency over `years` (mean of label > 0), cached as a tif."""
    import rasterio as rio
    if os.path.exists(out_path):
        return out_path
    total, profile = None, None
    for y in years:
        with rio.open(os.path.join(label_dir, f"label_{y}.tif")) as s:
            lab = s.read(1) > 0
            if total is None:
                total, profile = np.zeros(lab.shape, np.uint16), s.profile
            elif s.transform != profile["transform"] or lab.shape != total.shape:
                raise ValueError(f"label_{y}.tif is not on the same grid")
        total += lab
    profile.update(dtype="float32", count=1, nodata=None, compress="deflate")
    with rio.open(out_path, "w", **profile) as d:
        d.write((total / len(years)).astype(np.float32), 1)
    print(f"[pyramid_fig] wrote {out_path}")
    return out_path


def load_from_csv(path):
    import csv
    by, order = {}, []
    for r in csv.DictReader(open(path)):
        name = r["model"]
        if name not in by:
            by[name] = []
            order.append(name)
        by[name].append({"block": int(r["block"]),
                         "pr_auc": float(r["pr_auc"]),
                         "roc_auc": float(r["roc_auc"]),
                         "kappa": float(r["kappa"]),
                         "prevalence": float(r["prevalence"])})
    blocks = sorted({lvl["block"] for rows in by.values() for lvl in rows})
    series = [{"name": n, "levels": sorted(by[n], key=lambda d: d["block"])}
              for n in order]
    return series, blocks


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-dir", default=DEFAULT_LABEL_DIR)
    ap.add_argument("--preds_root", default=DEFAULT_PREDS_ROOT)
    ap.add_argument("--fold", default="fwdpair_2022")
    ap.add_argument("--years", default="2022,2023", help="eval years of --fold")
    ap.add_argument("--clim_years", default="2013-2021",
                    help="the fold's train years (climatology)")
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--score_pool", choices=["mean", "max"], default="mean",
                    help="block pooling of every series' score (PR-AUC/ROC-AUC/kappa)")
    ap.add_argument("--from_csv", default=None,
                    help="restyle from an existing CSV instead of recomputing")
    ap.add_argument("--out_png", default=None,
                    help="default out/figures/fig_pyramid_<fold>.png (csv alongside)")
    args = ap.parse_args()
    suffix = "" if args.score_pool == "mean" else f"_{args.score_pool}pool"
    out_png = args.out_png or f"out/figures/fig_pyramid_{args.fold}{suffix}.png"
    out_csv = os.path.splitext(out_png)[0] + ".csv"

    if args.from_csv:
        series, blocks = load_from_csv(args.from_csv)
        _append_random(series)
        plot(series, blocks, out_png)
        return

    blocks = ([int(b) for b in args.blocks.split(",") if b.strip()]
              if args.blocks else DEFAULT_BLOCKS)
    years = [int(y) for y in args.years.split(",")]
    a, _, b = args.clim_years.partition("-")
    clim_years = list(range(int(a), int(b or a) + 1))
    if max(clim_years) >= min(years):
        raise ValueError(f"climatology years {args.clim_years} overlap eval years {years}")
    clim = build_climatology(
        args.label_dir, clim_years,
        os.path.join(args.label_dir, f"climatology_{clim_years[0]}_{clim_years[-1]}.tif"))

    def inventory(arch):
        return [(os.path.join(args.preds_root, arch, args.fold, str(y), "chips"), y)
                for y in years]

    series, rows = [], []
    for name, arch, *_ in MODELS:
        print(f"[pyramid_fig] {name}: {arch}/{args.fold} {years}", flush=True)
        levels = model_levels(inventory(arch), blocks, args.score_pool, "max",
                              0.5, "best")
        series.append({"name": name, "levels": levels})
        rows += [{"model": name, **lvl} for lvl in levels]

    ref = inventory(MODELS[0][1])  # score baselines on the same chips
    for name, kind, *_ in BASELINES:
        print(f"[pyramid_fig] baseline {name}", flush=True)
        levels = baseline_levels(ref, kind, args.label_dir, clim,
                                 blocks, args.score_pool, "max", 0.5, "best")
        series.append({"name": name, "levels": levels})
        rows += [{"model": name, **lvl} for lvl in levels]

    for r in rows:
        r["km"] = round(r["block"] * KM_PER_PIXEL, 3)
    write_csv(rows, out_csv)
    _append_random(series)
    plot(series, blocks, out_png)


if __name__ == "__main__":
    main()
