"""Publication figure (Fig 3): model accuracy across the spatial pyramid.

Two panels side by side -- PR-AUC and Cohen's kappa -- for seven series scored on
the SAME validation chip population (2023 + 2024), pooled within chips from 1 px
(~0.56 km) to 128 px (~71 km). Every series is given its own colour AND line
pattern / marker so they stay distinguishable where the curves overlap;
Climatology and Last-year burn are drawn in shades of grey as free baselines.

Both panels use the mean-pooled score (the block's expected burned fraction). A
block is positive if any pixel in it burned. PR-AUC is a threshold-free ranking
metric; Cohen's kappa is chance-corrected agreement, reported as the best value
over a swept threshold (see `pyramid_compare._best_kappa` for why the threshold
is swept, not fixed). Read the curves ACROSS models at a fixed scale, not along
the x axis -- block prevalence rises with block size, so PR-AUC rises with it.

The per-block metrics are computed with `pyramid_compare.model_levels` /
`baseline_levels` (reused, single source of truth) and cached to a CSV; pass
`--from_csv` to restyle without the ~minutes-long recompute.

    .venv/bin/python scripts/analysis/make_pyramid_figure.py
    .venv/bin/python scripts/analysis/make_pyramid_figure.py --from_csv out/figures/fig_pyramid_val.csv
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyramid_compare import (  # noqa: E402
    DEFAULT_BLOCKS, GRID, INK_SECONDARY, KM_PER_PIXEL, SURFACE,
    baseline_levels, model_levels, write_csv,
)

# --------------------------------------------------------------------------- #
# Series: (name, prediction dir, colour, linestyle, marker). Models first
# (Okabe-Ito colours + distinct patterns), then the two grey baselines.
# --------------------------------------------------------------------------- #
MODELS = [
    ("Factored",  "out/baselines/factored_v1",       "#0072b2", "-",             "o"),
    ("MLP RF1",   "out/baselines/baseline_mlp_rf1",  "#d55e00", "--",            "s"),
    ("LSTM",      "out/baselines/baseline_lstm",     "#009e73", "-.",            "^"),
    ("U-Net",     "out/baselines/baseline_unet_full","#e69f00", ":",             "D"),
    ("ConvLSTM",  "out/baselines/baseline_convlstm", "#cc79a7", (0, (3, 1, 1, 1)),"v"),
]
BASELINES = [
    # (name, kind, colour, linestyle, marker) -- shades of grey.
    ("Climatology",    "climatology", "#4d4d4d", (0, (4, 2)), "x"),
    ("Last-year burn", "last_year",   "#9a988f", (0, (1, 1.5)), "P"),
]
# No-skill floor derived from the labels: PR-AUC = block prevalence, ROC-AUC = 0.5.
RANDOM = ("Random", "#8a8880", "--", "")  # (name, colour, linestyle, marker)

DEFAULT_LABEL_DIR = "out/label_mosaics"
DEFAULT_CLIM = "out/label_mosaics/climatology_2013_2022.tif"

# Metric -> (axis label, y-limits). Left: PR-AUC, a threshold-free ranking
# metric on the mean-pooled score. Right: Cohen's kappa (best over swept
# thresholds on the same mean-pooled score) -- chance-corrected agreement.
PANELS = [("pr_auc", "PR-AUC", (0.0, 1.0)),
          ("kappa", "Cohen's κ", (0.0, 0.9))]


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

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2), facecolor=SURFACE)
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
    fig.tight_layout(rect=[0, 0.135, 1, 1.0])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor=SURFACE)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", facecolor=SURFACE)
    plt.close(fig)
    print(f"[pyramid_fig] wrote {out_png}")
    print(f"[pyramid_fig] wrote {os.path.splitext(out_png)[0] + '.pdf'}")


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
    ap.add_argument("--climatology", default=DEFAULT_CLIM)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--from_csv", default=None,
                    help="restyle from an existing CSV instead of recomputing")
    ap.add_argument("--out_png", default="out/figures/fig_pyramid_val.png")
    ap.add_argument("--out_csv", default="out/figures/fig_pyramid_val.csv")
    args = ap.parse_args()

    if args.from_csv:
        series, blocks = load_from_csv(args.from_csv)
        _append_random(series)
        plot(series, blocks, args.out_png)
        return

    blocks = ([int(b) for b in args.blocks.split(",") if b.strip()]
              if args.blocks else DEFAULT_BLOCKS)
    series, rows = [], []
    for name, path, *_ in MODELS:
        print(f"[pyramid_fig] {name}: {path}", flush=True)
        levels = model_levels(path, blocks, "mean", "max", 0.5, "best")
        series.append({"name": name, "levels": levels})
        rows += [{"model": name, **lvl} for lvl in levels]

    ref = MODELS[0][1]  # score baselines on the same chips
    for name, kind, *_ in BASELINES:
        print(f"[pyramid_fig] baseline {name}", flush=True)
        levels = baseline_levels(ref, kind, args.label_dir, args.climatology,
                                 blocks, "mean", "max", 0.5, "best")
        series.append({"name": name, "levels": levels})
        rows += [{"model": name, **lvl} for lvl in levels]

    write_csv(rows, args.out_csv)
    _append_random(series)
    plot(series, blocks, args.out_png)


if __name__ == "__main__":
    main()
