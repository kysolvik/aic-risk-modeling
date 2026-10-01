"""Publication figure (Fig 6): PR-AUC by forested vs non-forested land cover.

Grouped bars, one pair per predictor (forest / non-forest PR-AUC), on the
identical pixel population. v3p version (9/30): the five v3p CV architectures on
their fwdpair_2022 eval years (2022 + 2023, same fold and names as Fig 3) plus the
two free baselines (burn frequency over the fold's train years, last-year burn). Each stratum's no-skill floor -- its fire
prevalence -- is drawn as a dotted line; the gap above it is the skill, because
raw PR-AUC scales with prevalence and prevalence differs between strata.

All heavy lifting (canonical chip set, shared labels + forest fraction, model /
baseline score reads, stratified PR-AUC) is reused from `compare_forest_split.py`.
Metrics are cached to a CSV; pass `--from_csv` to restyle without recomputing.

    .venv/bin/python scripts/analysis/make_forest_figure.py
    .venv/bin/python scripts/analysis/make_forest_figure.py \
        --from_csv out/figures/fig_forest_split_fwdpair_2022.csv

Forest chips must exist first, on the v3 grid (export_forest_chips.py with the
v3 profile template, no --invert_yres) under <forest_dir>/<year>/.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_forest_split import (  # noqa: E402
    FOREST_COLOR, NONFOREST_COLOR, GRID, INK_SECONDARY, SURFACE,
    _full_metrics, canonical_chips, load_baseline_scores, load_model_scores,
    load_shared, stratify, write_split_table,
)

# Predictors in display order: the five CV archs (as in Fig 3), then the two grey
# baselines. Chips are read from <preds_root>/<arch>/<fold>/<year>/chips/.
MODELS = [
    ("Factored", "factored_v3p_union4_monthlyattn_wide_yeargain"),
    ("U-Net",    "unet_v3p_union4"),
    ("ViT",      "vit_test_v3p_union4"),
    ("MLP",      "mlp_v3p_union4_flat"),
    ("LSTM",     "lstm_v3p_union4"),
]
BASELINES = [("Burn frequency", "climatology"), ("Last-year burn", "last_year")]
ORDER = [m[0] for m in MODELS] + [b[0] for b in BASELINES]

DEFAULT_FOREST_DIR = "out/forest_v3p"
DEFAULT_LABEL_DIR = "out/label_mosaics_v3p_union4"
DEFAULT_CLIM = "out/label_mosaics_v3p_union4/climatology_2013_2021.tif"
DEFAULT_PREDS_ROOT = "out/cv/preds"


def plot_split(order, per_model, prevalence, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)

    x = np.arange(len(order))
    w = 0.38
    forest = [per_model[n]["forest"]["pr_auc"] for n in order]
    nonf = [per_model[n]["non_forest"]["pr_auc"] for n in order]
    b1 = ax.bar(x - w / 2, forest, w, color=FOREST_COLOR, label="Forest", zorder=3)
    b2 = ax.bar(x + w / 2, nonf, w, color=NONFOREST_COLOR, label="Non-forest", zorder=3)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK_SECONDARY, padding=2)

    # No-skill floors: each stratum's fire prevalence (identical across predictors).
    hf = ax.axhline(prevalence["forest"], color=FOREST_COLOR, linestyle=":",
                    linewidth=1.5, zorder=2, label="Forest no-skill (prevalence)")
    hn = ax.axhline(prevalence["non_forest"], color=NONFOREST_COLOR, linestyle=":",
                    linewidth=1.5, zorder=2, label="Non-forest no-skill (prevalence)")

    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylabel("PR-AUC", fontsize=11.5, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    ax.set_ylim(0, max(0.001, max(forest + nonf)) * 1.18)
    ax.legend([b1, b2, hf, hn],
              ["Forest", "Non-forest", "Forest no-skill (prevalence)",
               "Non-forest no-skill (prevalence)"],
              loc="upper right", frameon=False, fontsize=9.5,
              labelcolor=INK_SECONDARY)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor=SURFACE)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", facecolor=SURFACE)
    plt.close(fig)
    print(f"[forest_fig] wrote {out_png}")
    print(f"[forest_fig] wrote {os.path.splitext(out_png)[0] + '.pdf'}")


def load_from_csv(path):
    import csv
    per_model, prevalence = {}, {}
    for r in csv.DictReader(open(path)):
        name, stratum = r["model"], r["stratum"]
        per_model.setdefault(name, {})[stratum] = {"pr_auc": float(r["pr_auc"])}
        if stratum in ("forest", "non_forest"):
            prevalence[stratum] = float(r["prevalence"])
    order = [n for n in ORDER if n in per_model] or list(per_model)
    return order, per_model, prevalence


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forest_dir", default=DEFAULT_FOREST_DIR)
    ap.add_argument("--label-dir", dest="label_dir", default=DEFAULT_LABEL_DIR)
    ap.add_argument("--climatology", default=DEFAULT_CLIM)
    ap.add_argument("--preds_root", default=DEFAULT_PREDS_ROOT)
    ap.add_argument("--fold", default="fwdpair_2022")
    ap.add_argument("--years", default="2022,2023", help="eval years of --fold")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="forest-fraction cutoff defining the two strata")
    ap.add_argument("--from_csv", default=None, help="restyle from an existing CSV")
    ap.add_argument("--out_png", default="out/figures/fig_forest_split_fwdpair_2022.png")
    ap.add_argument("--out_csv", default="out/figures/fig_forest_split_fwdpair_2022.csv")
    args = ap.parse_args()

    if args.from_csv:
        order, per_model, prevalence = load_from_csv(args.from_csv)
        plot_split(order, per_model, prevalence, args.out_png)
        return

    years = [int(y) for y in args.years.split(",")]

    def spec(arch):  # explicit [(chips_dir, year)]: the fold dir also holds test years
        return [(os.path.join(args.preds_root, arch, args.fold, str(y), "chips"), y)
                for y in years]

    chips = canonical_chips(spec(MODELS[0][1]), args.forest_dir, None)
    labels, forest = load_shared(chips)
    print(f"[forest_fig] {len(chips)} chips, {labels.size} pixels", flush=True)

    scores_by = {}
    for name, arch in MODELS:
        print(f"[forest_fig] reading {name}: {arch}/{args.fold}", flush=True)
        scores_by[name] = load_model_scores(spec(arch), chips, None)
    for name, kind in BASELINES:
        print(f"[forest_fig] reading baseline {name}", flush=True)
        scores_by[name] = load_baseline_scores(chips, kind, args.label_dir,
                                               args.climatology)

    def _prev(mask):
        return float(labels[mask].mean()) if mask.any() else float("nan")
    prevalence = {"forest": _prev(forest >= args.threshold),
                  "non_forest": _prev(forest < args.threshold)}

    full = lambda s, l: _full_metrics(s, l, 0.5)  # noqa: E731
    per_model = {name: stratify(scores_by[name], labels, forest, args.threshold,
                                full, include_all=True) for name in ORDER}
    write_split_table(ORDER, per_model, args.threshold, args.out_csv,
                      os.path.splitext(args.out_csv)[0] + ".md")
    plot_split(ORDER, per_model, prevalence, args.out_png)


if __name__ == "__main__":
    main()
