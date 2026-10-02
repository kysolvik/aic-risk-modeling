"""Publication figure (Fig 6): PR-AUC by forested vs non-forested land cover.

Grouped bars, one pair per predictor (forest / non-forest PR-AUC), on the
identical pixel population. v3p version (9/30): the five v3p CV architectures on
their fwdpair_2022 eval years (2022 + 2023, same fold and names as Fig 3) plus the
two free baselines (burn frequency over the fold's train years, last-year burn). Each stratum's no-skill floor -- its fire
prevalence -- is drawn as a dotted line; the gap above it is the skill, because
raw PR-AUC scales with prevalence and prevalence differs between strata.

Every predictor is scored on one canonical chip set with shared labels + forest
fraction. Metrics are cached to a CSV; pass `--from_csv` to restyle without
recomputing.

    .venv/bin/python scripts/figures/make_forest_figure.py
    .venv/bin/python scripts/figures/make_forest_figure.py \
        --from_csv out/figures/fig_forest_split_fwdpair_2022.csv

Forest chips must exist first, on the v3 grid (export_forest_chips.py with the
v3 profile template) under <forest_dir>/<year>/.
"""

import argparse
import os

import numpy as np

from aic_risk_modeling.eval.chips import chip_pairs, read_window
from aic_risk_modeling.eval.metrics import binary_metrics
from style import INK_SECONDARY, SURFACE, save_figure, style_axes

FOREST_COLOR = "#1baf7a"      # forest stratum (green)
NONFOREST_COLOR = "#eb6834"   # non-forest stratum (orange)

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


def _bounds_match(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(tuple(a), tuple(b)))


def _chip_key(path):
    """The '<x>-<y>.tif' id shared by the out_/mask_/forest_ chips of one tile."""
    return os.path.basename(path).split("_", 1)[1]


def _inventory(spec):
    """[(out_path, year)] for an explicit [(chips_dir, year), ...] spec."""
    return [(o, str(year)) for chips_dir, year in spec for o, _ in chip_pairs(chips_dir)]


def canonical_chips(spec, forest_dir):
    """Ordered chip list from the reference model: key, year, mask + forest path.

    Every model is scored on this one identical, sorted pixel population; the
    labels and forest fraction are read once from here rather than re-read per
    model (they are byte-identical across models).
    """
    chips = []
    for out_path, year in _inventory(spec):
        fpath = os.path.join(forest_dir, year,
                             os.path.basename(out_path).replace("out_", "forest_", 1))
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"missing forest chip {fpath} for {out_path} -- run "
                "export_forest_chips.py for that year first")
        mask_path = os.path.join(os.path.dirname(out_path),
                                 os.path.basename(out_path).replace("out_", "mask_", 1))
        chips.append({"key": (year, _chip_key(out_path)), "year": year,
                      "mask_path": mask_path, "forest_path": fpath})
    chips.sort(key=lambda c: c["key"])
    return chips


def load_shared(chips):
    """(labels, forest_frac) flat over the canonical chips; records per-chip bounds.

    Labels and forest are identical across models, so they are read exactly once.
    The bounds captured here drive the baseline windowed reads.
    """
    import rasterio as rio
    labels, forest = [], []
    for c in chips:
        with rio.open(c["mask_path"]) as m:
            lb = m.read(1)
            c["bounds"] = tuple(m.bounds)
            c["shape"] = lb.shape
        with rio.open(c["forest_path"]) as f:
            fr = f.read(1).astype(np.float32)
            fb = f.bounds
        if fr.shape != lb.shape:
            raise ValueError(f"forest {fr.shape} != mask {lb.shape} for {c['mask_path']}")
        if not _bounds_match(c["bounds"], fb):
            raise ValueError(f"forest bounds {tuple(fb)} != mask bounds {c['bounds']} "
                             f"for {c['mask_path']}")
        labels.append((lb > 0).ravel())
        forest.append(fr.ravel())
    return np.concatenate(labels), np.concatenate(forest)


def load_model_scores(spec, chips):
    """Model scores flat, in the canonical chip order (joined by the <x>-<y> key)."""
    import rasterio as rio
    by_key = {(year, _chip_key(out_path)): out_path for out_path, year in _inventory(spec)}
    missing = [c["key"] for c in chips if c["key"] not in by_key]
    if missing:
        raise FileNotFoundError(
            f"{spec[0][0]} is missing {len(missing)} of {len(chips)} chips, "
            f"e.g. {missing[:3]}")
    scores = []
    for c in chips:
        with rio.open(by_key[c["key"]]) as s:
            scores.append(s.read(1).astype(np.float32).ravel())
    return np.concatenate(scores)


def load_baseline_scores(chips, kind, label_dir, clim_path):
    """Baseline score field flat, in canonical chip order (windowed mosaic reads).

    `kind` is "last_year" (previous year's burn from label_<year-1>.tif) or
    "climatology" (the prebuilt mean-burn-frequency raster), windowed to each
    chip's bounds.
    """
    import rasterio as rio
    handles = {}

    def _src(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if path not in handles:
            handles[path] = rio.open(path)
        return handles[path]

    scores = []
    try:
        for c in chips:
            src = (_src(clim_path) if kind == "climatology"
                   else _src(os.path.join(label_dir, f"label_{int(c['year']) - 1}.tif")))
            sc = read_window(src, c["bounds"], c["shape"]).astype(np.float32)
            if kind == "last_year":
                sc = (sc > 0).astype(np.float32)
            scores.append(sc.ravel())
    finally:
        for h in handles.values():
            h.close()
    return np.concatenate(scores)


def _full_metrics(scores, labels, hard_threshold):
    """PR-AUC + prevalence + lift + F1 for one already-subset pixel population."""
    if labels.size == 0:
        return dict(pr_auc=float("nan"), prevalence=float("nan"),
                    lift=float("nan"), f1=float("nan"), n_pixels=0)
    m = binary_metrics(labels, scores >= hard_threshold, scores=scores)
    prev = float(labels.mean())
    return dict(pr_auc=float(m["pr_auc"]), prevalence=prev,
                lift=(float(m["pr_auc"]) / prev if prev > 0 else float("nan")),
                f1=float(m["f1"]), n_pixels=int(labels.size))


def stratify(scores, labels, forest, cutoff, metric_fn, include_all):
    """Apply `metric_fn(scores, labels)` to the forest / non_forest (/ all) strata."""
    fmask = forest >= cutoff
    out = {"forest": metric_fn(scores[fmask], labels[fmask]),
           "non_forest": metric_fn(scores[~fmask], labels[~fmask])}
    if include_all:
        out["all"] = metric_fn(scores, labels)
    return out


METRIC_COLS = ["pr_auc", "lift", "prevalence", "f1", "n_pixels"]


def write_split_table(order, per_model, cutoff, out_csv, out_md):
    header = ["model", "stratum", *METRIC_COLS]
    lines = [",".join(header)]
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    for name in order:
        for stratum in ("forest", "non_forest", "all"):
            m = per_model[name][stratum]
            vals = [f"{m['pr_auc']:.4f}", f"{m['lift']:.3f}", f"{m['prevalence']:.4f}",
                    f"{m['f1']:.4f}", str(m["n_pixels"])]
            lines.append(",".join([name, stratum, *vals]))
            md.append("| " + " | ".join([name, stratum, *vals]) + " |")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(out_md, "w") as f:
        f.write(f"# PR-AUC by land cover (forest fraction cutoff {cutoff:g})\n\n"
                "`lift` = pr_auc / prevalence is the fair forest-vs-non-forest "
                "comparison; raw PR-AUC scales with each stratum's fire "
                "prevalence.\n\n" + "\n".join(md) + "\n")
    print(f"[forest-split] wrote {out_csv} and {out_md}")


def plot_split(order, per_model, prevalence, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 5.4), facecolor=SURFACE)
    style_axes(ax, grid_axis="y")

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
    save_figure(fig, out_png, "forest_fig")
    plt.close(fig)


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

    chips = canonical_chips(spec(MODELS[0][1]), args.forest_dir)
    labels, forest = load_shared(chips)
    print(f"[forest_fig] {len(chips)} chips, {labels.size} pixels", flush=True)

    scores_by = {}
    for name, arch in MODELS:
        print(f"[forest_fig] reading {name}: {arch}/{args.fold}", flush=True)
        scores_by[name] = load_model_scores(spec(arch), chips)
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
