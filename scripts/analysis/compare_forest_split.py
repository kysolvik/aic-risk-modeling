"""PR-AUC split by forested vs non-forested land cover.

The pooled comparison (`compare_baselines.py`) scores every val pixel together,
which blends two very different regimes: fire in standing forest
(deforestation / degradation) and fire on already-cleared land (pasture / ag
burns). This script stratifies the identical pixel population by the MapBiomas
forest fraction and reports PR-AUC in each stratum, for the headline models plus
the free climatology / last-year baselines.

The forest fraction is `im_forest_-1` (the PREVIOUS year's MapBiomas fraction --
using the current year would reclassify just-burned pixels as non-forest and
define the stratum by the outcome), materialized per chip by
`export_forest_chips.py` into `<forest_dir>/<year>/forest_<x>-<y>.tif`. Those
chips register pixel-for-pixel with the prediction chips and share the `<x>-<y>`
filename, so evaluation joins them by name.

A pixel is "forest" if its forest fraction >= --threshold (default 0.5).

  PR-AUC is prevalence-dependent, and fire prevalence differs between the two
  strata, so raw PR-AUC is NOT directly comparable forest-vs-non-forest. The
  table therefore also reports `lift = pr_auc / prevalence`, and both figures
  draw the stratum's no-skill floor (its fire prevalence) as a dashed line -- the
  gap above that line is the skill.

Two products:
  * split  -- table (csv+md) and grouped-bar figure at the single --threshold.
  * sweep  -- PR-AUC vs the forest-fraction cutoff (--sweep), a two-panel line
              figure. As the cutoff moves, the stratum population (and its
              prevalence) changes, so the dashed no-skill line moves too.

Usage (2023-24 validation, pooled):
    .venv/bin/python scripts/analysis/compare_forest_split.py \
        --tif "factored_v1=out/baselines/factored_v1" \
        --tif "MTSViT v56=out/baselines/mtsvit_test_v56" \
        --tif "MLP=out/baselines/baseline_mlp" \
        --forest_dir out/forest \
        --label-dir /path/to/label_mosaics \
        --climatology /path/to/climatology_2013_2022.tif \
        --out_dir out/forest_split/val

Usage (2025 holdout -- flat dirs, so pass the year explicitly):
    .venv/bin/python scripts/analysis/compare_forest_split.py \
        --tif "factored_v1=out/preds2025/factored_v1" \
        --tif "MTSViT v56=out/preds2025/mtsvit_test_v56" \
        --tif "MLP=out/preds2025/baseline_mlp_rf9" \
        --forest_dir out/forest --year 2025 \
        --label-dir /path/to/label_mosaics \
        --climatology /path/to/climatology_2013_2022.tif \
        --out_dir out/forest_split/2025
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)  # for pyramid_compare (chip_inventory + palette)

from aic_risk_modeling.eval.eval import _binary_metrics  # noqa: E402
from pyramid_compare import (  # noqa: E402
    chip_inventory, SERIES_COLORS, CONTEXT_COLOR, LASTYEAR_COLOR,
    INK_PRIMARY, INK_SECONDARY, SURFACE, GRID,
)

FOREST_COLOR = "#1baf7a"      # forest stratum (green)
NONFOREST_COLOR = "#eb6834"   # non-forest stratum (orange)
BASELINE_NAMES = {"Last-year burn", "Climatology"}


def _resolve_year(year_from_dir, override):
    if override is not None:
        return str(override)
    if str(year_from_dir).isdigit():
        return str(year_from_dir)
    raise SystemExit(
        f"could not infer a year from chip dir {year_from_dir!r}; pass --year "
        "(the 2025 holdout dirs are flat, with no <year> subdirectory)")


def _forest_chip_path(out_path, forest_dir, year):
    base = os.path.basename(out_path).replace("out_", "forest_", 1)
    return os.path.join(forest_dir, str(year), base)


def _bounds_match(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(tuple(a), tuple(b)))


def _chip_key(path):
    """The '<x>-<y>.tif' id shared by the out_/mask_/forest_ chips of one tile."""
    return os.path.basename(path).split("_", 1)[1]


def canonical_chips(reference_dir, forest_dir, year_override):
    """Ordered chip list from the reference model: key, year, mask + forest path.

    Every model is scored on this one identical, sorted pixel population; the
    labels and forest fraction are read once from here rather than re-read per
    model (they are byte-identical across models).
    """
    chips = []
    for out_path, mask_path, ydir in chip_inventory(reference_dir):
        year = _resolve_year(ydir, year_override)
        fpath = _forest_chip_path(out_path, forest_dir, year)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"missing forest chip {fpath} for {out_path} -- run "
                "export_forest_chips.py for that year first")
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


def load_model_scores(model_dir, chips, year_override):
    """Model scores flat, in the canonical chip order (joined by the <x>-<y> key)."""
    import rasterio as rio
    by_key = {}
    for out_path, mask_path, ydir in chip_inventory(model_dir):
        year = _resolve_year(ydir, year_override)
        by_key[(year, _chip_key(out_path))] = out_path
    missing = [c["key"] for c in chips if c["key"] not in by_key]
    if missing:
        raise FileNotFoundError(
            f"{model_dir} is missing {len(missing)} of {len(chips)} chips, "
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
    chip's bounds -- the same pattern as `pyramid_compare.baseline_levels`.
    """
    import rasterio as rio
    from rasterio.windows import from_bounds
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
            win = from_bounds(*c["bounds"], transform=src.transform).round_offsets().round_lengths()
            sc = src.read(1, window=win).astype(np.float32)
            if sc.shape != c["shape"]:
                raise ValueError(f"baseline window {sc.shape} != chip {c['shape']}")
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
    m = _binary_metrics(labels, scores >= hard_threshold, scores=scores)
    prev = float(labels.mean())
    return dict(pr_auc=float(m["pr_auc"]), prevalence=prev,
                lift=(float(m["pr_auc"]) / prev if prev > 0 else float("nan")),
                f1=float(m["f1"]), n_pixels=int(labels.size))


def _light_metrics(scores, labels):
    """PR-AUC + prevalence + lift only -- the sweep recomputes this many times, so
    it skips the confusion-matrix metrics (F1/kappa) the sweep never reports."""
    from sklearn.metrics import average_precision_score
    if labels.size == 0:
        return dict(pr_auc=float("nan"), prevalence=float("nan"),
                    lift=float("nan"), n_pixels=0)
    prev = float(labels.mean())
    ap = float(average_precision_score(labels, scores)) if labels.any() else float("nan")
    return dict(pr_auc=ap, prevalence=prev,
                lift=(ap / prev if prev > 0 else float("nan")), n_pixels=int(labels.size))


def stratify(scores, labels, forest, cutoff, metric_fn, include_all):
    """Apply `metric_fn(scores, labels)` to the forest / non_forest (/ all) strata."""
    fmask = forest >= cutoff
    out = {"forest": metric_fn(scores[fmask], labels[fmask]),
           "non_forest": metric_fn(scores[~fmask], labels[~fmask])}
    if include_all:
        out["all"] = metric_fn(scores, labels)
    return out


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
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


def write_sweep_csv(order, sweep, cutoffs, out_csv):
    header = ["model", "stratum", "cutoff", "pr_auc", "prevalence", "lift", "n_pixels"]
    lines = [",".join(header)]
    for name in order:
        for stratum in ("forest", "non_forest"):
            for c in cutoffs:
                m = sweep[name][c][stratum]
                lines.append(",".join([
                    name, stratum, f"{c:g}", f"{m['pr_auc']:.4f}",
                    f"{m['prevalence']:.4f}", f"{m['lift']:.3f}", str(m["n_pixels"])]))
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[forest-split] wrote {out_csv}")


def _model_style(order):
    """Colour/linestyle per model: first 3 models get the categorical slots,
    baselines are neutral dashed, any extra models recede to grey."""
    style = {}
    slot = 0
    for name in order:
        if name in BASELINE_NAMES:
            style[name] = (LASTYEAR_COLOR, "--")
        elif slot < len(SERIES_COLORS):
            style[name] = (SERIES_COLORS[slot], "-")
            slot += 1
        else:
            style[name] = (CONTEXT_COLOR, "-")
    return style


def plot_split(order, per_model, prevalence, cutoff, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7.0, 1.35 * len(order) + 2), 5.0),
                           facecolor=SURFACE)
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
    ax.bar(x - w / 2, forest, w, color=FOREST_COLOR, label="Forest", zorder=3)
    ax.bar(x + w / 2, nonf, w, color=NONFOREST_COLOR, label="Non-forest", zorder=3)

    # No-skill floors: each stratum's fire prevalence (identical across models).
    # Labelled via the legend rather than inline text, to avoid overlapping bars.
    ax.axhline(prevalence["forest"], color=FOREST_COLOR, linestyle=":",
               linewidth=1.4, zorder=2, label="Forest no-skill (prevalence)")
    ax.axhline(prevalence["non_forest"], color=NONFOREST_COLOR, linestyle=":",
               linewidth=1.4, zorder=2, label="Non-forest no-skill (prevalence)")

    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9, color=INK_SECONDARY, rotation=20, ha="right")
    ax.set_ylabel("PR-AUC", fontsize=9.5, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    ax.set_ylim(0, max(0.001, max(forest + nonf)) * 1.18)
    ax.set_title(f"{title}\nForest fraction cutoff {cutoff:g}; dotted lines are the "
                 "no-skill floor (stratum fire prevalence)",
                 fontsize=10.5, color=INK_PRIMARY, pad=12, loc="left", linespacing=1.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"[forest-split] wrote {out_png}")


def plot_sweep(order, sweep, prev_curve, cutoffs, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = _model_style(order)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True, facecolor=SURFACE)
    for ax, stratum, sub in zip(
            axes, ("forest", "non_forest"),
            ("Forest pixels (fraction >= cutoff)", "Non-forest pixels (fraction < cutoff)")):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        markers = {"Climatology": "^", "Last-year burn": "s"}
        for name in order:
            y = [sweep[name][c][stratum]["pr_auc"] for c in cutoffs]
            col, ls = style[name]
            ax.plot(cutoffs, y, marker=markers.get(name, "o"),
                    markersize=5.5, color=col, linestyle=ls,
                    linewidth=2.0 if name not in BASELINE_NAMES else 1.5,
                    label=name, zorder=4)
        # no-skill floor per stratum shifts with the cutoff (population changes)
        ax.plot(cutoffs, [prev_curve[c][stratum] for c in cutoffs], color=INK_SECONDARY,
                linestyle=":", linewidth=1.3, label="No-skill (prevalence)", zorder=2)
        ax.set_xlabel("forest-fraction cutoff", fontsize=9.5, color=INK_SECONDARY)
        ax.set_title(sub, fontsize=10.5, color=INK_PRIMARY, pad=10, loc="left")
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    axes[0].set_ylabel("PR-AUC", fontsize=9.5, color=INK_SECONDARY)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.suptitle(title, fontsize=13, color=INK_PRIMARY, x=0.008, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"[forest-split] wrote {out_png}")


def replot_from_csv(out_dir, threshold, title):
    """Redraw both figures from split.csv / sweep.csv already in out_dir."""
    import csv
    split_path = os.path.join(out_dir, "split.csv")
    with open(split_path) as f:
        rows = list(csv.DictReader(f))
    order, per_model, prevalence = [], {}, {}
    for r in rows:
        name, stratum = r["model"], r["stratum"]
        if name not in per_model:
            per_model[name] = {}
            order.append(name)
        per_model[name][stratum] = {k: (int(r[k]) if k == "n_pixels" else float(r[k]))
                                    for k in ("pr_auc", "lift", "prevalence", "f1", "n_pixels")}
        if stratum in ("forest", "non_forest"):
            prevalence[stratum] = float(r["prevalence"])
    plot_split(order, per_model, prevalence, threshold,
               os.path.join(out_dir, "split.png"), title)

    sweep_path = os.path.join(out_dir, "sweep.csv")
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            srows = list(csv.DictReader(f))
        sweep, prev_curve = {}, {}
        for r in srows:
            name, stratum, c = r["model"], r["stratum"], float(r["cutoff"])
            sweep.setdefault(name, {}).setdefault(c, {})[stratum] = {
                "pr_auc": float(r["pr_auc"]), "prevalence": float(r["prevalence"]),
                "lift": float(r["lift"]), "n_pixels": int(r["n_pixels"])}
            prev_curve.setdefault(c, {})[stratum] = float(r["prevalence"])
        cutoffs = sorted(prev_curve)
        plot_sweep(order, sweep, prev_curve, cutoffs,
                   os.path.join(out_dir, "sweep.png"),
                   f"{title} -- vs forest-fraction cutoff")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", action="append", default=[], metavar="NAME=DIR",
                    help="model prediction dir (repeatable). The first is the "
                         "reference chip set for the baselines.")
    ap.add_argument("--forest_dir", default=None,
                    help="root of forest chips: <forest_dir>/<year>/forest_*.tif "
                         "(not needed with --replot)")
    ap.add_argument("--replot", action="store_true",
                    help="redraw split.png/sweep.png from the CSVs already in "
                         "--out_dir, without re-reading any chips")
    ap.add_argument("--year", default=None,
                    help="force the year (needed for the flat 2025 holdout dirs)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="forest-fraction cutoff for the split table+bars")
    ap.add_argument("--sweep", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                    help="comma list of cutoffs for the sweep plot ('' to skip)")
    ap.add_argument("--hard_threshold", type=float, default=0.5,
                    help="score cutoff for the F1/precision/recall columns "
                         "(PR-AUC ignores it)")
    ap.add_argument("--label-dir", dest="label_dir", default=None,
                    help="dir of label_<year>.tif full-basin mosaics (enables the "
                         "last-year baseline)")
    ap.add_argument("--climatology", default=None,
                    help="climatology raster (enables the climatology baseline)")
    ap.add_argument("--out_dir", required=True,
                    help="writes split.{csv,md,png} and sweep.{csv,png} here")
    ap.add_argument("--title", default="PR-AUC by land cover")
    args = ap.parse_args()

    if args.replot:
        replot_from_csv(args.out_dir, args.threshold, args.title)
        return

    if not args.forest_dir:
        raise SystemExit("--forest_dir is required (except with --replot)")
    entries = []
    for item in args.tif:
        if "=" not in item:
            raise SystemExit(f"--tif needs NAME=DIR, got {item!r}")
        name, path = item.split("=", 1)
        entries.append((name.strip(), path.strip()))
    if not entries:
        raise SystemExit("need at least one --tif")

    reference_dir = entries[0][1]
    # One canonical, sorted chip population; labels + forest read once.
    chips = canonical_chips(reference_dir, args.forest_dir, args.year)
    labels, forest = load_shared(chips)
    years = sorted({c["year"] for c in chips})
    print(f"[forest-split] {len(chips)} chips over year(s) {','.join(years)}, "
          f"{labels.size} pixels; reading model scores", flush=True)

    order = []
    scores_by = {}  # name -> flat scores aligned to (labels, forest)
    for name, path in entries:
        print(f"[forest-split] reading {name}: {path}", flush=True)
        scores_by[name] = load_model_scores(path, chips, args.year)
        order.append(name)

    for name, kind in (("Last-year burn", "last_year"), ("Climatology", "climatology")):
        needs = args.label_dir if kind == "last_year" else args.climatology
        if not needs:
            print(f"[forest-split] skipping {name} (no "
                  f"{'--label-dir' if kind == 'last_year' else '--climatology'})")
            continue
        try:
            print(f"[forest-split] reading baseline {name}", flush=True)
            scores_by[name] = load_baseline_scores(chips, kind, args.label_dir,
                                                   args.climatology)
            order.append(name)
        except FileNotFoundError as e:
            print(f"[forest-split] skipping {name}: missing {e}")

    def _prev(mask):
        return float(labels[mask].mean()) if mask.any() else float("nan")

    prevalence = {"forest": _prev(forest >= args.threshold),
                  "non_forest": _prev(forest < args.threshold)}

    full = lambda s, l: _full_metrics(s, l, args.hard_threshold)  # noqa: E731
    per_model = {name: stratify(scores_by[name], labels, forest, args.threshold,
                                full, include_all=True) for name in order}
    os.makedirs(args.out_dir, exist_ok=True)
    write_split_table(order, per_model, args.threshold,
                      os.path.join(args.out_dir, "split.csv"),
                      os.path.join(args.out_dir, "split.md"))
    plot_split(order, per_model, prevalence, args.threshold,
               os.path.join(args.out_dir, "split.png"), args.title)

    cutoffs = [float(c) for c in args.sweep.split(",") if c.strip()]
    if len(cutoffs) >= 2:
        sweep = {name: {c: stratify(scores_by[name], labels, forest, c,
                                    _light_metrics, include_all=False)
                        for c in cutoffs} for name in order}
        prev_curve = {c: {"forest": _prev(forest >= c),
                          "non_forest": _prev(forest < c)} for c in cutoffs}
        write_sweep_csv(order, sweep, cutoffs, os.path.join(args.out_dir, "sweep.csv"))
        plot_sweep(order, sweep, prev_curve, cutoffs,
                   os.path.join(args.out_dir, "sweep.png"),
                   f"{args.title} -- vs forest-fraction cutoff")
    else:
        print("[forest-split] sweep skipped (need >=2 cutoffs)")


if __name__ == "__main__":
    main()
