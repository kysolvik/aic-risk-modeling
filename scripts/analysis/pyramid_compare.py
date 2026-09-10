"""Compare several models' accuracy across a pyramid of spatial scales.

The single-model version of this lives in the eval module
(`aic_risk_modeling.eval.pyramid_pool_stats`), which max-pools one model's
prediction and label field at 1x1, 2x2, 4x4 ... and reports PR-AUC / F1 per
level. This script does the same coarsening for MANY models on one shared chip
population and draws them on one pair of axes, so the question becomes "at what
spatial scale do these models separate from each other, and from a free
baseline?" rather than "how does one model behave at scale".

Chips are 128x128 at 0.005 deg, so the pyramid spans ~0.55 km (1 px) to ~71 km
(128 px, the whole chip). Pooling is done WITHIN a chip and never across chips:
chips are scattered tiles, so a block spanning two of them would be meaningless.

Two pooling rules, one per panel, because the two metrics ask different questions:

  PR-AUC / mean-pooled   block score = MEAN predicted probability = the expected
                         burned fraction of the block. Mean-pooling keeps the
                         score discriminative as blocks grow; max-pooling
                         saturates toward 1 and throws ranking information away.
                         This is the "rank neighbourhoods by fire risk" question.

  F1 / max-pooled        block score = MAX predicted probability. This is the
                         detection question -- "did the model raise an alarm
                         anywhere in this neighbourhood" -- and matches the
                         semantics of `pyramid_pool_stats`. By default the
                         threshold is SWEPT and the best F1 reported
                         (`--f1-mode best`), because the models here are trained
                         with pos_weight 10 while the baselines are on a natural
                         scale: at a fixed 0.5 the comparison measures
                         calibration rather than ranking, and climatology in
                         particular is penalised for recall it never had a
                         chance at. `--f1-mode threshold` restores F1@--threshold.

Both panels use the same label rule: a block is positive if ANY pixel in it
burned (max-pool of the 0/1 label). Both `--prauc-pool` and `--f1-pool` are
overridable if you want to see the other convention.

IMPORTANT -- read the curves ACROSS MODELS at a fixed scale, not along the x
axis. Block prevalence rises with block size (at 128 px nearly every chip
contains some fire), and PR-AUC rises mechanically with prevalence, so a curve
sloping up does not by itself mean the model is "better at coarse scale". The
`prevalence` and `lift` (pr_auc / prevalence) columns in the CSV are there to
make that explicit; `lift` is the scale-comparable number.

Two free baselines are drawn as dashed neutral lines:

  last-year burn   the previous year's burned mask, as a prediction of this
                   year. Persistence. Binary at 1 px, but becomes a real
                   continuous score once pooled (the block's burned fraction).
  climatology      per-pixel mean burn frequency over --clim-years (2013-2022 by
                   default), which excludes every evaluation year here, so there
                   is no leakage.

Both are read by windowed reads out of full-basin label mosaics
(`gs://aic-amazon/preds/mtsvit_v44_<year>/preds_mask.tif`), which were verified
pixel-identical to the per-chip `mask_` rasters before this script was written.

Usage:
    .venv/bin/python scripts/analysis/pyramid_compare.py \
        --tif "Factored v1=out/baselines/factored_v1" \
        --tif "MTSViT v56=out/baselines/mtsvit_test_v56" \
        --tif "MLP rf9=out/baselines/baseline_mlp" \
        --label-dir /path/to/label_mosaics \
        --out_csv out/pyramid/pyramid_val.csv \
        --out_png out/pyramid/pyramid_val.png
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aic_risk_modeling.eval.eval import _binary_metrics, _block_max_pool  # noqa: E402

DEFAULT_BLOCKS = [1, 2, 4, 8, 16, 32, 64, 128]
KM_PER_PIXEL = 0.005 * 111.32  # 0.005 deg at the equator, ~0.557 km

# Categorical slots 1-3 of the reference palette. Validated all-pairs, light
# mode: worst CVD dE 9.2, worst normal-vision dE 24.0. Only three, because a
# line chart's series all overlap (the all-pairs case) and slots 4+ fail there.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
CONTEXT_COLOR = "#b6b4ab"   # extra models, de-emphasised
LASTYEAR_COLOR = "#52514e"  # neutral ink -- persistence is a reference, not a rival
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e4e3de"


def _block_mean_pool(arr, block):
    """Non-overlapping ``block`` x ``block`` MEAN-pool of a 2-D array.

    Unlike `_block_max_pool`, partial edge blocks cannot be zero-padded without
    biasing the mean downward, so this requires H and W to be exact multiples of
    ``block``. Chips are 128x128 and the blocks are powers of two, so that holds
    here; the check exists to fail loudly if a differently-shaped chip appears.
    """
    block = int(block)
    if block <= 1:
        return np.asarray(arr, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    height, width = arr.shape
    if height % block or width % block:
        raise ValueError(f"mean-pool needs H,W divisible by {block}; got {arr.shape}")
    return arr.reshape(height // block, block,
                       width // block, block).mean(axis=(1, 3))


def _best_f1(labels, scores):
    """Best achievable F1 over all thresholds, plus the P/R/threshold there.

    Models here are trained with weighted BCE (pos_weight 10), so their
    probabilities are inflated relative to a natural-scale baseline like
    climatology. Scoring every series at one fixed threshold would compare
    calibration, not ranking -- climatology loses on recall simply because its
    values rarely exceed 0.5. Sweeping the threshold removes that confound and
    asks what each score field could achieve at its own best operating point.
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(labels, scores)
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1), 0.0)
    i = int(np.argmax(f1))
    # precision_recall_curve returns len(thr) == len(prec) - 1
    t = float(thr[i]) if i < len(thr) else float("inf")
    return float(f1[i]), float(prec[i]), float(rec[i]), t


def _best_kappa(labels, scores):
    """Best achievable Cohen's kappa over all thresholds, plus that threshold.

    Kappa is threshold-dependent (unlike PR-AUC / ROC-AUC), so -- exactly as for
    `_best_f1` -- the threshold is swept rather than fixed at 0.5, otherwise the
    weighted-BCE models (inflated probabilities) and the natural-scale baselines
    would be compared on calibration, not agreement. Vectorised over the sorted
    scores: for every distinct cut, predict-positive = {score >= cut} and read
    kappa = (p_o - p_e) / (1 - p_e) off the running confusion counts.
    """
    y = (np.asarray(labels) > 0).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n = y.size
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0, float("inf")  # kappa undefined with one class; no skill = 0
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    tp = np.cumsum(y[order]).astype(np.float64)   # positives among the top k
    k = np.arange(1, n + 1, dtype=np.float64)      # predicted-positive count
    tn = n_neg - (k - tp)
    p_o = (tp + tn) / n
    p_e = (k / n) * (n_pos / n) + ((n - k) / n) * (n_neg / n)
    denom = 1.0 - p_e
    kappa = np.where(denom > 0, (p_o - p_e) / np.where(denom > 0, denom, 1.0), 0.0)
    # Only cuts at a distinct-score boundary are real operating points (ties must
    # move together); predicting the all-negative side always scores kappa 0.
    boundary = np.ones(n, dtype=bool)
    boundary[:-1] = s_sorted[1:] != s_sorted[:-1]
    kappa_at_cut = np.where(boundary, kappa, -np.inf)
    i = int(np.argmax(kappa_at_cut))
    best = float(kappa[i])
    if best <= 0.0:
        return 0.0, float("inf")
    return best, float(s_sorted[i])


def _pool(arr, block, how):
    return _block_max_pool(arr, block) if how == "max" else _block_mean_pool(arr, block)


def chip_inventory(directory):
    """[(out_path, mask_path, year)] for every chip pair under `directory`.

    The year is the basename of the chip's parent directory, matching the
    layout the predict runbook writes (``<model>/<year>/out_*.tif``).

    Only the two documented layouts are accepted: chips directly in `directory`
    (flat holdout) or one level down in a per-year subdirectory. Anything deeper
    (e.g. a stray ``<model>/<year>/chips/out_*.tif``) is ignored, so an unrelated
    nested export can't leak into a validation figure.
    """
    root = os.path.normpath(directory)
    out_paths = sorted(
        p for p in glob.glob(os.path.join(directory, "**", "out_*.tif"),
                             recursive=True)
        if os.path.dirname(os.path.normpath(p)) == root
        or os.path.dirname(os.path.dirname(os.path.normpath(p))) == root)
    if not out_paths:
        raise FileNotFoundError(f"no out_*.tif chips under {directory}")
    items = []
    for out_path in out_paths:
        parent = os.path.dirname(out_path)
        mask_path = os.path.join(parent,
                                 os.path.basename(out_path).replace("out_", "mask_", 1))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"missing mask for {out_path}")
        items.append((out_path, mask_path, os.path.basename(parent)))
    return items


def _levels_from_chips(score_iter, blocks, prauc_pool, f1_pool, threshold,
                       f1_mode):
    """Per-block metrics from an iterator of (score_2d, label_2d) chip pairs."""
    acc = {b: {"pa_s": [], "f1_s": [], "y": []} for b in blocks}
    for score, label in score_iter:
        lab_f = (np.asarray(label) > 0).astype(np.float32)
        for b in blocks:
            acc[b]["pa_s"].append(_pool(score, b, prauc_pool).ravel())
            acc[b]["f1_s"].append(_pool(score, b, f1_pool).ravel())
            acc[b]["y"].append((_block_max_pool(lab_f, b) > 0).ravel())
    levels = []
    for b in blocks:
        y = np.concatenate(acc[b]["y"])
        pa_s = np.concatenate(acc[b]["pa_s"])
        f1_s = np.concatenate(acc[b]["f1_s"])
        prev = float(y.mean())
        pa = _binary_metrics(y, pa_s > threshold, scores=pa_s)
        try:
            from sklearn.metrics import roc_auc_score
            roc = float(roc_auc_score(y, pa_s)) if 0.0 < prev < 1.0 else float("nan")
        except Exception:
            roc = float("nan")
        if f1_mode == "best":
            f1_val, f1_p, f1_r, f1_thr = _best_f1(y, f1_s)
        else:
            m = _binary_metrics(y, f1_s > threshold, scores=f1_s)
            f1_val, f1_p, f1_r, f1_thr = (float(m["f1"]), float(m["precision"]),
                                          float(m["recall"]), threshold)
        # Best-kappa on the SAME mean-pooled score PR-AUC/ROC-AUC use, so the
        # figure keeps one score convention across panels; threshold swept for
        # the same calibration reason as best-F1.
        kappa_val, kappa_thr = _best_kappa(y, pa_s)
        levels.append({
            "block": b, "km": round(b * KM_PER_PIXEL, 3), "n_blocks": int(y.size),
            "prevalence": prev,
            "pr_auc": float(pa["pr_auc"]),
            "roc_auc": roc,
            "kappa": kappa_val, "kappa_threshold": kappa_thr,
            "lift": float(pa["pr_auc"]) / prev if prev > 0 else float("nan"),
            "f1": f1_val, "precision": f1_p, "recall": f1_r,
            "f1_threshold": f1_thr,
        })
        acc[b] = None  # release
    return levels


def model_levels(directory, blocks, prauc_pool, f1_pool, threshold, f1_mode):
    import rasterio as rio

    def gen():
        for out_path, mask_path, _ in chip_inventory(directory):
            with rio.open(out_path) as s:
                score = s.read(1).astype(np.float32)
            with rio.open(mask_path) as m:
                yield score, m.read(1)
    return _levels_from_chips(gen(), blocks, prauc_pool, f1_pool, threshold,
                              f1_mode)


def baseline_levels(reference_dir, kind, label_dir, clim_path, blocks,
                    prauc_pool, f1_pool, threshold, f1_mode):
    """Levels for a free baseline scored on the same chips as the models.

    `kind` is "last_year" (previous year's burn mask) or "climatology" (the
    prebuilt mean-burn-frequency raster). Both come from full-basin mosaics read
    through a window matching each chip's bounds.
    """
    import rasterio as rio
    from rasterio.windows import from_bounds

    handles = {}

    def _src(path):
        if path not in handles:
            handles[path] = rio.open(path)
        return handles[path]

    def gen():
        for out_path, mask_path, year in chip_inventory(reference_dir):
            with rio.open(mask_path) as m:
                label = m.read(1)
                bounds = m.bounds
            if kind == "climatology":
                src = _src(clim_path)
            else:
                prev = int(year) - 1
                src = _src(os.path.join(label_dir, f"label_{prev}.tif"))
            win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            score = src.read(1, window=win).astype(np.float32)
            if score.shape != label.shape:
                raise ValueError(f"baseline window {score.shape} != chip {label.shape} "
                                 f"for {out_path}")
            if kind == "last_year":
                score = (score > 0).astype(np.float32)
            yield score, label
    try:
        return _levels_from_chips(gen(), blocks, prauc_pool, f1_pool, threshold,
                                  f1_mode)
    finally:
        for h in handles.values():
            h.close()


def write_csv(rows, path):
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = ["model", "block", "km", "n_blocks", "prevalence", "pr_auc", "roc_auc",
            "kappa", "kappa_threshold", "lift", "f1", "precision", "recall",
            "f1_threshold"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"[pyramid] wrote {path}")


def plot(series, blocks, path, title, prauc_pool, f1_pool, threshold, f1_mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), facecolor=SURFACE)
    panels = [
        ("pr_auc", f"PR-AUC  ({prauc_pool}-pooled score)",
         "Ranking: are the right neighbourhoods flagged?"),
        ("f1", (f"Best F1  ({f1_pool}-pooled score, threshold swept)" if f1_mode == "best"
                else f"F1  ({f1_pool}-pooled score, threshold {threshold:g})"),
         "Detection: is an alarm raised in the right neighbourhood?"),
    ]
    xs = np.arange(len(blocks))
    for ax, (metric, heading, sub) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        for s in series:
            y = [lvl[metric] for lvl in s["levels"]]
            ax.plot(xs, y, marker=s["marker"], color=s["color"],
                    linewidth=2.0 if s["kind"] != "context" else 1.2,
                    linestyle=s["linestyle"], markersize=6.5,
                    label=s["name"], zorder=s["z"],
                    alpha=1.0 if s["kind"] != "context" else 0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b}\n{b * KM_PER_PIXEL:.3g} km" for b in blocks],
                           fontsize=8.5, color=INK_SECONDARY)
        ax.set_xlabel("block size  (pixels per side / kilometres)", fontsize=9.5,
                      color=INK_SECONDARY)
        ax.set_ylabel(metric.replace("pr_auc", "PR-AUC").replace("f1", "F1"),
                      fontsize=9.5, color=INK_SECONDARY)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
        ax.set_title(f"{heading}\n{sub}", fontsize=10.5, color=INK_PRIMARY,
                     pad=12, loc="left", linespacing=1.5)
        ax.set_xlim(-0.35, len(blocks) - 1 + 0.35)
        ax.set_ylim(0, 1)

    axes[0].legend(loc="lower right", frameon=False, fontsize=8.5,
                   labelcolor=INK_SECONDARY)
    fig.suptitle(title, fontsize=13, color=INK_PRIMARY, x=0.008, ha="left",
                 y=0.995)
    fig.text(0.008, 0.008,
             "Blocks pooled within chips. A block is positive if any pixel in it burned, "
             "so prevalence rises with block size and\nPR-AUC rises with it — compare models "
             "at a fixed scale, not along the x axis (see the `lift` column in the CSV).",
             fontsize=7.8, color=INK_SECONDARY, va="bottom")
    fig.tight_layout(rect=[0, 0.055, 1, 0.955])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"[pyramid] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", action="append", default=[], metavar="NAME=DIR",
                    help="model prediction dir; repeatable. First 3 get colour, "
                         "the rest are drawn as de-emphasised context lines.")
    ap.add_argument("--label-dir", required=True,
                    help="directory of label_<year>.tif full-basin mosaics")
    ap.add_argument("--climatology", default=None,
                    help="path to the climatology raster (skip the baseline if unset)")
    ap.add_argument("--blocks", default=None,
                    help="comma-separated block sides (default 1,2,...,128)")
    ap.add_argument("--prauc-pool", choices=["mean", "max"], default="mean")
    ap.add_argument("--f1-pool", choices=["mean", "max"], default="max")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--f1-mode", choices=["best", "threshold"], default="best",
                    help="'best' sweeps the threshold for max F1 (removes the "
                         "calibration confound between weighted-BCE models and "
                         "natural-scale baselines); 'threshold' uses --threshold")
    ap.add_argument("--title", default="Accuracy vs spatial scale")
    ap.add_argument("--from_csv", default=None,
                    help="re-plot from an existing --out_csv instead of "
                         "recomputing (the metrics take ~30 min; the figure "
                         "should not)")
    ap.add_argument("--color", action="append", default=[],
                    help="model/baseline name to give a colour slot; repeatable, "
                         "max 3. Everything else recedes to grey. Defaults to the "
                         "first two --tif plus Climatology.")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    blocks = ([int(b) for b in args.blocks.split(",") if b.strip()]
              if args.blocks else DEFAULT_BLOCKS)

    entries = []
    for item in args.tif:
        if "=" not in item:
            raise SystemExit(f"--tif needs NAME=DIR, got {item!r}")
        name, path = item.split("=", 1)
        if not name.strip():
            raise SystemExit(f"--tif has an empty name: {item!r}")
        entries.append((name.strip(), path))
    if not entries and not args.from_csv:
        raise SystemExit("need at least one --tif (or --from_csv to re-plot)")

    if args.from_csv:
        import csv as _csv
        rows = []
        for r in _csv.DictReader(open(args.from_csv)):
            rows.append({"model": r["model"],
                         **{k: (int(r[k]) if k in ("block", "n_blocks") else float(r[k]))
                            for k in ("block", "km", "n_blocks", "prevalence",
                                      "pr_auc", "lift", "f1", "precision", "recall")}})
        order, by = [], {}
        for r in rows:
            by.setdefault(r["model"], []).append(r)
            if r["model"] not in order:
                order.append(r["model"])
        blocks = sorted({r["block"] for r in rows})
        wanted = args.color or [n for n in order if n != "Last-year burn"][:2] + ["Climatology"]
        series = []
        for name in order:
            levels = sorted(by[name], key=lambda r: r["block"])
            base = name in ("Last-year burn", "Climatology")
            if name in wanted:
                col, kind = SERIES_COLORS[wanted.index(name)], "model"
            elif base:
                col, kind = LASTYEAR_COLOR, "baseline"
            else:
                col, kind = CONTEXT_COLOR, "context"
            series.append({"name": name, "levels": levels, "color": col,
                           "linestyle": "--" if base else "-",
                           "marker": ("^" if name == "Climatology" else "s") if base
                                     else ("o" if kind == "model" else ""),
                           "kind": "context" if kind == "context" else kind,
                           "z": 5 if name in wanted else (4 if base else 2)})
        plot(series, blocks, args.out_png, args.title, args.prauc_pool,
             args.f1_pool, args.threshold, args.f1_mode)
        return

    series, rows = [], []
    for i, (name, path) in enumerate(entries):
        print(f"[pyramid] {name}: {path}", flush=True)
        levels = model_levels(path, blocks, args.prauc_pool, args.f1_pool,
                              args.threshold, args.f1_mode)
        for lvl in levels:
            rows.append({"model": name, **lvl})
        # Only the first two models get a colour; climatology takes the third
        # slot. The rest recede to grey -- on this figure the top models overlap
        # almost exactly, so drawing them all in colour hides the comparison
        # that matters (models vs a free baseline) behind a bundle of lines.
        colored = i < len(SERIES_COLORS) - 1
        series.append({"name": name, "levels": levels,
                       "color": SERIES_COLORS[i] if colored else CONTEXT_COLOR,
                       "linestyle": "-", "marker": "o" if colored else "",
                       "kind": "model" if colored else "context",
                       "z": 6 - i if colored else 2})

    ref = entries[0][1]
    baselines = [("Last-year burn", "last_year")]
    if args.climatology:
        baselines.append((f"Climatology", "climatology"))
    for name, kind in baselines:
        print(f"[pyramid] baseline {name}", flush=True)
        levels = baseline_levels(ref, kind, args.label_dir, args.climatology,
                                 blocks, args.prauc_pool, args.f1_pool,
                                 args.threshold, args.f1_mode)
        for lvl in levels:
            rows.append({"model": name, **lvl})
        is_clim = kind == "climatology"
        series.append({"name": name, "levels": levels,
                       "color": SERIES_COLORS[2] if is_clim else LASTYEAR_COLOR,
                       "linestyle": "--",
                       "marker": "^" if is_clim else "s", "kind": "baseline",
                       "z": 5 if is_clim else 4})

    write_csv(rows, args.out_csv)
    plot(series, blocks, args.out_png, args.title, args.prauc_pool,
         args.f1_pool, args.threshold, args.f1_mode)


if __name__ == "__main__":
    main()
