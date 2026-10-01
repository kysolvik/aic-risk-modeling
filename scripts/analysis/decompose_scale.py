"""Split a model's pooled PR-AUC into chip-level intensity vs within-chip localisation.

Pooled PR-AUC over every val pixel -- the number `compare_baselines.py` reports --
is dominated by getting each 71 km chip's overall burn RATE right, not by ranking
pixels inside it. Measured on the 2026-09-03 suite, an oracle that knows each
chip's true burn rate and predicts it FLAT inside the chip already scores 0.3145,
i.e. 88% of the best model's 0.3566. That means a single-number bench cannot tell
"the model localises better" from "the model got chip intensity luckier", and the
two are improved by completely different parts of an architecture.

Columns:
  pr_auc              as scored today (the compare_baselines number)
  oracle_chip_only    true chip burn rate, flat inside each chip. Chip-level
                      information ALONE, no within-chip ranking at all.
  within_chip_only    predictions divided by their own chip mean, which strips the
                      model's chip-level signal and leaves pure localisation.
  oracle_chip_rescale predictions rescaled so each chip's mean matches truth. The
                      ceiling if the chip-intensity term were perfect -- headroom
                      for an explicit intensity head.
  chip_r              corr(model's chip mean, true chip rate). How good the model
                      already is at the chip-intensity job.

Only --tif input is supported: the decomposition needs per-chip structure, which a
flattened .npz (the RF artifact) does not carry.

Usage:
    .venv/bin/python scripts/analysis/decompose_scale.py \
        --tif "MLP=out/baselines/baseline_mlp" \
        --tif "MTSViT v44=out/baselines/mtsvit_test_v44" \
        --out_csv out/baselines/scale_decomposition.csv
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aic_risk_modeling.eval.eval import _binary_metrics  # noqa: E402

COLS = ["pr_auc", "oracle_chip_only", "within_chip_only", "oracle_chip_rescale",
        "oracle_within_chip", "chip_r", "prevalence", "n_chips", "n_pixels"]


def _pr_auc(labels, scores):
    return float(_binary_metrics(labels, scores >= 0.5, scores=scores)["pr_auc"])


def load_chips(directory, stride=1):
    """(scores, labels, chip_id) over every (out_, mask_) pair under `directory`.

    `stride` subsamples each chip on a regular grid. Chip identity is preserved,
    so the decomposition is unaffected; it just makes the sweep cheaper.
    """
    import rasterio as rio
    out_paths = sorted(glob.glob(os.path.join(directory, "**", "out_*.tif"), recursive=True))
    if not out_paths:
        raise FileNotFoundError(f"no out_*.tif chips under {directory}")
    scores, labels, chip = [], [], []
    for i, out_path in enumerate(out_paths):
        mask_path = os.path.join(os.path.dirname(out_path),
                                 os.path.basename(out_path).replace("out_", "mask_", 1))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"missing mask for {out_path}")
        with rio.open(out_path) as s:
            p = s.read(1)[::stride, ::stride].astype(np.float32)
        with rio.open(mask_path) as m:
            y = (m.read(1)[::stride, ::stride] > 0)
        scores.append(p.ravel())
        labels.append(y.ravel())
        chip.append(np.full(y.size, i, dtype=np.int32))
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(chip)


def decompose(scores, labels, chip):
    n_chips = int(chip.max()) + 1
    count = np.bincount(chip, minlength=n_chips).astype(np.float64)
    true_rate = np.bincount(chip, weights=labels, minlength=n_chips) / count
    pred_rate = np.maximum(np.bincount(chip, weights=scores, minlength=n_chips) / count, 1e-9)

    rescaled = np.clip(scores * (true_rate / pred_rate)[chip], 0.0, 1.0)

    # The complement of oracle_chip_rescale: PERFECT within-chip ranking at the
    # model's OWN chip intensity. This is the ceiling for a better-localising
    # architecture, and it is what oracle_chip_rescale does not tell you -- that
    # column holds localisation fixed and perfects the chip scalar; this one holds
    # the chip scalar fixed and perfects localisation.
    #
    # Built as a WITHIN-CHIP PERMUTATION of the model's own scores: inside each
    # chip the score multiset is untouched and simply re-assigned so the largest
    # values land on the positives. Chip mean, chip variance and the model's
    # within-chip dynamic range are therefore all preserved exactly, and only the
    # ordering is perfected.
    #
    # Do NOT instead nudge positives above negatives by a tiny epsilon. That
    # collapses within-chip dynamic range to ~0, so the global ranking degenerates
    # to chip order alone and pooled AP FALLS (measured: MLP 0.3566 -> 0.2332).
    # The result is an artifact of the construction, not a statement about
    # localisation.
    # `labels` arrives as a bool array; unary minus is undefined on numpy bools, so
    # cast before negating to get descending (positives-first) order.
    by_score = np.lexsort((-scores, chip))                    # per chip, score desc
    by_label = np.lexsort((-labels.astype(np.int8), chip))    # per chip, positives first
    oracle_within = np.empty_like(scores)
    oracle_within[by_label] = scores[by_score]

    return {
        "pr_auc": _pr_auc(labels, scores),
        "oracle_chip_only": _pr_auc(labels, true_rate[chip]),
        "within_chip_only": _pr_auc(labels, scores / pred_rate[chip]),
        "oracle_chip_rescale": _pr_auc(labels, rescaled),
        "oracle_within_chip": _pr_auc(labels, oracle_within),
        "chip_r": float(np.corrcoef(pred_rate, true_rate)[0, 1]),
        "prevalence": float(labels.mean()),
        "n_chips": n_chips,
        "n_pixels": int(labels.size),
    }


def _parse_named(items):
    out = []
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", action="append", metavar="NAME=DIR", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="subsample each chip by this factor (2 is ~4x faster, same conclusions)")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    rows = []
    for name, directory in _parse_named(args.tif):
        scores, labels, chip = load_chips(directory, stride=args.stride)
        row = {"model": name, **decompose(scores, labels, chip)}
        rows.append(row)
        print(f"{name}: pr_auc={row['pr_auc']:.4f} "
              f"oracle_chip_only={row['oracle_chip_only']:.4f} "
              f"within_chip={row['within_chip_only']:.4f} "
              f"rescale={row['oracle_chip_rescale']:.4f} chip_r={row['chip_r']:.3f}")

    rows.sort(key=lambda r: -r["pr_auc"])
    header = ["model"] + COLS
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(fmt(r[c]) for c in header) + "\n")
        print(f"wrote {args.out_csv}")
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w") as f:
            f.write("# Scale decomposition (all val pixels)\n\n")
            f.write("| " + " | ".join(header) + " |\n")
            f.write("|" + "---|" * len(header) + "\n")
            for r in rows:
                f.write("| " + " | ".join(fmt(r[c]) for c in header) + " |\n")
        print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
