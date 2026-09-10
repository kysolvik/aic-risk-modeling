"""Aggregate every baseline into one comparison table over every val pixel.

Each model is scored on the identical val-pixel population through the existing
metric path (`aic_risk_modeling.eval.eval._binary_metrics`: PR-AUC, F1,
precision, recall, kappa, accuracy). Models enter as one of two prediction
artifacts:

  --tif  NAME=DIR   a directory of `scripts/predict/predict.py` output chips
                    (`out_<x>-<y>.tif` = fire probability, `mask_<x>-<y>.tif` =
                    ground-truth label). Use for the neural baselines + the
                    MTSViT champion.
  --npz  NAME=FILE  an .npz with `scores` and `labels` arrays. Use for the
                    tabular RF baseline (written by scripts/train/baseline_rf.py).

Producing the neural artifacts first (one per config + checkpoint):
    for m in baseline_unet baseline_convlstm baseline_lstm baseline_mlp mtsvit_test_v44; do
      .venv/bin/python scripts/predict/predict.py \
        --config_path configs/$m.json --checkpoint gs://aic-amazon/models/$m.pt \
        --data_dir gs://aic-amazon/data/fullgrid_v2/allpreds_2023/ \
        --output_dir out/baselines/$m/2023/ ; done   # repeat for 2024

Then:
    .venv/bin/python scripts/analysis/compare_baselines.py \
        --tif "MTSViT v44=out/baselines/mtsvit_test_v44" \
        --tif "U-Net=out/baselines/baseline_unet" \
        --tif "ConvLSTM=out/baselines/baseline_convlstm" \
        --tif "LSTM=out/baselines/baseline_lstm" \
        --tif "MLP=out/baselines/baseline_mlp" \
        --npz "Random Forest=out/baselines/rf/val_preds.npz" \
        --out_csv out/baselines/comparison.csv --out_md out/baselines/comparison.md
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aic_risk_modeling.eval.eval import _binary_metrics  # noqa: E402

METRIC_COLS = ["pr_auc", "f1", "precision", "recall", "kappa", "accuracy"]


def _load_tif_dir(directory):
    """Flatten every (out_, mask_) chip pair under `directory` to 1-D arrays.

    Subdirectories are searched too, so one model's several prediction years can
    live side by side under one root.
    """
    import rasterio as rio
    out_paths = sorted(glob.glob(os.path.join(directory, "**", "out_*.tif"),
                                 recursive=True))
    if not out_paths:
        raise FileNotFoundError(f"no out_*.tif chips under {directory}")
    scores, labels = [], []
    for out_path in out_paths:
        mask_path = os.path.join(os.path.dirname(out_path),
                                 os.path.basename(out_path).replace("out_", "mask_", 1))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"missing mask for {out_path}")
        with rio.open(out_path) as s:
            scores.append(s.read(1).ravel().astype(np.float32))
        with rio.open(mask_path) as m:
            labels.append((m.read(1).ravel() > 0))
    return np.concatenate(scores), np.concatenate(labels)


def _load_npz(path):
    d = np.load(path)
    return d["scores"].astype(np.float32).ravel(), (d["labels"].ravel() > 0)


def score_model(scores, labels, threshold):
    m = _binary_metrics(labels, scores >= threshold, scores=scores)
    m["n_pixels"] = int(labels.size)
    m["prevalence"] = float(labels.mean())
    return m


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
    ap.add_argument("--tif", action="append", default=[],
                    help="NAME=DIR of predict.py out_/mask_ chips (repeatable)")
    ap.add_argument("--npz", action="append", default=[],
                    help="NAME=FILE npz with scores+labels (repeatable)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    rows = []
    for name, path in _parse_named(args.tif):
        print(f"[compare] reading tif dir for {name!r}: {path}", flush=True)
        rows.append((name, score_model(*_load_tif_dir(path), args.threshold)))
    for name, path in _parse_named(args.npz):
        print(f"[compare] reading npz for {name!r}: {path}", flush=True)
        rows.append((name, score_model(*_load_npz(path), args.threshold)))
    if not rows:
        raise SystemExit("no models given (use --tif and/or --npz)")

    rows.sort(key=lambda r: r[1]["pr_auc"], reverse=True)

    header = ["model", *METRIC_COLS, "prevalence", "n_pixels"]
    lines = [",".join(header)]
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    for name, m in rows:
        vals = [f"{m[c]:.4f}" for c in METRIC_COLS]
        vals += [f"{m['prevalence']:.4f}", str(m["n_pixels"])]
        lines.append(",".join([name, *vals]))
        md.append("| " + " | ".join([name, *vals]) + " |")

    table = "\n".join(md)
    print("\n" + table + "\n")
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[compare] wrote {args.out_csv}")
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w") as f:
            f.write("# Baseline comparison (all val pixels)\n\n" + table + "\n")
        print(f"[compare] wrote {args.out_md}")


if __name__ == "__main__":
    main()
