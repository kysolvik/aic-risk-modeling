"""Protocol-correct summary of training runs, straight from their per-epoch CSVs.

Available the moment a job finishes -- no prediction chips needed -- so it is the
first thing to run on a completed job. It reports the statistics the measurement
protocol calls for, not just `best val_pr_auc`:

  best / @ep    the usual headline, and the epoch it happened. Max over a noisy
                series is upward-biased: v44's last-7-epoch range is 0.0262, larger
                than the 0.0222 gap it was once used to explain.
  last3         mean of the final three epochs -- the less biased statistic, and the
                one to compare across runs.
  TRUNC         best val_pr_auc landed on the FINAL epoch, i.e. the run was still
                improving when it died. v44 and v44_retrain both did this
                (early_stopping_metric: "loss"), which makes any comparison against
                a converged baseline unconverged-vs-converged.
  slope         val_pr_auc trend over the last 3 epochs; clearly positive means
                still climbing.
  gap           train - val pr_auc at the final epoch. Large = overfitting, ~0 or
                negative = underfitting / still has room.

Usage:
    .venv/bin/python scripts/analysis/run_status.py                      # default set
    .venv/bin/python scripts/analysis/run_status.py --runs factored_v1 baseline_mlp_rf1
    .venv/bin/python scripts/analysis/run_status.py --local out          # local CSVs
"""

import argparse
import os
import sys

import numpy as np

REMOTE = "gs://aic-amazon/models"
DEFAULT = ["baseline_mlp", "baseline_convlstm", "baseline_unet_full", "baseline_unet",
           "baseline_lstm", "mtsvit_test_v44", "mtsvit_test_v44_retrain",
           "mtsvit_test_v49", "mtsvit_test_v50", "mtsvit_test_v51", "mtsvit_test_v52",
           "mtsvit_test_v53", "mtsvit_test_v54", "mtsvit_test_v55",
           "mtsvit_test_v56", "mtsvit_test_v57", "mtsvit_test_v44b",
           "baseline_mlp_rf1", "baseline_mlp_rf9", "baseline_mlp_rf17", "factored_v1"]


# `trainer.run` writes the CSV to splitext(model_output_path)[0] + '.csv', so a
# re-run with the same model_output_path silently overwrites both the checkpoint and
# the curve. That already happened once: gs://aic-amazon/models/mtsvit_test_v44.csv is
# the RETRAIN (best 0.3235), while the original v44 (best 0.3342) survives only as
# out/mtsvit_test_v44.csv. Hence --prefer_local: read the local copy when it exists.
OVERWRITTEN_IN_GCS = {"mtsvit_test_v44": "gs:// copy is the retrain; original is local only"}


def read_csv(name, local=None, prefer_local=None):
    """Per-epoch rows for one run, from a local dir or the models bucket."""
    import csv as _csv
    for directory in ([local] if local else []) + ([prefer_local] if prefer_local else []):
        path = os.path.join(directory, f"{name}.csv")
        if os.path.exists(path):
            with open(path) as f:
                return list(_csv.DictReader(f))
    if local:
        return None
    from tensorflow.io import gfile
    path = f"{REMOTE}/{name}.csv"
    if not gfile.exists(path):
        return None
    with gfile.GFile(path, "r") as f:
        return list(_csv.DictReader(f))


def summarize(rows):
    v = np.array([float(r["val_pr_auc"]) for r in rows])
    t = np.array([float(r["pr_auc"]) for r in rows])
    n = len(v)
    best_i = int(v.argmax())
    last3 = v[-3:] if n >= 3 else v
    slope = float(np.polyfit(np.arange(len(last3)), last3, 1)[0]) if len(last3) > 1 else 0.0
    return {
        "n": n,
        "best": float(v[best_i]),
        "best_ep": best_i,
        "truncated": best_i == n - 1,
        "last3": float(last3.mean()),
        "slope": slope,
        "gap": float(t[-1] - v[-1]),
        "range7": float(v[-7:].max() - v[-7:].min()) if n >= 7 else float(v.max() - v.min()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=DEFAULT)
    ap.add_argument("--local", default=None, help="read <dir>/<name>.csv instead of GCS")
    ap.add_argument("--prefer_local", default="out",
                    help="check this dir first, then GCS (default 'out'; guards against "
                         "runs whose GCS CSV was overwritten by a re-run)")
    ap.add_argument("--sort", default="last3", choices=["last3", "best", "name"])
    args = ap.parse_args()

    got, missing = [], []
    for name in args.runs:
        rows = read_csv(name, args.local, args.prefer_local)
        if not rows:
            missing.append(name)
            continue
        got.append((name, summarize(rows)))

    if args.sort != "name":
        got.sort(key=lambda kv: -kv[1][args.sort])

    print(f"{'run':<26}{'eps':>4}{'best':>8}{'@ep':>5}{'last3':>8}"
          f"{'slope':>9}{'gap':>8}  flags")
    for name, s in got:
        flags = []
        if s["truncated"]:
            flags.append("TRUNC(peak=final)")
        if s["slope"] > 0.002:
            flags.append("still-rising")
        if s["gap"] > 0.04:
            flags.append("overfit")
        elif s["gap"] < 0.005:
            flags.append("underfit")
        print(f"{name:<26}{s['n']:>4}{s['best']:>8.4f}{s['best_ep']:>5}{s['last3']:>8.4f}"
              f"{s['slope']:>+9.4f}{s['gap']:>+8.4f}  {' '.join(flags)}")
    if missing:
        print(f"\nnot finished / no CSV yet ({len(missing)}): {' '.join(missing)}")
    # The incumbent bar is itself a max over a 12-epoch, still-rising, truncated
    # series; MLP and U-Net(full) are tied on every last-k window.
    print("\nreference bars: MLP best 0.3564 (max of a truncated series) | "
          "MLP/unet_full last3 ~0.346-0.348 | same-config seed spread 0.0107")
    for name, why in OVERWRITTEN_IN_GCS.items():
        if any(n == name for n, _ in got):
            print(f"note: {name} -- {why}")


if __name__ == "__main__":
    main()
