"""Print one TSV row per (fold, eval year) of a CV-protocol arch/stage, for the docker scripts.

Usage: cv_protocol_rows.py <arch> <stage> [protocol.csv]  (env FOLDS / YEARS filter or override)
Fields: fold_id, year, config_gs, model_gs, stats_gs, predict_root."""
import csv
import os
import re
import sys


def _env_list(name):
    return [v for v in re.split(r"[ ;,]+", os.environ.get(name, "")) if v]


arch, stage = sys.argv[1], sys.argv[2]
path = sys.argv[3] if len(sys.argv) > 3 else "out/cv/protocol.csv"
folds, years = _env_list("FOLDS"), _env_list("YEARS")
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        if r["arch"] == arch and r["stage"] == stage:
            if folds and r["fold_id"] not in folds:
                continue
            ys = years or [y.strip() for y in r["eval_years"].split(";") if y.strip()]
            for y in ys:
                print("\t".join([r["fold_id"], y, r["config_gs"],
                                 r["model_gs"], r["stats_gs"], r["predict_root"]]))
