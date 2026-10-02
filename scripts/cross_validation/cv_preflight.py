#!/usr/bin/env python
"""CPU pre-flight for the CV protocol -- run before spending any Vertex $. Read-only.

Re-derives every job in out/cv/protocol.csv from the ACTUAL config files, not the
manifest: leakage guards (2024/25 never in a fold, final evaluates exactly them),
each ablation arm / seed replicate differs from its base by exactly the intended
thing, ablation arms keep the base's stats, matched budget, no im_loss* inputs,
md_year only as the gamma year_group, gamma covers every train/val/eval year, gamma
sign guard, unique model paths, and one model per architecture builds from a
protocol config.

Usage: .venv/bin/python scripts/cross_validation/cv_preflight.py [--arch ARCH ...]
"""

import argparse
import csv
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

from aic_risk_modeling.train.factored import YearOffset      # noqa: E402
from aic_risk_modeling.train import trainer                  # noqa: E402
import cv_make_folds as mk                                   # noqa: E402

PROTOCOL = os.path.join(REPO, "out", "cv", "protocol.csv")


def _dir_years(dirs):
    return sorted(int(d.rstrip("/").rsplit("_", 1)[1]) for d in dirs)


def protocol_row_errors(r, cfg):
    """Every guard for one protocol row, evaluated on the config that will actually run."""
    spec = {"fold_id": r["fold_id"], "stage": r["stage"],
            "train": _dir_years(cfg["data_dirs"]), "val": _dir_years(cfg.get("val_data_dirs", [])),
            "eval": mk.parse_years(r["eval_years"]), "seed": cfg.get("seed"),
            "base_fold": r["base_fold"], "probe": r["probe"]}
    errs = mk.protocol_violations(spec) + mk.config_guard_errors(cfg)
    if spec["train"] != mk.parse_years(r["train_years"]) or spec["val"] != mk.parse_years(r["val_years"]):
        errs.append("config years disagree with the manifest")
    want = mk.job_budget(mk.PROTOCOL_BUDGET, bool(spec["val"]))
    if r.get("steps_per_epoch"):                     # per-fold value (--chips_per_year)
        want["steps_per_epoch"] = int(r["steps_per_epoch"])
    for k, v in want.items():
        if cfg.get(k) != v:
            errs.append(f"budget {k}={cfg.get(k)!r}, protocol {v!r}")
    if not spec["val"]:
        errs += [f"no-val job still sets {k}" for k in mk.NO_VAL_DROP if k in cfg]
    if "early_stopping_metric" in cfg:
        errs.append("early_stopping_metric set (protocol stops on the checkpoint metric)")
    if cfg.get("model_output_path") != r["model_gs"] or cfg.get("stats_path") != r["stats_gs"]:
        errs.append("model/stats path disagree with the manifest")
    yo = (cfg.get("decoder_config") or {}).get("year_offset")
    if r["gamma_gs"]:
        if not yo or yo.get("coeffs_path") != r["gamma_gs"]:
            errs.append("year_offset.coeffs_path disagrees with the manifest")
        g = YearOffset.from_json(os.path.join(REPO, r["gamma_local"]), input_name="md_year", strict=True)
        need = spec["train"] + spec["val"] + spec["eval"]
        try:
            g(torch.tensor([[float(y)] for y in need]))
        except KeyError:
            errs.append(f"gamma {r['gamma_name']} misses a train/val/eval year")
        if r["sign_guard_ok"] != "True":
            errs.append(f"gamma {r['gamma_name']} fell back to SOI-only (b_prev >= 0)")
    elif yo is not None:
        errs.append("config has year_offset but the manifest has no gamma")
    return spec, errs


def check_protocol(archs=None):
    """Protocol invariants for every architecture in out/cv/protocol.csv."""
    if not os.path.exists(PROTOCOL):
        print("no out/cv/protocol.csv -- run cv_make_folds.py first")
        return False
    rows = [r for r in csv.DictReader(open(PROTOCOL)) if not archs or r["arch"] in archs]
    print(f"protocol: {len(rows)} jobs, archs {sorted({r['arch'] for r in rows})}")
    specs, row_of, bad = {}, {}, 0
    for r in rows:
        cfg = json.load(open(os.path.join(REPO, r["config_local"])))
        spec, errs = protocol_row_errors(r, cfg)
        specs[(r["arch"], r["fold_id"])] = spec
        row_of[(r["arch"], r["fold_id"])] = r
        if errs:
            bad += 1
            print(f"  [FAIL] {r['arch']}/{r['fold_id']}: {'; '.join(errs)}")
    for (arch, fid), spec in specs.items():
        r = row_of[(arch, fid)]
        if not spec["base_fold"]:
            continue
        base = specs.get((arch, spec["base_fold"]))
        errs = ["base fold missing"] if base is None else mk.arm_diff_errors(spec, base)
        if base is not None and r["stage"] in ("ablateA", "ablateB") and \
                r["stats_gs"] != row_of[(arch, spec["base_fold"])]["stats_gs"]:
            errs.append("ablation arm does not reuse the base's stats")
        if errs:
            bad += 1
            print(f"  [FAIL] {arch}/{fid} vs {spec['base_fold']}: {'; '.join(errs)}")
    paths = [r["model_gs"] for r in rows]
    dupes = sorted({p for p in paths if paths.count(p) > 1})
    if dupes:
        bad += 1
        print(f"  [FAIL] model_output_path collisions (overwrite trap): {dupes}")
    print(f"  {bad} failure(s)" if bad else
          f"  all {len(rows)} jobs clean (leakage guards, arm diffs, budget, gamma, unique paths)")

    for arch in sorted({r["arch"] for r in rows}):
        r = next(x for x in rows if x["arch"] == arch and x["stage"] == "folds")
        cfg = json.load(open(os.path.join(REPO, r["config_local"])))
        yo = (cfg.get("decoder_config") or {}).get("year_offset")
        if yo is not None:
            yo["coeffs_path"] = os.path.join(REPO, r["gamma_local"])
        try:
            branch_models = trainer.build_all_models(cfg["input_features"])
            model = trainer.build_decoder(cfg["decoder"], branch_models, cfg["decoder_config"])
        except (ValueError, TypeError) as e:
            bad += 1
            print(f"  [FAIL] {arch} does not build from {r['config_local']}: {str(e).splitlines()[0]}")
            continue
        n_params = sum(p.numel() for p in model.parameters())
        msg = f"  built {arch} from {r['config_local']}: {n_params:,} params"
        if yo is not None:
            y = mk.parse_years(r["eval_years"])[0]
            msg += f", gamma({y})={float(model.year(torch.tensor([[float(y)]])).flatten()[0]):+.4f}"
        print(msg)
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", nargs="*", default=None, help="limit to these archs")
    ok = check_protocol(ap.parse_args().arch)
    print("\nPRE-FLIGHT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
