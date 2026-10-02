#!/usr/bin/env python
"""Resumable one-architecture protocol scoring (appends to out/cv/score_parts/<arch>.csv); --merge folds parts in.

Usage: cv_score_parallel.py --arch ARCH --label_dir DIR  |  cv_score_parallel.py --merge --arch ARCH ..."""
import argparse
import csv
import glob
import os

import pandas as pd

import cv_collect_results as cc
import cv_make_folds as mk

PARTS_DIR = os.path.join(cc.OUT_DIR, "score_parts")


def part_path(arch):
    return os.path.join(PARTS_DIR, f"{arch}.csv")


def score_arch(arch, report, label_dir):
    rows = [r for r in csv.DictReader(open(cc.PROTOCOL)) if r["arch"] == arch]
    by_key = {(r["arch"], r["fold_id"]): r for r in rows}
    path = part_path(arch)
    part = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    done = set() if part.empty else set(zip(part.fold_id, part.year))
    clim = cc.Climatology(label_dir)
    os.makedirs(PARTS_DIR, exist_ok=True)
    for r in rows:
        if r["stage"] not in cc.REPORT_STAGES[report] or mk.gate_errors(r["stage"], arch):
            continue
        for y in mk.parse_years(r["eval_years"]):
            if (r["fold_id"], y) in done:
                continue
            if not glob.glob(os.path.join(cc.REPO, r["predict_root"], str(y), "**", "out_*.tif"),
                             recursive=True):
                print(f"[pending] {arch}/{r['fold_id']}/{y}", flush=True)
                continue
            print(f"[score] {arch}/{r['fold_id']} {y}", flush=True)
            base = by_key.get((arch, r["base_fold"])) if r["base_fold"] else None
            rec = pd.DataFrame([cc.score_protocol_year(r, y, base, clim)])
            part = rec if part.empty else pd.concat([part, rec], ignore_index=True)
            part.to_csv(path, index=False)


def merge(archs, report, ref_arch):
    parts = [pd.read_csv(part_path(a)) for a in archs if os.path.exists(part_path(a))]
    fresh = pd.concat(parts, ignore_index=True)
    cache = pd.read_csv(cc.SCORES) if os.path.exists(cc.SCORES) else pd.DataFrame()
    if not cache.empty:
        keep = ~cache.set_index(["arch", "fold_id", "year"]).index.isin(
            fresh.set_index(["arch", "fold_id", "year"]).index)
        cache = pd.concat([cache[keep], fresh], ignore_index=True)
    else:
        cache = fresh
    cache = cache.sort_values(["arch", "stage", "fold_id", "year"])
    cache.to_csv(cc.SCORES, index=False)
    print(f"[merge] {len(fresh)} rows from {len(parts)} part files -> {cc.SCORES}")
    rows = [r for r in csv.DictReader(open(cc.PROTOCOL)) if r["arch"] in archs]
    cc.write_protocol_report(report, cache[cache.arch.isin(archs)], rows, ref_arch)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", nargs="+", required=True)
    ap.add_argument("--protocol", choices=["folds"], default="folds")
    ap.add_argument("--label_dir", default=cc.LABEL_DIR)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--ref_arch", default="factored_v3p_union4_monthlyattn_wide_yeargain")
    args = ap.parse_args()
    if args.merge:
        merge(args.arch, args.protocol, args.ref_arch)
        return
    if len(args.arch) != 1:
        ap.error("score one --arch per process (use --merge for several)")
    score_arch(args.arch[0], args.protocol, args.label_dir)


if __name__ == "__main__":
    main()
