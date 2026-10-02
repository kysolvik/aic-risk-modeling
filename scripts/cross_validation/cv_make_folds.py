#!/usr/bin/env python
"""Generate the temporal-CV protocol jobs for one architecture.

PROTOCOL (decided 2026-09-15):

  folds    fwdpair_<t>: train 2013..t-1, early-stop AND validate on {t, t+1}, t=2018..2022.
           Same model on both years -> the t->t+1 amplitude is a leak-free measurement.
  final    final_all: train 2013-2023, no val, fixed 25 epochs, keep the last epoch
           (checkpoint_metric 'last'); test 2024+2025 -- WRITE-ONCE. Deviation
           2026-09-29 from the pre-registered train 2013-22 / early-stop 2023.
  seedrep  fwdpair_2020 at seed 55: noise floor for the ablations.
  ablateA  fwdpair_2018 / fwdpair_2020 + {2023 placebo, 2024 high, 2025 low}.
  ablateB  final minus {2017,2020} highs / {2013,2018} lows / {2016,2022} mids (placebo).
  posthoc  lead1_2023: train 2013-2022, early-stop AND validate on 2023 only (the fold
           recipe, single year so 2024 never enters). Lead-1 2023 point for the Fig 5
           expected-vs-actual series; added 2026-10-01 after the final was scored, so
           it never feeds selection or the 2024/25 test.

2024/2025 never enter a fold. `final`/`seedrep` are gated on out/cv/selection_frozen.json,
`ablateA`/`ablateB`/`posthoc` on out/cv/final_scored.json (and on being the selected
architecture).
`--allow_early_final` (ALLOW_EARLY_FINAL=1 for run_protocol_train.sh) opens the final
TRAINING gate before selection (deviation 2026-09-29); predicting/scoring stay gated.
Stats and gamma depend only on the training years, so they are shared across
architectures. Ablation arms reuse their base's stats (normalization held fixed) and
refit gamma.

Writes, per --arch: configs/cv/<arch>/<fold>.json, out/cv/gamma/gamma_<name>.json,
out/cv/protocol.csv (all architectures, merged), and
  out/cv/run_protocol_stats.sh   {folds|final|posthoc|all}
  out/cv/run_protocol_train.sh   <arch> {folds|final|seedrep|ablateA|ablateB|posthoc}
  out/cv/run_protocol_predict.sh <arch> {folds|final|seedrep|ablateA|ablateB|posthoc}
NOTHING here launches Vertex; run_protocol_train.sh is handed to the user to launch.

Usage:
    .venv/bin/python scripts/cross_validation/cv_make_folds.py --arch unet_v3p_union4 --chips_per_year 2556
    .venv/bin/python scripts/cross_validation/cv_make_folds.py --check_gate final --arch unet_v3p_union4
"""

import argparse
import copy
import csv
import json
import math
import os
import stat
import sys

from aic_risk_modeling.eval import year_offset as fyo

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- paths -----------------------------------------------------------------
CONFIG_DIR = os.path.join(REPO, "configs", "cv")
OUT_DIR = os.path.join(REPO, "out", "cv")
GAMMA_DIR = os.path.join(OUT_DIR, "gamma")

GS = "gs://aic-amazon"
DATA_VERSION = "v3_patched"   # data bucket suffix (fullgrid_<DATA_VERSION>); set from --data_version in main()
def data_dir_gs(y):   return f"{GS}/data/fullgrid_{DATA_VERSION}/allpreds_{y}/"
def gamma_stem(c):
    if GAMMA_LONG["tag"]:                            # long/weighted recipe: own namespace
        return f"{DATA_VERSION}_{GAMMA_LONG['tag']}_{c}"
    return f"{DATA_VERSION}_{c}"
def gamma_gs(c):      return f"{GS}/configs/cv/gamma_{gamma_stem(c)}.json"
def stats_gs(c):      return f"{GS}/data/fullgrid_{DATA_VERSION}/stats_cv/{c}.json"

ALL_YEARS = list(range(2013, 2026))          # 2013..2025 inclusive

# Gamma recipe, set from the CLI in main(). The defaults are the v3p yeargain recipe:
# LONG gamma fit on the targets-only panel (build_target_panel.py, 2002+), fit years =
# fit_start .. 2012 (before any fold's data) PLUS the fold's own train years (so ablation
# drops stay dropped), centred on the fold's train years, chip-weighted per `weighting`,
# emitted through `emit_through`. `--gamma_panel_kind chip` fits on a chip panel
# instead (the v3 mod14 archs). `tag` namespaces the file names.
GAMMA_KW = dict(target="bd", prev_burn="bd", space="logit")
GAMMA_LONG = dict(kind="target", weighting="burn", fit_start=2002, emit_through=2026,
                  tag="bd_2002_burn")


# --- gamma per fold --------------------------------------------------------
def make_gamma(panel_df, canon, train_years):
    """Fit gamma on `train_years`, emit offsets for all panel years, write JSON.

    Reuses the vetted eval.year_offset functions. Honors its sign guard: if the
    prev-burn coefficient comes out >= 0 (persistence, which lags every turn), fall
    back to a SOI-only gamma and flag it.
    """
    long_mode = GAMMA_LONG["kind"] == "target"
    if long_mode:
        # pre-network years only (< 2013, before any fold's data): an ablation that
        # drops 2013 must not get it back through the long extension
        fit_years = list(range(GAMMA_LONG["fit_start"], min(ALL_YEARS))) + list(train_years)
        w = GAMMA_LONG["weighting"]
    else:
        fit_years, w = list(train_years), "equal"
    beta, _ = fyo.fit_final(panel_df, fit_years, terms=fyo.BOTH, weighting=w)
    if beta[2] < 0:
        terms, terms_name = fyo.BOTH, "soi+prev"
        b_soi, b_prev, guard_ok = float(beta[1]), float(beta[2]), True
    else:                                             # sign guard fallback
        beta, _ = fyo.fit_final(panel_df, fit_years, terms=fyo.SOI, weighting=w)
        terms, terms_name = fyo.SOI, "soi_only(fallback:b_prev>=0)"
        b_soi, b_prev, guard_ok = float(beta[1]), None, False
    offsets, level = fyo.build_offsets(panel_df, beta, terms=terms, center_years=train_years)
    doc = {
        "version": f"gamma_cv_{canon}",
        "fit": {**GAMMA_KW, "fit_years": fit_years, "terms_used": terms_name,
                "protocol": "in-sample fit on fold train years; offsets emitted for all panel years"},
        "terms": ["soi_y1ond", "log_basin_prev_burn"],
        "coeffs": {"b0": float(beta[0]), "b_soi": b_soi, "b_prev": b_prev},
        "centering": {"removed_level": level, "center_years": list(train_years)},
        "per_year_offset": {str(y): v for y, v in offsets.items()},
    }
    if long_mode:                                     # chip-panel docs stay byte-identical
        doc["fit"].update(panel_kind="target", weighting=w, prev_burn="bd_count",
                          fit_start=GAMMA_LONG["fit_start"])
        doc["emit_years"] = sorted(int(y) for y in offsets)
    path = os.path.join(GAMMA_DIR, f"gamma_{gamma_stem(canon)}.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return {"b0": float(beta[0]), "b_soi": b_soi, "b_prev": b_prev,
            "sign_guard_ok": guard_ok, "terms": terms_name}


# --- config per fold -------------------------------------------------------


# --- helper-script generation ---------------------------------------------
def _write_exec(path, text):
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IRWXU)


# ===========================================================================
# PROTOCOL: forward t/t+1 folds, write-once 2024/25 final, post-hoc ablations
# ===========================================================================
PROTOCOL_CSV = os.path.join(OUT_DIR, "protocol.csv")
SELECTION_FROZEN = os.path.join(OUT_DIR, "selection_frozen.json")
FINAL_SCORED = os.path.join(OUT_DIR, "final_scored.json")

FINAL_TEST_YEARS = (2024, 2025)      # write-once: never in any fold's train/val/eval
FOLD_ORIGINS = tuple(range(2018, 2023))
# Deviation 2026-09-29: the final trains on every pre-test year with no val set, for a
# fixed 25 epochs, keeping the last (fold curves: best - last3 ~0.001-0.006 PR-AUC).
# Renamed from `final` so its stats/gamma/model paths never collide with the
# pre-registered 2013-22 recipe.
FINAL_ID = "final_all"
FINAL_TRAIN = tuple(range(2013, 2024))
FINAL_VAL = ()
PROBE_BASES = (2018, 2020)           # ablation A bases: a low->high and a high->low pair
PROBE_YEARS = (2023, 2024, 2025)     # 2023 ~ climatology-average year = placebo
DROP_SETS = ((2017, 2020), (2013, 2018), (2016, 2022))   # ablation B: highs, lows, mids
SEEDREP_ORIGIN, SEEDREP_SEED = 2020, 55
# Post-hoc (2026-10-01): one-year-ahead 2023 for Fig 5, same recipe as a fold.
POSTHOC_ORIGIN = 2023
DEFAULT_SEED = 54
# Matched budget across architectures. Patience 8 (not the trainer default 4): the
# two-year val sets include low-prevalence years, and patience 4 has already killed
# a low-prevalence run mid-climb (v41).
PROTOCOL_BUDGET = {"epochs": 25, "steps_per_epoch": 9065, "early_stopping_patience": 8,
                   "checkpoint_metric": "pr_auc"}
# A job with no val years trains the full budget and keeps the last epoch.
NO_VAL_BUDGET = {"checkpoint_metric": "last"}
NO_VAL_DROP = ("early_stopping_patience", "val_data_dirs", "val_tfrecord_pattern", "val_cache_dir")
STAGES = ("folds", "seedrep", "final", "ablateA", "ablateB", "posthoc")
SELECTED_ONLY = {"seedrep", "ablateA", "ablateB", "posthoc"}


def years_str(ys):
    return ";".join(str(y) for y in ys)


def parse_years(s):
    return [int(y) for y in str(s).split(";") if y]


def protocol_specs():
    """Every protocol training job, bases always listed before the arms that use them."""
    specs = []

    def add(fold, stage, train, val, ev, seed=DEFAULT_SEED, base="", probe=""):
        specs.append({"fold_id": fold, "stage": stage, "train": sorted(train),
                      "val": list(val), "eval": list(ev), "seed": seed,
                      "base_fold": base, "probe": probe})

    for t in FOLD_ORIGINS:
        add(f"fwdpair_{t}", "folds", range(2013, t), (t, t + 1), (t, t + 1))
    by_id = {s["fold_id"]: s for s in specs}
    b = by_id[f"fwdpair_{SEEDREP_ORIGIN}"]
    add(f"{b['fold_id']}_s{SEEDREP_SEED}", "seedrep", b["train"], b["val"], b["eval"],
        seed=SEEDREP_SEED, base=b["fold_id"])
    add(FINAL_ID, "final", FINAL_TRAIN, FINAL_VAL, FINAL_TEST_YEARS)
    for t in PROBE_BASES:
        b = by_id[f"fwdpair_{t}"]
        for y in PROBE_YEARS:
            add(f"{b['fold_id']}_add{y}", "ablateA", b["train"] + [y], b["val"], b["eval"],
                base=b["fold_id"], probe=f"+{y}")
    for drop in DROP_SETS:
        add(f"final_drop{drop[0]}-{drop[1]}", "ablateB",
            [y for y in FINAL_TRAIN if y not in drop], FINAL_VAL, FINAL_TEST_YEARS,
            base=FINAL_ID, probe="-" + ",".join(map(str, drop)))
    t = POSTHOC_ORIGIN
    add(f"lead1_{t}", "posthoc", range(2013, t), (t,), (t,))
    return specs


def protocol_violations(s):
    """Leakage-guard violations for one spec (empty list = clean)."""
    tr, va, ev = set(s["train"]), set(s["val"]), set(s["eval"])
    fin = set(FINAL_TEST_YEARS)
    errs = []
    if tr & va:
        errs.append(f"train/val overlap {sorted(tr & va)}")
    if tr & ev:
        errs.append(f"train/eval overlap {sorted(tr & ev)}")
    if s["stage"] in ("folds", "seedrep", "posthoc"):
        if (tr | va | ev) & fin:
            errs.append(f"fold touches write-once years {sorted((tr | va | ev) & fin)}")
        if va != ev:
            errs.append("fold val years must equal eval years")
    if s["stage"] in ("final", "ablateB"):
        if (tr | va) & fin:
            errs.append(f"final trains/early-stops on test years {sorted((tr | va) & fin)}")
        if ev != fin:
            errs.append(f"final must evaluate exactly {sorted(fin)}")
    if s["stage"] == "ablateA" and (ev | va) & fin:
        errs.append("ablation A must never early-stop or evaluate on 2024/2025")
    return errs


def arm_diff_errors(s, base):
    """An ablation arm / seed replicate must differ from its base by exactly the intended thing."""
    tr, btr = set(s["train"]), set(base["train"])
    errs = []
    if s["val"] != base["val"] or s["eval"] != base["eval"]:
        errs.append("val/eval differ from base")
    if s["stage"] == "seedrep":
        if tr != btr:
            errs.append("seed replicate changed the training years")
        if s["seed"] == base["seed"]:
            errs.append("seed replicate has the base seed")
    elif s["stage"] == "ablateA":
        want = {int(s["probe"].lstrip("+"))}
        if not (btr <= tr and tr - btr == want):
            errs.append(f"arm adds {sorted(tr - btr)}, expected {sorted(want)}")
    elif s["stage"] == "ablateB":
        want = {int(y) for y in s["probe"].lstrip("-").split(",")}
        if not (tr <= btr and btr - tr == want):
            errs.append(f"arm drops {sorted(btr - tr)}, expected {sorted(want)}")
    if s["stage"] != "seedrep" and s["seed"] != base["seed"]:
        errs.append("ablation arm changed the seed")
    return errs


def protocol_resources(specs):
    """fold_id -> {stats_name, gamma_name}, named per fold and shared across architectures.

    Ablation arms keep their base fold's stats (normalization held fixed) but refit gamma.
    """
    res = {}
    for s in specs:
        fid = s["fold_id"]
        stats = res[s["base_fold"]]["stats_name"] if s["stage"] in ("ablateA", "ablateB") else fid
        res[fid] = {"stats_name": stats, "gamma_name": fid}
    return res


def arch_paths(arch, job):
    return {"config_local": f"configs/cv/{arch}/{job}.json",
            "config_gs": f"{GS}/configs/cv/{arch}/{job}.json",
            "model_gs": f"{GS}/models/cv/{arch}/{job}.pt",
            "predict_root": f"out/cv/preds/{arch}/{job}"}


def config_guard_errors(cfg):
    """Matched-information guards: no static-snapshot leak, no absolute year as an input."""
    errs = []
    feats = cfg.get("input_features", {}) or {}
    for g, spec in feats.items():
        for n in spec.get("feature_names", []):
            if n.startswith("im_loss"):
                errs.append(f"{g}:{n} is the static Hansen snapshot (future-loss leak)")
    # Target-year bands (step 0) are labels, never inputs -- e.g. im_BurnDate_viirs_0,
    # im_viirs_noaa20_0, im_aqua_0 all ship in the patched export.
    targets = {f"{n}_0" for n in (cfg.get("output_features", {}) or {}).get("feature_names", [])}
    for g, spec in feats.items():
        steps = spec.get("timesteps") or []
        for n in spec.get("feature_names", []):
            for full in ([f"{n}_{t}" for t in steps] if steps else [n]):
                if full.endswith("_0") or full in targets:
                    errs.append(f"{g}:{full} is a target-year (step 0) band (label leak)")
    dc =cfg.get("decoder_config", {}) or {}
    yg = dc.get("year_group")
    for g, spec in feats.items():
        if "md_year" in spec.get("feature_names", []):
            if g != yg:
                errs.append(f"md_year group '{g}' is a network input (year_group={yg})")
            elif any(g in (dc.get(k) or []) for k in ("pixel_groups", "context_groups")):
                errs.append(f"md_year group '{g}' is also routed into the network")
    return errs


def job_budget(budget, has_val):
    """The budget keys one job's config must carry (no-val jobs keep the last epoch)."""
    out = dict(budget or PROTOCOL_BUDGET)
    if not has_val:
        out.update(NO_VAL_BUDGET)
        for k in NO_VAL_DROP:
            out.pop(k, None)
    return out


def make_protocol_config(base, spec, paths, stats_path, gamma_path, budget=None):
    cfg = copy.deepcopy(base)
    cfg["data_dirs"] = [data_dir_gs(y) for y in spec["train"]]
    cfg["val_data_dirs"] = [data_dir_gs(y) for y in spec["val"]]
    cfg["stats_path"] = stats_path
    cfg["model_output_path"] = paths["model_gs"]
    yo = (cfg.get("decoder_config") or {}).get("year_offset")
    if yo is not None:
        yo.pop("offsets", None)
        yo["coeffs_path"] = gamma_path
    cfg.pop("early_stopping_metric", None)          # stop on the checkpoint metric
    cfg.update(job_budget(budget, bool(spec["val"])))
    if not spec["val"]:
        for k in NO_VAL_DROP:
            cfg.pop(k, None)
    cfg["seed"] = spec["seed"]
    path = os.path.join(REPO, paths["config_local"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def budget_overrides(base, budget=None):
    """Keys the matched budget changes in this architecture's base config."""
    budget = budget or PROTOCOL_BUDGET
    out = {k: (base.get(k), v) for k, v in budget.items() if base.get(k) != v}
    if "early_stopping_metric" in base and base["early_stopping_metric"] != budget["checkpoint_metric"]:
        out["early_stopping_metric"] = (base["early_stopping_metric"], "(checkpoint metric)")
    return out


def gate_errors(stage, arch, frozen=SELECTION_FROZEN, scored=FINAL_SCORED,
                allow_early_final=False):
    """Why `stage` may not run yet for `arch` (empty list = allowed).

    `allow_early_final` opens only the final TRAINING gate before selection
    (deviation 2026-09-29). Only run_protocol_train.sh passes it; predicting and
    scoring the final stay gated on selection_frozen.json.
    """
    gate = {"final": frozen, "seedrep": frozen, "ablateA": scored, "ablateB": scored,
            "posthoc": scored}.get(stage)
    if stage not in STAGES:
        return [f"unknown stage {stage!r}"]
    if gate is None or (stage == "final" and allow_early_final):
        return []
    errs = []
    for g in ([frozen, scored] if gate == scored else [frozen]):
        if not os.path.exists(g):
            errs.append(f"{stage} is gated on {os.path.relpath(g, REPO)}, which does not exist")
    if not errs and stage in SELECTED_ONLY:
        with open(frozen) as f:
            selected = json.load(f).get("arch")
        if selected != arch:
            errs.append(f"{stage} runs for the selected architecture only "
                        f"(selection_frozen arch={selected!r}, got {arch!r})")
    return errs


def fold_steps_per_epoch(train_years, chips_per_year, batch_size):
    """Optimizer steps for one full pass over a fold, = ceil(total_chips / batch).

    steps_per_epoch here only shapes the cosine LR schedule (warmup 1 epoch, decay
    the rest); it does NOT cap the data -- each epoch is a full pass. So to keep the
    schedule aligned with each fold's actual run length it must scale with the
    fold's training size, not be a flat constant.
    """
    return math.ceil(len(train_years) * chips_per_year / batch_size)


def generate_protocol(arch, base, panel_df, budget=None, chips_per_year=None):
    """Configs + gammas for one architecture; returns its manifest rows.

    `budget` overrides PROTOCOL_BUDGET. When `chips_per_year` is given, each fold's
    steps_per_epoch is computed per fold (ceil(train_years * chips_per_year /
    batch_size)) so the LR schedule matches that fold's real run length, overriding
    any flat steps_per_epoch in the budget. A base config without a `year_offset`
    decoder term (any non-gamma architecture) skips gamma fitting entirely, so no
    panel is required for it.
    """
    batch_size = int(base.get("batch_size", 1))
    errs = config_guard_errors(base)
    if errs:
        raise SystemExit(f"[protocol] {arch} base config fails guards:\n  " + "\n  ".join(errs))
    uses_gamma = "year_offset" in (base.get("decoder_config") or {})
    specs = protocol_specs()
    by_id = {s["fold_id"]: s for s in specs}
    for s in specs:
        v = protocol_violations(s) + (arm_diff_errors(s, by_id[s["base_fold"]]) if s["base_fold"] else [])
        if v:
            raise SystemExit(f"[protocol] {s['fold_id']} violates the protocol: {v}")
    res = protocol_resources(specs)
    gamma_info = {}
    if uses_gamma:
        if panel_df is None:
            raise SystemExit(f"[protocol] {arch} uses a year_offset but no panel was loaded")
        for s in specs:                               # arch-independent; deterministic rewrite
            g = res[s["fold_id"]]["gamma_name"]
            if g not in gamma_info:
                gamma_info[g] = make_gamma(panel_df, g, s["train"])
    loss = base.get("loss_function", "")
    rows = []
    for s in specs:
        p = arch_paths(arch, s["fold_id"])
        r = res[s["fold_id"]]
        g = (gamma_info[r["gamma_name"]] if uses_gamma
             else {"b_soi": "", "b_prev": "", "sign_guard_ok": "", "terms": ""})
        g_gs = gamma_gs(r["gamma_name"]) if uses_gamma else ""
        fold_budget = dict(budget or PROTOCOL_BUDGET)
        if chips_per_year:
            fold_budget["steps_per_epoch"] = fold_steps_per_epoch(
                s["train"], chips_per_year, batch_size)
        make_protocol_config(base, s, p, stats_gs(r["stats_name"]), g_gs, budget=fold_budget)
        rows.append({
            "arch": arch, "fold_id": s["fold_id"], "stage": s["stage"],
            "steps_per_epoch": fold_budget["steps_per_epoch"],
            "train_years": years_str(s["train"]), "n_train_years": len(s["train"]),
            "val_years": years_str(s["val"]), "eval_years": years_str(s["eval"]),
            "seed": s["seed"], "base_fold": s["base_fold"], "probe": s["probe"],
            **p,
            "stats_name": r["stats_name"], "stats_gs": stats_gs(r["stats_name"]),
            "gamma_name": r["gamma_name"] if uses_gamma else "",
            "gamma_local": (os.path.relpath(os.path.join(GAMMA_DIR, f"gamma_{gamma_stem(r['gamma_name'])}.json"), REPO)
                            if uses_gamma else ""),
            "gamma_gs": g_gs,
            "loss_function": loss, "pos_weight": base.get("pos_weight", 9.0),
            "b_soi": g["b_soi"], "b_prev": g["b_prev"],
            "sign_guard_ok": g["sign_guard_ok"], "gamma_terms": g["terms"],
        })
    return rows


def merge_protocol_manifest(rows, arch, path=PROTOCOL_CSV):
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if r["arch"] != arch]
    merged = old + rows
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(merged)
    return merged


def _gate_line(stage_var="$STAGE", arch_var="$ARCH", early_final=False):
    early = ' ${ALLOW_EARLY_FINAL:+--allow_early_final}' if early_final else ""
    return (f'.venv/bin/python scripts/cross_validation/cv_make_folds.py --check_gate "{stage_var}" '
            f'--arch "{arch_var}"{early} || exit 1')


def gen_protocol_scripts(rows):
    head = ["#!/usr/bin/env bash", "set -euo pipefail", f'cd "{REPO}"']

    # stats: architecture-independent, one per training set, skip what already exists
    lines = head[:1] + ["# Per-training-set normalization stats -> GCS (local CPU, reads GCS; no Vertex).",
                        "# Ablation arms reuse their base's stats, so only folds/final/posthoc need any.",
                        "# Usage: run_protocol_stats.sh {folds|final|posthoc|all}"] + head[1:] + \
            ['GROUP="${1:-folds}"', ""]
    for group, stages in (("folds", {"folds"}), ("final", {"final"}), ("posthoc", {"posthoc"}),
                          ("all", {"folds", "final", "posthoc"})):
        lines.append(f'if [ "$GROUP" = "{group}" ]; then')
        seen = set()
        for r in rows:
            if r["stage"] not in stages or r["stats_name"] in seen:
                continue
            seen.add(r["stats_name"])
            dirs = " ".join(data_dir_gs(y) for y in parse_years(r["train_years"]))
            lines.append(f'  if gsutil -q stat {r["stats_gs"]}; then echo "[stats] {r["stats_name"]} exists, skip"; '
                         f'else echo "[stats] {r["stats_name"]}"; .venv/bin/python -m aic_risk_modeling.train.data_stats '
                         f'--data_dirs {dirs} --output {r["stats_gs"]}; fi')
        lines.append("fi")
    _write_exec(os.path.join(OUT_DIR, "run_protocol_stats.sh"), "\n".join(lines) + "\n")

    # train: LAUNCHES VERTEX JOBS -- handed to the user
    lines = head[:1] + ["# Launch protocol training on Vertex. LAUNCHES VERTEX JOBS.",
                        "# Usage: run_protocol_train.sh <arch> {folds|final|seedrep|ablateA|ablateB|posthoc}",
                        "# ALLOW_EARLY_FINAL=1 trains the final before selection is frozen (deviation 2026-09-29)."] + head[1:] + \
            ['ARCH="$1"; STAGE="$2"', _gate_line(early_final=True), ""]
    for arch in sorted({r["arch"] for r in rows}):
        for stage in STAGES:
            sel = [r for r in rows if r["arch"] == arch and r["stage"] == stage]
            if not sel:
                continue
            lines.append(f'if [ "$ARCH" = "{arch}" ] && [ "$STAGE" = "{stage}" ]; then')
            for r in sel:
                lines.append(f'  echo "[train] {arch} {r["fold_id"]}"')
                lines.append(f'  gsutil -q stat {r["stats_gs"]} || {{ echo "MISSING stats {r["stats_gs"]}" >&2; exit 1; }}')
                lines.append(f'  if gsutil -q stat {r["model_gs"]}; then echo "  {r["model_gs"]} exists -- '
                             f'refusing to overwrite (delete it by hand to retrain)" >&2; exit 1; fi')
                lines.append(f'  gsutil cp {r["config_local"]} {r["config_gs"]}')
                if r["gamma_gs"]:
                    lines.append(f'  gsutil cp {r["gamma_local"]} {r["gamma_gs"]}')
                lines.append(f'  .venv/bin/python scripts/train/train_vertex.py {r["config_gs"]} '
                             f'cvp_{arch}_{r["fold_id"]}')
            lines.append("fi")
    _write_exec(os.path.join(OUT_DIR, "run_protocol_train.sh"), "\n".join(lines) + "\n")

    # predict: every eval year of every job, skipping years already predicted
    lines = head[:1] + ["# Predict each protocol job's eval years with its own checkpoint + stats.",
                        "# Usage: run_protocol_predict.sh <arch> {folds|final|seedrep|ablateA|ablateB|posthoc}"] + head[1:] + \
            ['ARCH="$1"; STAGE="$2"', _gate_line(), ""]
    for arch in sorted({r["arch"] for r in rows}):
        for stage in STAGES:
            sel = [r for r in rows if r["arch"] == arch and r["stage"] == stage]
            if not sel:
                continue
            lines.append(f'if [ "$ARCH" = "{arch}" ] && [ "$STAGE" = "{stage}" ]; then')
            for r in sel:
                for y in parse_years(r["eval_years"]):
                    out = f'{r["predict_root"]}/{y}/'
                    lines.append(f'  if compgen -G "{out}chips/out_*.tif" > /dev/null; then '
                                 f'echo "[predict] {r["fold_id"]} {y} exists, skip"; else '
                                 f'echo "[predict] {r["fold_id"]} {y}"; '
                                 f'.venv/bin/python scripts/predict/predict.py --config_path {r["config_local"]} '
                                 f'--checkpoint {r["model_gs"]} --data_dir {data_dir_gs(y)} '
                                 f'--stats_path {r["stats_gs"]} --output_dir {out}; fi')
            lines.append("fi")
    _write_exec(os.path.join(OUT_DIR, "run_protocol_predict.sh"), "\n".join(lines) + "\n")


# --- main ------------------------------------------------------------------
def main():
    global DATA_VERSION
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True,
                    help="architecture tag; paths nest under it")
    ap.add_argument("--base_config", default=None,
                    help="base training config for --arch (default configs/<arch>.json)")
    ap.add_argument("--data_version", default=DATA_VERSION,
                    help="data bucket suffix: fullgrid_<data_version> for data + stats_cv")
    ap.add_argument("--chips_per_year", type=int, default=None,
                    help="examples per training year; sets each fold's steps_per_epoch = "
                         "ceil(n_train_years * chips_per_year / batch_size) so the LR "
                         "schedule matches that fold's real run length")
    ap.add_argument("--check_gate", choices=STAGES, default=None,
                    help="exit non-zero unless STAGE may run for --arch; generates nothing")
    ap.add_argument("--allow_early_final", action="store_true",
                    help="with --check_gate final: open the final TRAINING gate before "
                         "selection_frozen.json exists (deviation 2026-09-29)")
    ap.add_argument("--panel", default=os.path.join(REPO, "out", "target_panel", "panel.parquet"),
                    help="panel for gamma fitting (default the targets-only panel; pass "
                         "out/chip_panel_v3/panel.parquet with --gamma_panel_kind chip)")
    ap.add_argument("--gamma_target", default=GAMMA_KW["target"], choices=sorted(fyo.TARGETS),
                    help="gamma year-level target (union3 = bd+snpp+mod14 on a chip panel)")
    ap.add_argument("--gamma_prev", default=GAMMA_KW["prev_burn"], choices=sorted(fyo.PREV_BANDS),
                    help="chip panel: gamma prev-year-burn predictor (union3 = the 3 sensors)")
    ap.add_argument("--gamma_panel_kind", default=GAMMA_LONG["kind"], choices=["chip", "target"],
                    help="target = long targets-only panel (prev-burn is the MCD64 count, "
                         "--gamma_prev is ignored); chip = a chip panel")
    ap.add_argument("--gamma_weighting", default=GAMMA_LONG["weighting"], choices=["equal", "burn"],
                    help="chip weighting of the year effect (burn needs --gamma_panel_kind target)")
    ap.add_argument("--gamma_fit_start", type=int, default=GAMMA_LONG["fit_start"],
                    help="target panel: first gamma fit year (earlier than any train year)")
    ap.add_argument("--gamma_emit_through", type=int, default=GAMMA_LONG["emit_through"],
                    help="target panel: emit offsets through this (predict-only) year")
    args = ap.parse_args()

    # Fold the recipe choice into GAMMA_KW / GAMMA_LONG so both the panel load and the
    # recipe recorded in each gamma JSON (make_gamma) reflect it.
    GAMMA_KW["target"], GAMMA_KW["prev_burn"] = args.gamma_target, args.gamma_prev
    if args.gamma_weighting != "equal" and args.gamma_panel_kind != "target":
        ap.error("--gamma_weighting burn needs --gamma_panel_kind target")
    if args.gamma_panel_kind == "target":
        GAMMA_LONG.update(kind="target", weighting=args.gamma_weighting,
                          fit_start=args.gamma_fit_start, emit_through=args.gamma_emit_through,
                          tag=f"{args.gamma_target}_{args.gamma_fit_start}_{args.gamma_weighting}")
    else:
        GAMMA_LONG.update(kind="chip", weighting="equal", fit_start=None, emit_through=None, tag="")

    DATA_VERSION = args.data_version
    budget = dict(PROTOCOL_BUDGET)

    if args.check_gate:
        errs = gate_errors(args.check_gate, args.arch, allow_early_final=args.allow_early_final)
        if args.allow_early_final and args.check_gate == "final" and not os.path.exists(SELECTION_FROZEN):
            print("GATE OVERRIDE: training the final before selection is frozen "
                  "(deviation 2026-09-29; predict/score stay gated)", file=sys.stderr)
        for e in errs:
            print(f"GATE CLOSED: {e}", file=sys.stderr)
        sys.exit(1 if errs else 0)

    for d in (CONFIG_DIR, OUT_DIR, GAMMA_DIR):
        os.makedirs(d, exist_ok=True)

    base_path = args.base_config or os.path.join(REPO, "configs", f"{args.arch}.json")
    with open(base_path) as f:
        base = json.load(f)
    panel_df = None
    if "year_offset" in (base.get("decoder_config") or {}):
        if GAMMA_LONG["kind"] == "target":
            panel_df = fyo.load_target_panel(args.panel, target=GAMMA_KW["target"], space=GAMMA_KW["space"],
                                             emit_through=GAMMA_LONG["emit_through"])
            panel_df = panel_df[panel_df.year >= GAMMA_LONG["fit_start"]]
        else:
            panel_df = fyo.load_panel(args.panel, **GAMMA_KW)

    rows = generate_protocol(args.arch, base, panel_df, budget=budget,
                             chips_per_year=args.chips_per_year)
    merged = merge_protocol_manifest(rows, args.arch)
    gen_protocol_scripts(merged)

    print(f"\n[protocol] {args.arch}: {len(rows)} jobs from {os.path.relpath(base_path, REPO)}")
    for stage in STAGES:
        ids = [r["fold_id"] for r in rows if r["stage"] == stage]
        print(f"  {stage:<8} {len(ids):>2}  {' '.join(ids)}")
    if args.chips_per_year:
        print(f"  per-fold steps_per_epoch ({args.chips_per_year} chips/yr, "
              f"batch {base.get('batch_size', 1)}):")
        for r in rows:
            if r["stage"] in ("folds", "final", "posthoc"):
                print(f"    {r['fold_id']:<14} {r['n_train_years']}yr -> {r['steps_per_epoch']}")
    over = budget_overrides(base, budget)
    if args.chips_per_year:
        over.pop("steps_per_epoch", None)   # superseded by the per-fold values above
    if over:
        print("  matched budget overrides: " +
              ", ".join(f"{k} {a!r}->{b!r}" for k, (a, b) in over.items()))
    bad = [r["fold_id"] for r in rows if r["gamma_name"] and not r["sign_guard_ok"]]
    if bad:
        print(f"  WARNING gamma sign-guard fallback (SOI-only) on: {bad}")
    print(f"  manifest: {os.path.relpath(PROTOCOL_CSV, REPO)} ({len(merged)} rows, "
          f"{len({r['arch'] for r in merged})} arch)")
    print("  scripts:  out/cv/run_protocol_{stats,train,predict}.sh")


if __name__ == "__main__":
    main()
