#!/usr/bin/env python
"""Score temporal-CV protocol predictions and write the protocol reports.

PROTOCOL mode (`--protocol folds|final|ablate`, see cv_make_folds.py) scores every
(job, eval year) of out/cv/protocol.csv that has predictions, caching rows in
out/cv/protocol_scores.csv and per-chip totals in out/cv/scores/:

  per year   pr_auc, within_chip_only, chip_r (decompose), prevalence, A (actual burned
             px), E_defl (deflated expected px), logbias = log(E_defl/A), brier (deflated),
             clim_pr_auc + pr_auc_skill vs the fold climatology (PIXEL-WISE burn
             frequency over the train-year label mosaics -- the primary, standard
             forecast-verification reference; folds/final only), plus clim_pr_auc_9x9
             + pr_auc_skill_9x9 against a 9x9-box-mean (receptive-field-matched)
             climatology reported as a secondary; persist_pr_auc + pr_auc_skill_persist
             against last-year persistence (label_{year-1}); clim_/persist_ E, logbias,
             brier for both references (cached rows are backfilled without rescoring the
             model), and the same E/logbias/brier
             with gamma REMOVED (network only) and, for ablation arms, with the BASE
             fold's gamma swapped in. Gamma is an additive logit term, so the swap
             sigmoid(logit q - gamma_job(t) + gamma_other(t)) is exact.
  per job    best / best_ep / last3 / truncated from its training CSV
             (out/cv/train_csv/[<arch>/]<job>.csv; --fetch_train_csv pulls it from GCS)
  per pair   (report time) R_A, R_E, log_ratio_err = log(R_E/R_A), amp_frac =
             log R_E / log R_A (turn pairs only), per-chip r(dE, dA); total and network-only

`final` and `ablate` are gated exactly like the run scripts (cv_make_folds.gate_errors).
`--protocol final` writes out/cv/final_scored.json, which opens the ablation gate.
`--protocol ablate` only scores; cv_year_sensitivity.py writes that report.

Usage:
    .venv/bin/python scripts/cross_validation/cv_collect_results.py --protocol folds
    .venv/bin/python scripts/cross_validation/cv_collect_results.py --protocol final
"""

import argparse
import csv
import datetime
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))  # decompose_scale

from decompose_scale import decompose, _pr_auc   # noqa: E402

OUT_DIR = os.path.join(REPO, "out", "cv")


# ===========================================================================
# PROTOCOL scoring
# ===========================================================================
PROTOCOL = os.path.join(OUT_DIR, "protocol.csv")
SCORES = os.path.join(OUT_DIR, "protocol_scores.csv")
CHIP_SCORES_DIR = os.path.join(OUT_DIR, "scores")
TRAIN_CSV_DIR = os.path.join(OUT_DIR, "train_csv")
LABEL_DIR = os.path.join(REPO, "out", "label_mosaics_v3p_union4")
CLIM_KERNEL_PIXELWISE = 1             # primary: standard per-pixel climatology (no pooling)
CLIM_KERNEL_9X9 = 9                   # secondary: the models' 9x9 (~5 km) receptive field
TURN_MIN = math.log(1.25)             # |log A(t+1)/A(t)| below this = flat pair, no amp_frac
REPORT_STAGES = {"folds": ("folds",), "final": ("final",),
                 "ablate": ("folds", "seedrep", "final", "ablateA", "ablateB")}
EPS = 1e-7


def deflate(q, pos_weight):
    """Invert the weighted-BCE optimum q = w*p/(w*p+1-p) back to p (compare_year_totals)."""
    return q / (pos_weight - (pos_weight - 1.0) * q)


def swap_gamma(q, g_from, g_to):
    """Re-express probabilities under a different additive year offset (exact)."""
    q = np.clip(np.asarray(q, dtype=np.float64), EPS, 1.0 - EPS)
    return 1.0 / (1.0 + np.exp(-(np.log(q / (1.0 - q)) - g_from + g_to)))


def box_mean(a, k):
    """k x k mean over a 2-D array, zero-padded at the edges (integral image, odd k)."""
    r = k // 2
    p = np.pad(np.asarray(a, dtype=np.float64), r)
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k)).astype(np.float32)


def pr_auc_skill(score, ref):
    return (score - ref) / (1.0 - ref) if ref < 1.0 else float("nan")


def pair_metrics(a_t, a_t1, e_t, e_t1, chip_a_t=None, chip_a_t1=None, chip_e_t=None, chip_e_t1=None):
    """Same model on two consecutive years: how well it tracks the change."""
    log_ra, log_re = math.log(a_t1 / a_t), math.log(e_t1 / e_t)
    out = {"R_A": a_t1 / a_t, "R_E": e_t1 / e_t, "log_ratio_err": log_re - log_ra,
           "turn": abs(log_ra) >= TURN_MIN,
           "amp_frac": log_re / log_ra if abs(log_ra) >= TURN_MIN else float("nan"),
           "r_dE_dA": float("nan")}
    if chip_a_t is not None:
        da, de = np.asarray(chip_a_t1) - chip_a_t, np.asarray(chip_e_t1) - chip_e_t
        if da.std() > 0 and de.std() > 0:
            out["r_dE_dA"] = float(np.corrcoef(de, da)[0, 1])
    return out


def load_year_chips(pred_dir):
    """(keys, bounds, scores[n,H,W] float32, labels[n,H,W] bool), in load_chips order."""
    import rasterio as rio
    paths = sorted(glob.glob(os.path.join(pred_dir, "**", "out_*.tif"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no out_*.tif under {pred_dir}")
    keys, bounds, scores, labels = [], [], [], []
    for p in paths:
        m = os.path.join(os.path.dirname(p), os.path.basename(p).replace("out_", "mask_", 1))
        with rio.open(p) as s:
            scores.append(s.read(1).astype(np.float32))
        with rio.open(m) as s:
            labels.append(s.read(1) > 0)
            bounds.append(tuple(s.bounds))
        keys.append(os.path.basename(p)[4:-4])
    return keys, bounds, np.stack(scores), np.stack(labels)


class Climatology:
    """Fold climatology from the full-basin label mosaics, windowed onto the chips.

    The primary reference is the PIXEL-WISE burn frequency (`kernel=1`, no spatial
    pooling) -- the standard forecast-verification climatology (grid-cell-wise
    long-run frequency). A `kernel>1` box mean gives the receptive-field-matched
    variant reported as a secondary. The per-fold pixel-wise frequency is cached
    once and box-meaned per kernel on demand, so both references share one mosaic read.
    """

    def __init__(self, label_dir=LABEL_DIR):
        self.label_dir = label_dir
        self._years, self.transform = {}, None
        self._fold_key, self._fold_freq, self._fold_clim = None, None, {}
        self._persist = {}

    def _mosaic(self, y):
        if y not in self._years:
            import rasterio as rio
            path = os.path.join(self.label_dir, f"label_{y}.tif")
            with rio.open(path) as s:
                if self.transform is None:
                    self.transform, self.shape = s.transform, s.shape
                elif s.transform != self.transform or s.shape != self.shape:
                    raise ValueError(f"{path} is not on the same grid as the other mosaics")
                self._years[y] = (s.read(1) > 0)
        return self._years[y]

    def windows(self, bounds):
        from rasterio.windows import from_bounds
        out = []
        for b in bounds:
            w = from_bounds(*b, transform=self.transform).round_offsets().round_lengths()
            out.append((int(w.row_off), int(w.col_off), int(w.height), int(w.width)))
        return out

    def chips(self, train_years, bounds, kernel=CLIM_KERNEL_PIXELWISE):
        key = tuple(sorted(train_years))
        if self._fold_key != key:                    # recompute only when the fold changes
            total = np.zeros(self._mosaic(key[0]).shape, dtype=np.uint16)
            for y in key:
                total += self._mosaic(y)
            self._fold_key = key
            self._fold_freq = (total / len(key)).astype(np.float32)   # pixel-wise burn frequency
            self._fold_clim = {}
        if kernel not in self._fold_clim:
            self._fold_clim[kernel] = (self._fold_freq if kernel == 1
                                       else box_mean(self._fold_freq, kernel))
        return self._cut(self._fold_clim[kernel], bounds)

    def persistence(self, year, bounds, kernel=CLIM_KERNEL_PIXELWISE):
        """Last-year reference: the label mosaic of `year` (= eval year - 1), windowed."""
        if (year, kernel) not in self._persist:
            m = self._mosaic(year).astype(np.float32)
            self._persist[(year, kernel)] = m if kernel == 1 else box_mean(m, kernel)
        return self._cut(self._persist[(year, kernel)], bounds)

    def _cut(self, grid, bounds):
        out = []
        for r0, c0, h, w in self.windows(bounds):
            v = np.zeros((h, w), dtype=np.float32)
            rs, cs = max(r0, 0), max(c0, 0)
            re, ce = min(r0 + h, grid.shape[0]), min(c0 + w, grid.shape[1])
            if re > rs and ce > cs:
                v[rs - r0:re - r0, cs - c0:ce - c0] = grid[rs:re, cs:ce]
            out.append(v)
        return np.stack(out)


def gamma_table(path):
    if not path:
        return None
    with open(os.path.join(REPO, path)) as f:
        return {int(y): float(v) for y, v in json.load(f)["per_year_offset"].items()}


def effective_pos_weight(row):
    from aic_risk_modeling.train import losses
    return float(row["pos_weight"]) if row["loss_function"] in losses.POS_WEIGHT_LOSSES else 1.0


def year_totals(scores, labels, pos_weight, g_from=None, g_to=None, batch=128):
    """Deflated expected burned px per chip + Brier, optionally under a gamma swap."""
    n = scores.shape[0]
    e_chip = np.zeros(n)
    sq = 0.0
    for i in range(0, n, batch):
        q = scores[i:i + batch].astype(np.float64)
        if g_from is not None:
            q = swap_gamma(q, g_from, g_to)
        p = deflate(np.clip(q, 0.0, 1.0), pos_weight)
        e_chip[i:i + batch] = p.reshape(p.shape[0], -1).sum(axis=1)
        sq += float(((p - labels[i:i + batch]) ** 2).sum())
    a_chip = labels.reshape(n, -1).sum(axis=1).astype(np.float64)
    e, a = float(e_chip.sum()), float(a_chip.sum())
    return {"E": e, "A": a, "logbias": math.log(e / a), "brier": sq / labels.size,
            "chip_E": e_chip, "chip_A": a_chip}


def train_csv_path(row):
    sub = "" if row["arch"] == "factored_v1" else f"{row['arch']}/"
    return os.path.join(TRAIN_CSV_DIR, f"{sub}{row['fold_id']}.csv")


def summarize_curve(rows):
    """Best / last-3 / truncation summary of a training CSV's val_pr_auc curve."""
    v = np.array([float(r["val_pr_auc"]) for r in rows])
    best = int(v.argmax())
    return {"best": float(v[best]), "best_ep": best, "last3": float(v[-3:].mean()),
            "truncated": best == len(v) - 1, "n_epochs": len(v)}


def training_summary(row, fetch=False):
    import subprocess
    path = train_csv_path(row)
    if not os.path.exists(path) and fetch:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        remote = os.path.splitext(row["model_gs"])[0] + ".csv"
        subprocess.run(["gsutil", "-q", "cp", remote, path], check=False)
    if not os.path.exists(path):
        return {"best": float("nan"), "best_ep": -1, "last3": float("nan"),
                "truncated": "", "n_epochs": 0}
    return summarize_curve(list(csv.DictReader(open(path))))


REF_COLS = ("clim_pr_auc", "clim_pr_auc_9x9", "clim_E", "clim_logbias", "clim_brier",
            "persist_pr_auc", "persist_pr_auc_9x9", "persist_E", "persist_logbias", "persist_brier",
            "persist_skill")
SKILL_COLS = ("pr_auc_skill", "pr_auc_skill_9x9", "pr_auc_skill_persist")


def reference_scores(labels, bounds, train, year, clim):
    """No-skill references on the same chips + labels as the model.

    clim     pixel-wise train-year burn frequency (+ 9x9 box mean as a secondary)
    persist  last year's burned mask, label_{year-1}. Legitimate at the January issue
             date (every driver is already Y-1), so for the t+1 eval year of a fold it is
             the t label even though t is held out from training.
    Each gets PR-AUC, expected burned px E (the reference summed as a probability),
    log(E/A) and Brier; persist_skill = persistence PR-AUC skill vs climatology.
    Returns (columns, per-chip E arrays for the pair metrics).
    """
    n = labels.shape[0]
    lab = labels.reshape(-1)
    a = float(labels.sum())
    out, chip = {k: float("nan") for k in REF_COLS}, {}
    refs = {"clim": lambda k: clim.chips(train, bounds, kernel=k)}
    if os.path.exists(os.path.join(clim.label_dir, f"label_{year - 1}.tif")):
        refs["persist"] = lambda k: clim.persistence(year - 1, bounds, kernel=k)
    for name, get in refs.items():
        r = get(CLIM_KERNEL_PIXELWISE)
        e_chip = r.reshape(n, -1).sum(axis=1).astype(np.float64)
        e = float(e_chip.sum())
        out[f"{name}_pr_auc"] = _pr_auc(lab, r.reshape(-1))
        out[f"{name}_pr_auc_9x9"] = _pr_auc(lab, get(CLIM_KERNEL_9X9).reshape(-1))
        out[f"{name}_E"] = e
        out[f"{name}_logbias"] = math.log(e / a) if e > 0 and a > 0 else float("nan")
        out[f"{name}_brier"] = float(((r - labels) ** 2).mean())
        chip[f"E_{name}"] = e_chip
    out["persist_skill"] = pr_auc_skill(out["persist_pr_auc"], out["clim_pr_auc"])
    return out, chip


def add_skills(out):
    """Model PR-AUC skill vs each reference (primary = pixel-wise climatology)."""
    out["pr_auc_skill"] = pr_auc_skill(out["pr_auc"], out["clim_pr_auc"])
    out["pr_auc_skill_9x9"] = pr_auc_skill(out["pr_auc"], out["clim_pr_auc_9x9"])
    out["pr_auc_skill_persist"] = pr_auc_skill(out["pr_auc"], out["persist_pr_auc"])


def score_protocol_year(row, year, base_row, clim, fetch=False):
    """One cached row of protocol_scores.csv (+ per-chip totals npz)."""
    pred_dir = os.path.join(REPO, row["predict_root"], str(year))
    keys, bounds, scores, labels = load_year_chips(pred_dir)
    n, hw = scores.shape[0], scores.shape[1] * scores.shape[2]
    m = decompose(scores.reshape(-1), labels.reshape(-1), np.repeat(np.arange(n, dtype=np.int32), hw))
    w = effective_pos_weight(row)
    train = [int(y) for y in row["train_years"].split(";")]
    out = {"arch": row["arch"], "fold_id": row["fold_id"], "stage": row["stage"],
           "base_fold": row["base_fold"], "probe": row["probe"], "year": year,
           "lead": year - max(train), "n_train_years": len(train),
           **{k: m[k] for k in ("pr_auc", "within_chip_only", "chip_r", "oracle_chip_only", "prevalence")},
           "n_chips": n}
    tot = year_totals(scores, labels, w)
    out.update(A=tot["A"], E_defl=tot["E"], logbias=tot["logbias"], brier=tot["brier"])
    chip = {"keys": np.array(keys), "A": tot["chip_A"], "E": tot["chip_E"]}

    g = gamma_table(row["gamma_local"])
    for tag in ("nogamma", "basegamma"):
        for k in ("E_defl", "logbias", "brier"):
            out[f"{k}_{tag}"] = float("nan")
    out["gamma_t"] = g[year] if g else 0.0
    if g:
        t = year_totals(scores, labels, w, g_from=g[year], g_to=0.0)
        out.update(E_defl_nogamma=t["E"], logbias_nogamma=t["logbias"], brier_nogamma=t["brier"])
        chip["E_nogamma"] = t["chip_E"]
    gb = gamma_table(base_row["gamma_local"]) if base_row is not None else None
    out["base_gamma_t"] = gb[year] if gb else float("nan")
    if g and gb and row["stage"] in ("ablateA", "ablateB"):
        t = year_totals(scores, labels, w, g_from=g[year], g_to=gb[year])
        out.update(E_defl_basegamma=t["E"], logbias_basegamma=t["logbias"], brier_basegamma=t["brier"])
        chip["E_basegamma"] = t["chip_E"]

    # References: pixel-wise climatology is the selection axis; 9x9 and persistence reported.
    out.update({k: float("nan") for k in REF_COLS})
    if clim is not None and row["stage"] in ("folds", "final"):
        ref, ref_chip = reference_scores(labels, bounds, train, year, clim)
        out.update(ref)
        chip.update(ref_chip)
    add_skills(out)
    out.update(training_summary(row, fetch))

    sub = "" if row["arch"] == "factored_v1" else f"{row['arch']}/"
    npz = os.path.join(CHIP_SCORES_DIR, f"{sub}{row['fold_id']}", f"{year}.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    np.savez_compressed(npz, **chip)
    return out


def backfill_references(cache, rows, clim):
    """Fill reference columns on cached rows scored before they existed (no model rescore)."""
    by_key = {(r["arch"], r["fold_id"]): r for r in rows}
    for c in REF_COLS + SKILL_COLS:
        if c not in cache.columns:
            cache[c] = float("nan")
    todo = cache.stage.isin(("folds", "final")) & cache.persist_pr_auc.isna()
    n_done = 0
    for i in cache.index[todo]:
        r = by_key.get((cache.at[i, "arch"], cache.at[i, "fold_id"]))
        y = int(cache.at[i, "year"])
        if r is None or not os.path.exists(os.path.join(clim.label_dir, f"label_{y - 1}.tif")):
            continue
        print(f"[refs] {r['arch']}/{r['fold_id']} {y}", flush=True)
        _, bounds, _, labels = load_year_chips(os.path.join(REPO, r["predict_root"], str(y)))
        ref, ref_chip = reference_scores(labels, bounds, [int(t) for t in r["train_years"].split(";")], y, clim)
        rec = {**cache.loc[i].to_dict(), **ref}
        add_skills(rec)
        for k in REF_COLS + SKILL_COLS:
            cache.at[i, k] = rec[k]
        sub = "" if r["arch"] == "factored_v1" else f"{r['arch']}/"
        npz = os.path.join(CHIP_SCORES_DIR, f"{sub}{r['fold_id']}", f"{y}.npz")
        if os.path.exists(npz):
            np.savez_compressed(npz, **{**dict(np.load(npz)), **ref_chip})
        n_done += 1
    return n_done


def load_chip_scores(row, year):
    sub = "" if row["arch"] == "factored_v1" else f"{row['arch']}/"
    return dict(np.load(os.path.join(CHIP_SCORES_DIR, f"{sub}{row['fold_id']}", f"{year}.npz")))


def score_protocol(report, archs=None, rescore=False, fetch=False, label_dir=LABEL_DIR):
    import cv_make_folds as mk
    rows = list(csv.DictReader(open(PROTOCOL)))
    if archs:
        rows = [r for r in rows if r["arch"] in archs]
    by_key = {(r["arch"], r["fold_id"]): r for r in rows}
    cache = pd.read_csv(SCORES) if os.path.exists(SCORES) else pd.DataFrame()
    done = set() if rescore or cache.empty else set(zip(cache.arch, cache.fold_id, cache.year))
    clim = Climatology(label_dir) if os.path.isdir(label_dir) else None
    new, pending, blocked = [], [], set()
    for r in rows:
        if r["stage"] not in REPORT_STAGES[report]:
            continue
        gate = mk.gate_errors(r["stage"], r["arch"])
        if gate:
            blocked.add(f"{r['arch']}/{r['stage']}: {gate[0]}")
            continue
        for y in mk.parse_years(r["eval_years"]):
            if (r["arch"], r["fold_id"], y) in done:
                continue
            if not glob.glob(os.path.join(REPO, r["predict_root"], str(y), "**", "out_*.tif"), recursive=True):
                pending.append(f"{r['arch']}/{r['fold_id']}/{y}")
                continue
            print(f"[score] {r['arch']}/{r['fold_id']} {y}", flush=True)
            base = by_key.get((r["arch"], r["base_fold"])) if r["base_fold"] else None
            new.append(score_protocol_year(r, y, base, clim, fetch))
    for b in sorted(blocked):
        print(f"[gate] skipped {b}")
    if pending:
        print(f"[pending] no predictions yet ({len(pending)}): {' '.join(pending)}")
    n_refs = backfill_references(cache, rows, clim) if clim is not None and not cache.empty else 0
    if new:
        fresh = pd.DataFrame(new)
        if not cache.empty:
            keep = ~cache.set_index(["arch", "fold_id", "year"]).index.isin(
                fresh.set_index(["arch", "fold_id", "year"]).index)
            cache = pd.concat([cache[keep], fresh], ignore_index=True)
        else:
            cache = fresh
    if new or n_refs:
        cache = cache.sort_values(["arch", "stage", "fold_id", "year"])
        cache.to_csv(SCORES, index=False)
    return cache, rows


def pairs_table(scores, rows):
    """One row per (arch, job) with both eval years scored: the amplitude measurement."""
    by_key = {(r["arch"], r["fold_id"]): r for r in rows}
    out = []
    for (arch, fid), g in scores.groupby(["arch", "fold_id"]):
        if (arch, fid) not in by_key or len(g) < 2:
            continue
        g = g.sort_values("year")
        t, t1 = g.iloc[0], g.iloc[1]
        r = by_key[(arch, fid)]
        c0, c1 = load_chip_scores(r, int(t.year)), load_chip_scores(r, int(t1.year))
        rec = {"arch": arch, "fold_id": fid, "stage": t.stage, "years": f"{int(t.year)}-{int(t1.year)}"}
        rec.update(pair_metrics(t.A, t1.A, t.E_defl, t1.E_defl, c0["A"], c1["A"], c0["E"], c1["E"]))
        if "E_nogamma" in c0:
            net = pair_metrics(t.A, t1.A, t.E_defl_nogamma, t1.E_defl_nogamma,
                               c0["A"], c1["A"], c0["E_nogamma"], c1["E_nogamma"])
            rec.update({f"{k}_nogamma": net[k] for k in ("R_E", "log_ratio_err", "amp_frac", "r_dE_dA")})
        out.append(rec)
        for name in ("clim", "persist"):        # references on this arch's chips + labels
            e0, e1 = t.get(f"{name}_E", float("nan")), t1.get(f"{name}_E", float("nan"))
            if f"E_{name}" not in c0 or f"E_{name}" not in c1 or not (e0 > 0 and e1 > 0):
                continue
            ref = {"arch": arch, "fold_id": fid, "stage": t.stage, "years": rec["years"], "ref": name}
            ref.update(pair_metrics(t.A, t1.A, e0, e1, c0["A"], c1["A"], c0[f"E_{name}"], c1[f"E_{name}"]))
            out.append(ref)
    df = pd.DataFrame(out)
    if not df.empty:
        df["ref"] = df["ref"].fillna("") if "ref" in df else ""
    return df


SKILL_TOL, AMP_TOL, N_FOLDS = 0.02, 0.10, 5      # notes/cv_preregistration.md section 5


def selection_verdicts(scores, pairs, ref):
    """The pre-registered winner rule, challenger vs reference, on fold scores + pairs."""
    s = scores[scores.stage == "folds"]
    r = s[s.arch == ref]
    out = []
    for arch in sorted(set(s.arch) - {ref}):
        m = s[s.arch == arch].merge(r, on=["fold_id", "year"], suffixes=("", "_ref"))
        if m.empty:
            continue
        d = m.pr_auc_skill - m.pr_auc_skill_ref
        fold_d = d.groupby(m.fold_id).mean()
        largest = m.loc[m.n_train_years.idxmax(), "fold_id"]
        pm = pd.DataFrame()
        if not pairs.empty:
            mp = pairs[pairs.ref == ""]         # model pairs only, not the references
            pm = mp[(mp.arch == arch) & mp.turn].merge(
                mp[(mp.arch == ref) & mp.turn], on="fold_id", suffixes=("", "_ref"))
        pair_d = (pm.log_ratio_err.abs() - pm.log_ratio_err_ref.abs()) if len(pm) else pd.Series(dtype=float)
        d_a = float(d.mean())
        d_b = float(pair_d.mean()) if len(pair_d) else float("nan")
        t = m.drop_duplicates("fold_id")
        d_last3, d_best = float((t.last3 - t.last3_ref).mean()), float((t.best - t.best_ref).mean())
        wins_a = d_a > SKILL_TOL and not d_b > AMP_TOL
        wins_b = d_b < -AMP_TOL and not d_a < -SKILL_TOL
        cons_a = fold_d.get(largest, 0.0) > 0 and int((fold_d > 0).sum()) >= 3
        cons_b = len(pair_d) >= 2 and bool((pair_d < 0).all())
        robust = bool(np.sign(d_last3) == np.sign(d_best)) if not (math.isnan(d_last3) or math.isnan(d_best)) else False
        complete = fold_d.size == N_FOLDS
        beats = ((wins_a and cons_a) or (wins_b and cons_b)) and robust
        out.append({"arch": arch, "ref": ref, "folds": int(fold_d.size), "d_skill": d_a, "d_abs_lre": d_b,
                    "folds_better": int((fold_d > 0).sum()), "largest_fold_d": float(fold_d.get(largest, float("nan"))),
                    "d_last3": d_last3, "d_best": d_best, "robust": robust,
                    "verdict": ("beats reference" if beats else "does not beat reference")
                               + ("" if complete else f" (INCOMPLETE: {fold_d.size}/{N_FOLDS} folds)")})
    return pd.DataFrame(out)


def _fmt(v, spec=".4f"):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def write_protocol_report(report, scores, rows, ref_arch="factored_v1"):
    stages = ("folds",) if report == "folds" else ("final",)
    s = scores[scores.stage.isin(stages)].copy()
    if s.empty:
        print(f"[report] nothing scored for {report} yet")
        return
    pairs = pairs_table(s, rows)
    pairs.to_csv(os.path.join(OUT_DIR, f"protocol_{report}_pairs.csv"), index=False)

    L = [f"# Protocol report: {report}", "",
         f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} by cv_collect_results.py. "
         "Selection rule: notes/cv_preregistration.md.", ""]
    L += ["## Per year", "",
          "skill = vs pixel-wise climatology (selection axis); skill_p = vs last-year persistence.", "",
          "| arch | job | year | lead | n_train | PR-AUC | clim | persist | skill | skill_p | within_chip | chip_r | log(E/A) | log(E/A) no γ | Brier | best | last3 | TRUNC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.sort_values(["arch", "fold_id", "year"]).iterrows():
        L.append(f"| {r.arch} | {r.fold_id} | {int(r.year)} | {int(r.lead)} | {int(r.n_train_years)} | "
                 f"{_fmt(r.pr_auc)} | {_fmt(r.clim_pr_auc)} | {_fmt(r.get('persist_pr_auc', float('nan')))} | "
                 f"{_fmt(r.pr_auc_skill)} | {_fmt(r.get('pr_auc_skill_persist', float('nan')))} | "
                 f"{_fmt(r.within_chip_only)} | {_fmt(r.chip_r, '.3f')} | {_fmt(r.logbias, '+.3f')} | "
                 f"{_fmt(r.logbias_nogamma, '+.3f')} | {_fmt(r.brier, '.5f')} | {_fmt(r.best)} | "
                 f"{_fmt(r.last3)} | {'yes' if str(r.truncated) == 'True' else ''} |")
    refs = s.drop_duplicates(["arch", "fold_id", "year"])
    if "persist_pr_auc" in refs and refs[list(REF_COLS)].notna().any().any():
        L += ["", "## References (on each arch's chips + labels)", "",
              "clim = pixel-wise train-year burn frequency; persist = last year's burned mask "
              "(label_{year-1}). E = the reference summed as a probability.", "",
              "| arch | job | year | clim PR-AUC | clim 9x9 | clim log(E/A) | clim Brier | "
              "persist PR-AUC | persist 9x9 | persist log(E/A) | persist Brier | persist skill vs clim |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in refs.sort_values(["arch", "fold_id", "year"]).iterrows():
            L.append(f"| {r.arch} | {r.fold_id} | {int(r.year)} | {_fmt(r.clim_pr_auc)} | {_fmt(r.clim_pr_auc_9x9)} | "
                     f"{_fmt(r.clim_logbias, '+.3f')} | {_fmt(r.clim_brier, '.5f')} | {_fmt(r.persist_pr_auc)} | "
                     f"{_fmt(r.persist_pr_auc_9x9)} | {_fmt(r.persist_logbias, '+.3f')} | "
                     f"{_fmt(r.persist_brier, '.5f')} | {_fmt(r.persist_skill, '+.4f')} |")
    ref_pairs = pairs[pairs.ref != ""] if not pairs.empty else pairs
    pairs = pairs[pairs.ref == ""] if not pairs.empty else pairs
    if not pairs.empty:
        L += ["", "## Two-year amplitude (same model on both years)", "",
              f"Turn = |log A ratio| ≥ log 1.25; amp_frac only on turns (1 = perfect, 0 = flat, <0 = inverted).", "",
              "| arch | job | years | R_A | R_E | log-ratio err | amp_frac | r(dE,dA) | R_E no γ | amp_frac no γ |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, p in pairs.sort_values(["arch", "fold_id"]).iterrows():
            L.append(f"| {p.arch} | {p.fold_id} | {p.years} | {p.R_A:.3f} | {p.R_E:.3f} | "
                     f"{p.log_ratio_err:+.3f} | {_fmt(p.amp_frac, '+.2f')} | {_fmt(p.r_dE_dA, '.3f')} | "
                     f"{_fmt(p.get('R_E_nogamma', float('nan')), '.3f')} | "
                     f"{_fmt(p.get('amp_frac_nogamma', float('nan')), '+.2f')} |")
    if not ref_pairs.empty:
        L += ["", "Reference amplitude: climatology is flat by construction (R_E = 1, amp_frac 0); "
              "persistence repeats last year's change (R_E = A(t)/A(t-1)).", "",
              "| arch | job | reference | years | R_A | R_E | log-ratio err | amp_frac | r(dE,dA) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, p in ref_pairs.sort_values(["arch", "fold_id", "ref"]).iterrows():
            L.append(f"| {p.arch} | {p.fold_id} | {p.ref} | {p.years} | {p.R_A:.3f} | {p.R_E:.3f} | "
                     f"{p.log_ratio_err:+.3f} | {_fmt(p.amp_frac, '+.2f')} | {_fmt(p.r_dE_dA, '.3f')} |")

    # selection axes per arch
    L += ["", "## Selection axes", "",
          "| arch | years scored | mean skill (a) | mean \\|log-ratio err\\| on turns (b) | mean last3 | TRUNC runs |",
          "|---|---|---|---|---|---|"]
    for arch, g in s.groupby("arch"):
        pa = pairs[(pairs.arch == arch) & pairs.turn] if not pairs.empty else pairs
        trunc = g.drop_duplicates("fold_id").truncated.astype(str).eq("True").sum()
        L.append(f"| {arch} | {len(g)} | {_fmt(g.pr_auc_skill.mean())} | "
                 f"{_fmt(pa.log_ratio_err.abs().mean() if len(pa) else float('nan'), '.3f')} | "
                 f"{_fmt(g.drop_duplicates('fold_id').last3.mean())} | {trunc} |")
    # the no-skill references on the same fold-years (per arch: v2/v3 chip sets differ)
    for arch, g in s.groupby("arch"):
        g = g.drop_duplicates(["fold_id", "year"])
        for name, label in (("clim", "climatology"), ("persist", "persistence")):
            if f"{name}_pr_auc" not in g or g[f"{name}_pr_auc"].isna().all():
                continue
            skill = 0.0 if name == "clim" else g.persist_skill.mean()
            pa = (ref_pairs[(ref_pairs.arch == arch) & (ref_pairs.ref == name) & ref_pairs.turn]
                  if not ref_pairs.empty else ref_pairs)
            L.append(f"| *ref: {label}* ({arch} chips) | {int(g[f'{name}_pr_auc'].notna().sum())} | "
                     f"{_fmt(skill)} | {_fmt(pa.log_ratio_err.abs().mean() if len(pa) else float('nan'), '.3f')} | — | — |")

    # the pre-registered winner rule, then the paired differences behind it
    if report == "folds" and ref_arch in set(s.arch) and len(set(s.arch)) > 1:
        v = selection_verdicts(s, pairs, ref_arch)
        L += ["", f"## Selection rule vs {ref_arch} (notes/cv_preregistration.md §5)", "",
              f"Beats = [Δskill > {SKILL_TOL} and Δ|lre| ≤ {AMP_TOL}, with fwdpair_2022 + ≥3/5 folds agreeing] or "
              f"[Δ|lre| < −{AMP_TOL} and Δskill ≥ −{SKILL_TOL}, both turn pairs agreeing], and Δlast3 has the sign of Δbest.", "",
              "| arch | folds | Δ skill (a) | Δ \\|log-ratio err\\| (b) | folds better | largest-fold Δ | Δ last3 | Δ best | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, x in v.iterrows():
            L.append(f"| {x.arch} | {x.folds} | {x.d_skill:+.4f} | {_fmt(x.d_abs_lre, '+.3f')} | {x.folds_better} | "
                     f"{_fmt(x.largest_fold_d, '+.4f')} | {_fmt(x.d_last3, '+.4f')} | {_fmt(x.d_best, '+.4f')} | {x.verdict} |")
        ref = s[s.arch == ref_arch].set_index(["fold_id", "year"])
        L += ["", f"## Paired differences vs {ref_arch} (per fold-year, skill)", "",
              "| arch | fold | year | Δ skill | Δ within_chip |", "|---|---|---|---|---|"]
        for arch, g in s[s.arch != ref_arch].groupby("arch"):
            for _, r in g.sort_values(["fold_id", "year"]).iterrows():
                if (r.fold_id, r.year) in ref.index:
                    b = ref.loc[(r.fold_id, r.year)]
                    L.append(f"| {arch} | {r.fold_id} | {int(r.year)} | {r.pr_auc_skill - b.pr_auc_skill:+.4f} | "
                             f"{r.within_chip_only - b.within_chip_only:+.4f} |")
    path = os.path.join(OUT_DIR, f"protocol_{report}.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] wrote {os.path.relpath(path, REPO)} (+ _pairs.csv)")

    if report == "final":
        import cv_make_folds as mk
        prior = json.load(open(mk.FINAL_SCORED)) if os.path.exists(mk.FINAL_SCORED) else None
        if prior:
            print(f"[final] already scored at {prior['scored_at']}; rewriting from the same predictions")
        doc = {"scored_at": prior["scored_at"] if prior else datetime.datetime.now().isoformat(timespec="seconds"),
               "rescored_at": datetime.datetime.now().isoformat(timespec="seconds") if prior else None,
               "selection": json.load(open(mk.SELECTION_FROZEN)),
               "per_year": s[[c for c in ("arch", "fold_id", "year", "pr_auc", "pr_auc_skill", "pr_auc_skill_9x9",
                                          "pr_auc_skill_persist", "within_chip_only", "chip_r", "logbias", "brier",
                                          "clim_pr_auc", "persist_pr_auc", "clim_brier", "persist_brier")
                              if c in s]].to_dict(orient="records"),
               "pairs": pairs.to_dict(orient="records") if not pairs.empty else []}
        with open(mk.FINAL_SCORED, "w") as f:
            json.dump(doc, f, indent=2, default=float)
        print(f"[final] wrote {os.path.relpath(mk.FINAL_SCORED, REPO)} -- ablation gate is now open")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=sorted(REPORT_STAGES), required=True)
    ap.add_argument("--arch", nargs="*", default=None, help="limit protocol scoring to these archs")
    ap.add_argument("--ref_arch", default="factored_v3p_union4_monthlyattn_wide_yeargain")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--fetch_train_csv", action="store_true",
                    help="gsutil cp missing training CSVs next to their checkpoints")
    ap.add_argument("--label_dir", default=LABEL_DIR)
    args = ap.parse_args()

    import cv_make_folds as mk
    if args.protocol == "final" and mk.gate_errors("final", args.ref_arch):
        sys.exit("GATE CLOSED: " + mk.gate_errors("final", args.ref_arch)[0])
    scores, rows = score_protocol(args.protocol, args.arch, args.rescore, args.fetch_train_csv,
                                  args.label_dir)
    if args.protocol != "ablate" and not scores.empty:
        write_protocol_report(args.protocol, scores, rows, args.ref_arch)


if __name__ == "__main__":
    main()
