"""Fit the frozen global year-intensity offset `gamma(t)` used by the factored model.

`gamma(t)` is a low-degree-of-freedom, GLOBAL, offline-fit additive logit offset:

    gamma(t) = b0 + b_soi * z(SOI_{Aug-Oct, Y-1}) + b_prev * z(log basin burn_{Y-1})

Why offline and low-dof: the network sees ~360 year-constant scalars against 10-13
distinct year-values, so a learned year head fits without fully identifying the year
Fitting <=3 parameters offline on the chip-year panel is well-conditioned;
letting the network do it is not.

Why global rather than spatially varying: a mean-zero spatial basis multiplied by a
year scalar has zero basin mean, so it is ORTHOGONAL to the year effect and cannot
change year-to-year amplitude at all. `--check` pins this as an invariant.

Why the prev-burn term is safe: it fits NEGATIVE (mean-reverting -- a big burn year
predicts a smaller next year), so it does not lag the turns the way a persistence
term would. `--check` pins the sign, and the fit refuses to emit if it flips.

The emitted offsets are MEAN-CENTERED over the fit years. Under weighted BCE the
model's optimal output is inflated (logit(q) ~= logit(p) + log(pos_weight) for small
p), so the level is absorbed by the network's own bias while the year-to-year
component passes through unchanged. Centering makes
gamma independent of `pos_weight`.

Usage:
    .venv/bin/python scripts/analysis/fit_year_offset.py --check
    .venv/bin/python scripts/analysis/fit_year_offset.py \
        --fit_years 2013-2022 --out out/gamma_v1.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PANEL = os.path.join(os.path.dirname(__file__), "..", "..", "out", "chip_panel", "panel.parquet")

# Targets are summed over their columns. "union_sum" approximates the actual
# training label (im_BurnDate OR im_viirs_snpp) by summing the two sensors; that
# over-counts their overlap, but the overlap is a near-constant ~15-22% of the
# union, so in log space it shifts the level and barely touches the YEAR effect,
# which is all gamma uses.
TARGETS = {"bd": ["burn_bd"], "snpp": ["burn_snpp"], "ft": ["burn_ft"],
           "union_sum": ["burn_bd", "burn_snpp"],
           "union3": ["burn_bd", "burn_snpp", "burn_mod14"]}
PREV_BANDS = {"bd": ["im_BurnDate_-1_mean"],
              "snpp": ["im_viirs_snpp_-1_mean"],
              "union_sum": ["im_BurnDate_-1_mean", "im_viirs_snpp_-1_mean"],
              "union3": ["im_BurnDate_-1_mean", "im_viirs_snpp_-1_mean", "im_mod14_-1_mean"]}
SOI_COL = "md_soi_y1ond"
CHIP_PIXELS = 128 * 128


def _z(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std()


def load_panel(path=PANEL, target="bd", prev_burn="union_sum", space="log1p"):
    """Chip-year frame with the response and the two year-level regressors.

    `prev_burn="union_sum"` sums the two sensors' previous-year bands. That
    over-counts their overlap (the sensors share only 15-22% of the union), but the
    scalar is z-scored per year, so a roughly proportional bias cancels. Use
    `--check` / `--sensitivity` to confirm the choice does not matter.
    """
    d = pd.read_parquet(path)
    cols = ["md_id", "year", "md_x", "md_y", SOI_COL] + list(TARGETS[target])
    cols += [c for c in PREV_BANDS[prev_burn] if c not in cols]
    d = d[list(dict.fromkeys(cols))].dropna().copy()

    burn = d[list(TARGETS[target])].sum(axis=1).to_numpy(dtype=float)
    if space == "log1p":
        d["y"] = np.log1p(burn)
    elif space == "logit":                      # correct space for an additive logit offset
        denom = CHIP_PIXELS * len(TARGETS[target])
        p = (burn + 0.5) / (denom + 1.0)
        d["y"] = np.log(p / (1.0 - p))
    else:
        raise ValueError(f"unknown space {space!r}")

    # Year-level regressors. Both are one scalar per year, broadcast to every chip.
    soi = d.groupby("year")[SOI_COL].first()
    d["zsoi"] = d.year.map((soi - soi.mean()) / soi.std())
    prev = sum(d.groupby("year")[c].mean() for c in PREV_BANDS[prev_burn])
    lprev = np.log1p(prev)
    d["zprev"] = d.year.map((lprev - lprev.mean()) / lprev.std())

    d["zlat"] = _z(d.md_y)
    d["zlon"] = _z(d.md_x)
    d["zne"] = _z(d.md_y + d.md_x)              # NE-SW axis
    return d


def _design(frame, terms):
    cols = [np.ones(len(frame))]
    for a, b in terms:
        cols.append(frame[a].to_numpy() * frame[b].to_numpy() if b else frame[a].to_numpy())
    return np.column_stack(cols)


def evaluate(d, terms, protocol="loyo", exclude_prev_from_clim=False, min_train_years=5):
    """Out-of-sample fit of the year effect.

    Chip climatology is always computed from TRAINING years only -- an in-sample
    climatology leaks the held-out year straight into the residual. With
    `exclude_prev_from_clim` the year t-1 is dropped too, which is what separates a
    genuine mean-reverting prev-burn signal from a climatology artifact (a high t-1
    raises the climatology and mechanically depresses the t residual).
    """
    years = sorted(d.year.unique())
    pred_year, act_year, betas = {}, {}, []
    for i, t in enumerate(years):
        if protocol == "forward" and i < min_train_years:
            continue
        drop = {t} | ({t - 1} if exclude_prev_from_clim else set())
        tr_years = [y for y in years if y not in drop and (protocol != "forward" or y < t)]
        tr, te = d[d.year.isin(tr_years)], d[d.year == t]

        clim = tr.groupby("md_id")["y"].mean()
        tr_res = (tr.y - tr.md_id.map(clim)).to_numpy()
        te = te[te.md_id.isin(clim.index)]
        te_res = (te.y - te.md_id.map(clim)).to_numpy()

        beta, *_ = np.linalg.lstsq(_design(tr, terms), tr_res, rcond=None)
        betas.append(beta)
        pred_year[t] = float((_design(te, terms) @ beta).mean())
        act_year[t] = float(te_res.mean())

    ks = sorted(pred_year)
    P = np.array([pred_year[k] for k in ks])
    A = np.array([act_year[k] for k in ks])
    turns = int(sum(np.sign(A[i] - A[i - 1]) == np.sign(P[i] - P[i - 1]) for i in range(1, len(ks))))
    B = np.array(betas)
    return {
        "years": ks,
        "pred_year": P, "act_year": A,
        "r_year": float(np.corrcoef(P, A)[0, 1]),
        "amplitude": float(np.exp(P.max() - P.min())),
        "actual_amplitude": float(np.exp(A.max() - A.min())),
        "turns": turns, "n_turns": len(ks) - 1,
        "beta_mean": B.mean(0), "beta_sd": B.std(0),
        "sign_consistent": bool(np.all(np.sign(B[:, 1:]) == np.sign(B[0, 1:]), axis=0).all()),
    }


def jackknife_r(result):
    """(min, max, most-influential-year) of r_year under leave-one-evaluated-year-out.

    With 8 evaluated years a single extreme year can carry the whole correlation --
    here 2024 does -- so the point estimate alone overstates the evidence.
    """
    P, A, years = result["pred_year"], result["act_year"], result["years"]
    rs = [(float(np.corrcoef(np.delete(P, i), np.delete(A, i))[0, 1]), int(years[i]))
          for i in range(len(years))]
    return min(r for r, _ in rs), max(r for r, _ in rs), min(rs)[1]


SOI = [("zsoi", None)]
PREV = [("zprev", None)]
BOTH = [("zsoi", None), ("zprev", None)]


def fit_final(d, fit_years, terms=BOTH):
    """Single in-sample fit over the training years -> the coefficients we ship."""
    tr = d[d.year.isin(fit_years)]
    clim = tr.groupby("md_id")["y"].mean()
    res = (tr.y - tr.md_id.map(clim)).to_numpy()
    beta, *_ = np.linalg.lstsq(_design(tr, terms), res, rcond=None)
    return beta, clim


def build_offsets(d, beta, terms=BOTH, center_years=None):
    """Per-year gamma, mean-centered over `center_years` (see module docstring)."""
    per_year = d.groupby("year").first().reset_index()
    vals = _design(per_year, terms) @ beta
    out = dict(zip(per_year.year.astype(int), vals))
    ref = center_years if center_years is not None else sorted(out)
    mu = float(np.mean([out[y] for y in ref if y in out]))
    return {int(y): float(v - mu) for y, v in out.items()}, mu


# --------------------------------------------------------------------------- checks
# Values measured 2026-09-04 on out/chip_panel/panel.parquet with
# target=bd, prev_burn=bd, space=log1p. These are the reference configuration for
# the regression checks; --sensitivity shows the other choices do not change the
# conclusions. Changing a number here means the fit changed -- find out why.
REF = dict(target="bd", prev_burn="bd", space="log1p")


def _chk(results, name, got, want, tol):
    ok = abs(got - want) <= tol
    results.append((ok, f"{name}: {got:+.4f} (expect {want:+.4f} +/- {tol})"))
    return ok


def run_checks(panel=PANEL, verbose=True):
    d = load_panel(panel, **REF)
    r = []

    loyo = {k: evaluate(d, t) for k, t in [("soi", SOI), ("prev", PREV), ("both", BOTH)]}
    _chk(r, "LOYO r_year SOI-only", loyo["soi"]["r_year"], 0.278, 0.01)
    _chk(r, "LOYO r_year prev-only", loyo["prev"]["r_year"], 0.422, 0.01)
    _chk(r, "LOYO r_year both", loyo["both"]["r_year"], 0.500, 0.01)

    b_soi_only = evaluate(d, SOI)
    _chk(r, "LOYO b_soi mean", b_soi_only["beta_mean"][1], -0.152, 0.005)
    _chk(r, "LOYO b_soi sd", b_soi_only["beta_sd"][1], 0.029, 0.005)
    r.append((b_soi_only["sign_consistent"], "LOYO b_soi sign consistent across all folds"))

    fwd = {k: evaluate(d, t, protocol="forward", exclude_prev_from_clim=True)
           for k, t in [("soi", SOI), ("both", BOTH)]}
    _chk(r, "forward-chained r_year both", fwd["both"]["r_year"], 0.775, 0.01)
    _chk(r, "forward-chained r_year SOI-only", fwd["soi"]["r_year"], 0.422, 0.01)
    r.append((fwd["both"]["turns"] == 6 and fwd["both"]["n_turns"] == 7,
              f"forward-chained turns both: {fwd['both']['turns']}/{fwd['both']['n_turns']} (expect 6/7)"))
    r.append((fwd["soi"]["turns"] == 5 and fwd["soi"]["n_turns"] == 7,
              f"forward-chained turns SOI-only: {fwd['soi']['turns']}/{fwd['soi']['n_turns']} (expect 5/7)"))

    # LEAK-FREE INVARIANT: the load-bearing check for keeping the prev-burn term at
    # all. If dropping year t-1 from the chip climatology moves b_prev, the negative
    # sign was a climatology artifact rather than a real mean-reverting signal.
    incl = evaluate(d, BOTH, exclude_prev_from_clim=False)["beta_mean"][2]
    excl = evaluate(d, BOTH, exclude_prev_from_clim=True)["beta_mean"][2]
    _chk(r, "b_prev with t-1 in climatology", incl, -0.1650, 0.005)
    _chk(r, "b_prev with t-1 excluded", excl, -0.1648, 0.005)
    r.append((abs(incl - excl) < 0.005,
              f"leak-free invariant: |b_prev shift| = {abs(incl - excl):.5f} < 0.005"))

    # SIGN GUARD: negative == mean-reverting (safe at turns). Positive == persistence,
    # which would lag every turn; refuse to ship such coefficients.
    r.append((excl < 0, f"sign guard: b_prev = {excl:+.4f} < 0 (mean-reverting, not persistence)"))

    # ORTHOGONALITY INVARIANT: this is what "keep gamma global" rests on. A mean-zero
    # spatial basis times a year scalar cannot move the year effect.
    base = evaluate(d, BOTH)
    for basis in ("zlat", "zlon", "zne"):
        alt = evaluate(d, BOTH + [("zsoi", basis)])
        same = (round(alt["r_year"], 3) == round(base["r_year"], 3)
                and round(alt["amplitude"], 3) == round(base["amplitude"], 3))
        r.append((same, f"orthogonality SOI x {basis}: r_year {alt['r_year']:.3f} vs "
                        f"{base['r_year']:.3f}, amp {alt['amplitude']:.3f} vs {base['amplitude']:.3f}"))

    if verbose:
        for ok, msg in r:
            print(f"  [{'ok ' if ok else 'FAIL'}] {msg}")
    n_bad = sum(1 for ok, _ in r if not ok)
    print(f"\n{len(r) - n_bad}/{len(r)} checks passed")
    return n_bad == 0


def run_sensitivity(panel=PANEL):
    print(f"{'target':<11}{'prev_burn':<12}{'space':<8}{'r_year':>9}{'b_soi':>9}{'b_prev':>9}{'turns':>8}")
    for target in ("bd", "snpp", "union_sum"):
        for prev in ("bd", "snpp", "union_sum"):
            for space in ("log1p", "logit"):
                d = load_panel(panel, target=target, prev_burn=prev, space=space)
                e = evaluate(d, BOTH, protocol="forward", exclude_prev_from_clim=True)
                turns = f"{e['turns']}/{e['n_turns']}"
                print(f"{target:<11}{prev:<12}{space:<8}{e['r_year']:>9.3f}"
                      f"{e['beta_mean'][1]:>9.4f}{e['beta_mean'][2]:>9.4f}{turns:>8}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", default=PANEL)
    p.add_argument("--check", action="store_true", help="run regression + invariant checks")
    p.add_argument("--sensitivity", action="store_true", help="sweep target / prev-burn / space")
    p.add_argument("--target", default="union_sum", choices=sorted(TARGETS))
    p.add_argument("--prev_burn", default="bd", choices=sorted(PREV_BANDS))
    p.add_argument("--space", default="logit", choices=["log1p", "logit"])
    p.add_argument("--fit_years", default="2013-2022", help="inclusive range, e.g. 2013-2022")
    p.add_argument("--out", default=None, help="write gamma JSON here")
    a = p.parse_args()

    if a.check:
        sys.exit(0 if run_checks(a.panel) else 1)
    if a.sensitivity:
        run_sensitivity(a.panel)
        return

    lo, hi = (int(v) for v in a.fit_years.split("-"))
    fit_years = list(range(lo, hi + 1))
    d = load_panel(a.panel, target=a.target, prev_burn=a.prev_burn, space=a.space)

    beta, _ = fit_final(d, fit_years)
    if beta[2] >= 0:
        sys.exit(f"REFUSING to emit: b_prev = {beta[2]:+.4f} >= 0. A positive prev-burn "
                 "coefficient is persistence, which lags every turn. Investigate before shipping.")

    fwd = evaluate(d, BOTH, protocol="forward", exclude_prev_from_clim=True)
    loyo = evaluate(d, BOTH, exclude_prev_from_clim=True)
    jack = jackknife_r(fwd)
    offsets, level = build_offsets(d, beta, center_years=fit_years)

    doc = {
        "version": "gamma_v1",
        "fit": {"target": a.target, "prev_burn": a.prev_burn, "space": a.space,
                "fit_years": fit_years, "protocol": "in-sample fit, forward-chained validation"},
        "terms": ["soi_y1ond", "log_basin_prev_burn"],
        "coeffs": {"b0": float(beta[0]), "b_soi": float(beta[1]), "b_prev": float(beta[2])},
        "centering": {"note": "offsets are mean-centered over fit_years; the level is absorbed "
                              "by the network bias, which makes gamma independent of pos_weight",
                      "removed_level": level},
        "per_year_offset": offsets,
        "validation": {
            # Forward-chained is the operational protocol, but it scores only the
            # last 8 years and those have a wider spread of true year effects
            # (sd 0.338 vs 0.215 before 2018), which flatters the correlation.
            # Quote r_year WITH the jackknife range and the all-years LOYO figure.
            "r_year_forward": fwd["r_year"],
            "r_year_forward_jackknife": {"min": jack[0], "max": jack[1],
                                         "most_influential_year": jack[2]},
            "r_year_loyo_all_years": loyo["r_year"],
            "n_eval_years": len(fwd["years"]),
            "n_transitions": fwd["n_turns"],
            "turns": f"{fwd['turns']}/{fwd['n_turns']}",
            "amplitude": fwd["amplitude"],
            "actual_amplitude": fwd["actual_amplitude"],
        },
    }
    print(json.dumps(doc["coeffs"], indent=2))
    print(f"forward-chained r_year={fwd['r_year']:.3f} "
          f"(jackknife {jack[0]:.3f}..{jack[1]:.3f}, {jack[2]} carries it)  "
          f"LOYO all 13 yrs={loyo['r_year']:.3f}")
    print(f"  {len(fwd['years'])} evaluated years, {fwd['n_turns']} transitions, "
          f"turns={fwd['turns']}/{fwd['n_turns']}  "
          f"amplitude={fwd['amplitude']:.2f}x (actual {fwd['actual_amplitude']:.2f}x)")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
