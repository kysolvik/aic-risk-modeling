"""Fit the frozen global year offset gamma(t) (eval/year_offset.py) and write it as JSON.

Usage: fit_year_offset.py --check  |  fit_year_offset.py --fit_years 2013-2022 --out gamma.json"""

import argparse
import json
import os
import sys

from aic_risk_modeling.eval.year_offset import (  # noqa: F401
    BOTH, PREV_BANDS, SOI, TARGETS, build_offsets, evaluate, fit_final, jackknife_r,
    load_panel, load_target_panel, run_checks)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL = os.path.join(REPO, "out", "chip_panel", "panel.parquet")
TARGET_PANEL = os.path.join(REPO, "out", "target_panel", "panel.parquet")


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
    p.add_argument("--panel_kind", default="chip", choices=["chip", "target"],
                   help="chip = fullgrid chip panel; target = targets-only panel")
    p.add_argument("--center_years", default=None,
                   help="range to mean-center over (default fit_years)")
    p.add_argument("--weighting", default="equal", choices=["equal", "burn"],
                   help="chip weighting of the year effect")
    p.add_argument("--emit_through", type=int, default=None,
                   help="target panel: emit predict-only years up to this")
    a = p.parse_args()

    if a.check:
        sys.exit(0 if run_checks(a.panel) else 1)
    if a.sensitivity:
        run_sensitivity(a.panel)
        return

    lo, hi = (int(v) for v in a.fit_years.split("-"))
    fit_years = list(range(lo, hi + 1))
    if a.panel_kind == "target":
        panel = a.panel if a.panel != PANEL else TARGET_PANEL
        d = load_target_panel(panel, target=a.target, space=a.space, emit_through=a.emit_through)
        val = d[d.year <= hi]
    else:
        if a.emit_through is not None:
            sys.exit("--emit_through needs --panel_kind target")
        d = load_panel(a.panel, target=a.target, prev_burn=a.prev_burn, space=a.space)
        val = d
    if a.center_years:
        clo, chi = (int(v) for v in a.center_years.split("-"))
        center_years = list(range(clo, chi + 1))
    else:
        center_years = fit_years

    if a.weighting != "equal" and a.panel_kind != "target":
        sys.exit("--weighting burn needs --panel_kind target")
    beta, _ = fit_final(d, fit_years, weighting=a.weighting)
    if beta[2] >= 0:
        sys.exit(f"REFUSING to emit: b_prev = {beta[2]:+.4f} >= 0. A positive prev-burn "
                 "coefficient is persistence, which lags every turn. Investigate before shipping.")

    fwd = evaluate(val, BOTH, protocol="forward", exclude_prev_from_clim=True, weighting=a.weighting)
    loyo = evaluate(val, BOTH, exclude_prev_from_clim=True, weighting=a.weighting)
    jack = jackknife_r(fwd)
    offsets, level = build_offsets(d, beta, center_years=center_years)

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
            # Forward-chained scores only the last 8 (wider-spread) years; quote it with the jackknife range.
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
    if a.panel_kind == "target":
        doc["version"] = "gamma_long_v1" if a.weighting == "equal" else "gamma_long_burn_v1"
        doc["fit"].update(panel_kind="target", weighting=a.weighting, panel=os.path.relpath(panel), prev_burn="bd_count",
                          protocol="in-sample fit, forward-chained validation on fit_years only")
        doc["centering"].update(center_years=center_years,
                                note="offsets are mean-centered over center_years (the network's "
                                     "training years); the level is absorbed by the network bias")
        doc["emit_years"] = sorted(offsets)
    print(json.dumps(doc["coeffs"], indent=2))
    print(f"forward-chained r_year={fwd['r_year']:.3f} "
          f"(jackknife {jack[0]:.3f}..{jack[1]:.3f}, {jack[2]} carries it)  "
          f"LOYO all {len(loyo['years'])} yrs={loyo['r_year']:.3f}")
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
