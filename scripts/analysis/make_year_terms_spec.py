"""Add a `year_terms` block to a Shapley driver spec from a gamma fit.

Splits the factored model's frozen year offset into its two regressor parts,
each centred on the gamma's own centre years (as fit_year_offset.build_offsets
centres their sum), so the parts add up to `per_year_offset` exactly:

    gamma(t) = b_soi * (zsoi_t - mean zsoi) + b_prev * (zprev_t - mean zprev)

and hands the SOI part to one driver and the previous-year-burn part to another
(eval/attribution.py `year_terms`). Refuses if the parts don't rebuild the gamma
table to 1e-9.

    .venv/bin/python scripts/analysis/make_year_terms_spec.py \
        --gamma out/cv/gamma/gamma_v3_patched_bd_2002_burn_final_all.json \
        --spec configs/attribution_drivers_v3p_yeargain.json \
        --out configs/attribution_drivers_v3p_yeargain_yearsplit.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_year_offset as fyo  # noqa: E402

PANEL = "out/target_panel/panel.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma", required=True, help="gamma JSON the checkpoint was trained with")
    ap.add_argument("--spec", required=True, help="driver spec to extend")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", default=PANEL, help="target panel the gamma was fit on")
    ap.add_argument("--soi_driver", default="climate_weather")
    ap.add_argument("--prev_driver", default="fire_history")
    a = ap.parse_args()

    g = json.load(open(a.gamma))
    fit = g["fit"]
    if fit.get("panel_kind") != "target" or fit.get("terms_used") != "soi+prev":
        raise SystemExit(f"expected a target-panel soi+prev gamma, got {fit}")
    d = fyo.load_target_panel(a.panel, target=fit["target"], space=fit["space"],
                              emit_through=max(g["emit_years"]))
    per_year = d.attrs.get("per_year")
    if per_year is None:
        per_year = d.groupby("year").first().reset_index()
    X = fyo._design(per_year, fyo.BOTH)                 # [1, zsoi, zprev]
    years = per_year.year.astype(int).to_numpy()
    centre = np.isin(years, g["centering"]["center_years"])
    soi = g["coeffs"]["b_soi"] * (X[:, 1] - X[centre, 1].mean())
    prev = g["coeffs"]["b_prev"] * (X[:, 2] - X[centre, 2].mean())

    off = g["per_year_offset"]
    soi_terms, prev_terms = {}, {}
    for i, y in enumerate(years):
        if str(y) not in off:
            continue
        err = abs(soi[i] + prev[i] - off[str(y)])
        if err > 1e-9:
            raise SystemExit(f"{y}: soi + prev = {soi[i] + prev[i]:.6f} != gamma "
                             f"{off[str(y)]:.6f} (|err| {err:.2e})")
        soi_terms[str(y)], prev_terms[str(y)] = float(soi[i]), float(prev[i])
    if set(soi_terms) != set(off):
        raise SystemExit(f"panel lacks gamma years {sorted(set(off) - set(soi_terms))}")

    spec = json.load(open(a.spec))
    for name in (a.soi_driver, a.prev_driver):
        if name not in spec["drivers"]:
            raise SystemExit(f"driver {name!r} not in {a.spec}")
    spec["year_terms"] = {a.soi_driver: soi_terms, a.prev_driver: prev_terms}
    spec["year_terms_source"] = {
        "gamma": os.path.basename(a.gamma), "panel": a.panel,
        "split": f"{a.soi_driver} <- b_soi*(zsoi - mean); {a.prev_driver} <- b_prev*(zprev - mean); "
                 f"centred on {g['centering']['center_years'][0]}-{g['centering']['center_years'][-1]}"}
    with open(a.out, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[year_terms] wrote {a.out}: {len(soi_terms)} years; e.g. "
          + ", ".join(f"{y} soi {soi_terms[y]:+.3f} prev {prev_terms[y]:+.3f}"
                      for y in ("2023", "2024", "2025") if y in soi_terms))


if __name__ == "__main__":
    main()
