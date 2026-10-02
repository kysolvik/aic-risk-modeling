#!/usr/bin/env python
"""Post-hoc year-sensitivity ablations for the selected architecture (see cv_make_folds.py).

Runs only after the write-once final test has been scored (out/cv/final_scored.json),
on rows already scored into out/cv/protocol_scores.csv by
`cv_collect_results.py --protocol ablate`.

Every quantity is a within-year paired difference, so year difficulty and prevalence
cancel: Delta_arm(y) = M_arm(y) - M_base(y). Each base is scored on one HIGH and one LOW
burn year (by panel BurnDate fraction), and the estimand is the contrast

    C_arm = Delta_arm(high year) - Delta_arm(low year)

  A  future probe   bases fwdpair_2018 (2018 low -> 2019 high), fwdpair_2020 (2020 high
                    -> 2021 low); arms add 2024 (high), 2025 (low) or 2023 (average,
                    placebo). All three add exactly one future year, so volume cancels.
                    H1  C(+high) > 0     H2  C(+low) < 0     key  C(+high) - C(+low) > 0
                    placebo C(+2023) ~ 0. Pooled over the two bases, which cross the
                    high year's lead (lead 2 in fwdpair_2018, lead 1 in fwdpair_2020).
  B  analog removal base = final (2024 high, 2025 low); arms drop the two highest, lowest
                    or middle past years (volume-matched).
                    prediction  C(-highs) < C(-lows)     key  C(-lows) - C(-highs) > 0
                    placebo C(-mids) ~ 0.

Metrics are oriented so + = better: -|log(E/A)| (year magnitude), -Brier, PR-AUC,
within-chip PR-AUC, chip r. For the magnitude metrics each Delta is also split into
  network  arm predictions re-expressed under the BASE's gamma (exact logit swap)
  gamma    total - network (what refitting gamma with the extra/removed years did)
Ranking metrics are (near-)invariant to a year-constant gamma, so they carry no split.

Noise floor: the seed replicate fwdpair_2020_s55 vs fwdpair_2020 (no data change).
sd(C) is taken as max(|C_seed|, sqrt(2) * rms Delta_seed) -- conservative, and a ONE
replicate estimate, so it is a floor to beat, not an inference. MDE = 2 sd, scaled by
1/sqrt(2) for the two-base pooled A contrasts and by sqrt(2) for key (difference) contrasts.
A hypothesis reads "supported" only if the sign is as predicted, |value| > MDE, and (pooled
A) both bases agree in sign.

Outputs: out/cv/year_sensitivity.{csv,md}, out/cv/fig_year_sensitivity.{png,pdf}

Usage: .venv/bin/python scripts/cross_validation/cv_year_sensitivity.py [--arch ARCH]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import cv_make_folds as mk

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

OUT_DIR = os.path.join(REPO, "out", "cv")
SCORES = os.path.join(OUT_DIR, "protocol_scores.csv")
PANEL = os.path.join(REPO, "out", "chip_panel", "panel.parquet")
CHIP_PIXELS = 128 * 128

# (key, axis label, column, column under the base's gamma, orientation)
METRICS = [
    ("magnitude", "−|Log(E/A)|", "logbias", "logbias_basegamma", "negabs"),
    ("brier", "−Brier", "brier", "brier_basegamma", "neg"),
    ("pr_auc", "PR-AUC", "pr_auc", None, "id"),
    ("within_chip", "Within-Chip PR-AUC", "within_chip_only", None, "id"),
    ("chip_r", "Chip r", "chip_r", None, "id"),
]
FIG_METRICS = ("magnitude", "brier", "pr_auc")
SERIES = {"total": "#2a78d6", "network": "#eb6834"}     # categorical slots 1-2
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3de", "#d9d8d2"


def burn_fraction(panel=PANEL):
    d = pd.read_parquet(panel, columns=["year", "burn_bd"])
    g = d.groupby("year")["burn_bd"].sum() / (d.groupby("year").size() * CHIP_PIXELS)
    return {int(y): float(v) for y, v in g.items()}


def orient(v, how):
    v = float(v)
    return -abs(v) if how == "negabs" else (-v if how == "neg" else v)


def ablation_contrasts(scores, arch, burn, stage):
    """Long table: one row per (arm, metric, component) with Delta(high), Delta(low), C."""
    s = scores[scores.arch == arch]
    idx = {(r["fold_id"], int(r["year"])): r for _, r in s.iterrows()}
    out = []
    for _, arm in s[s.stage == stage].drop_duplicates("fold_id").iterrows():
        years = sorted(y for f, y in idx if f == arm["fold_id"])
        base = arm["base_fold"]
        if len(years) != 2 or any((base, y) not in idx for y in years):
            continue
        hi, lo = sorted(years, key=lambda y: -burn[y])
        for key, _, col, bcol, how in METRICS:
            d = {}
            for comp, c in (("total", col), ("network", bcol)):
                if c is None or any(pd.isna(idx[(arm["fold_id"], y)][c]) for y in years):
                    continue
                d[comp] = [orient(idx[(arm["fold_id"], y)][c], how) - orient(idx[(base, y)][col], how)
                           for y in (hi, lo)]
            if "total" in d and "network" in d:
                d["gamma"] = [t - n for t, n in zip(d["total"], d["network"])]
            for comp, (dh, dl) in d.items():
                out.append({"stage": stage, "base_fold": base, "arm": arm["fold_id"],
                            "probe": arm["probe"], "metric": key, "component": comp,
                            "high_year": hi, "low_year": lo,
                            "delta_high": dh, "delta_low": dl, "contrast": dh - dl})
    return pd.DataFrame(out)


def noise_floor(scores, arch, burn):
    """metric -> sd of a single contrast, from the seed replicate (total component)."""
    rep = ablation_contrasts(scores.assign(stage=scores.stage.replace({"seedrep": "_rep"})),
                             arch, burn, "_rep")
    rep = rep[rep.component == "total"]
    out = {}
    for _, r in rep.iterrows():
        rms = math.sqrt((r.delta_high ** 2 + r.delta_low ** 2) / 2)
        out[r.metric] = {"c_seed": r.contrast, "rms_delta_seed": rms,
                         "sd_contrast": max(abs(r.contrast), math.sqrt(2) * rms)}
    return out


def probe_roles(burn, years=mk.PROBE_YEARS):
    """{year: 'high'|'low'|'placebo'} by burn fraction."""
    order = sorted(years, key=lambda y: burn[y])
    return {order[-1]: "high", order[0]: "low", **{y: "placebo" for y in order[1:-1]}}


def drop_roles(arms, burn):
    """{arm fold_id: 'highs'|'lows'|'mids'} by the mean burn of the dropped years."""
    mean = {a: np.mean([burn[int(y)] for y in p.lstrip("-").split(",")]) for a, p in arms.items()}
    order = sorted(mean, key=mean.get)
    return {order[-1]: "highs", order[0]: "lows", **{a: "mids" for a in order[1:-1]}}


def _verdict(value, mde, predicted_sign, agree=True):
    if any(map(lambda x: x is None or (isinstance(x, float) and math.isnan(x)), (value, mde))):
        return "no noise floor"
    if abs(value) <= mde:
        return "null (≤ MDE)"
    if predicted_sign == 0:
        return "placebo moved (> MDE)"
    if np.sign(value) != predicted_sign:
        return "opposite sign"
    return "supported" if agree else "sign ok, bases disagree"


def summarize(A, B, noise, burn):
    """Hypothesis table: one row per (analysis, test, metric, component)."""
    rows = []
    if not A.empty:
        roles = probe_roles(burn)
        A = A.assign(role=A.probe.str.lstrip("+").astype(int).map(roles))
        for (metric, comp), g in A.groupby(["metric", "component"]):
            sd = noise.get(metric, {}).get("sd_contrast", float("nan"))
            per = g.groupby("role").contrast.agg(["mean", "count", "min", "max"])
            agree = {r: bool(per.loc[r, "min"] * per.loc[r, "max"] > 0) for r in per.index}
            for role, sign, label in (("high", 1, "H1  C(+high) > 0"), ("low", -1, "H2  C(+low) < 0"),
                                      ("placebo", 0, "placebo  C(+avg) ~ 0")):
                if role in per.index:
                    n = int(per.loc[role, "count"])
                    mde = 2 * sd / math.sqrt(n)
                    rows.append({"analysis": "A", "test": label, "metric": metric, "component": comp,
                                 "value": per.loc[role, "mean"], "n_bases": n, "mde": mde,
                                 "verdict": _verdict(per.loc[role, "mean"], mde, sign, agree[role])})
            if {"high", "low"} <= set(per.index):
                n = int(min(per.loc["high", "count"], per.loc["low", "count"]))
                k = per.loc["high", "mean"] - per.loc["low", "mean"]
                mde = 2 * sd * math.sqrt(2) / math.sqrt(n)
                rows.append({"analysis": "A", "test": "key  C(+high) − C(+low) > 0", "metric": metric,
                             "component": comp, "value": k, "n_bases": n, "mde": mde,
                             "verdict": _verdict(k, mde, 1)})
    if not B.empty:
        roles = drop_roles(dict(B.drop_duplicates("arm")[["arm", "probe"]].values), burn)
        B = B.assign(role=B.arm.map(roles))
        for (metric, comp), g in B.groupby(["metric", "component"]):
            sd = noise.get(metric, {}).get("sd_contrast", float("nan"))
            c = g.set_index("role").contrast
            if {"highs", "lows"} <= set(c.index):
                k = c["lows"] - c["highs"]
                rows.append({"analysis": "B", "test": "key  C(−lows) − C(−highs) > 0", "metric": metric,
                             "component": comp, "value": k, "n_bases": 1, "mde": 2 * sd * math.sqrt(2),
                             "verdict": _verdict(k, 2 * sd * math.sqrt(2), 1)})
            if "mids" in c.index:
                rows.append({"analysis": "B", "test": "placebo  C(−mids) ~ 0", "metric": metric,
                             "component": comp, "value": c["mids"], "n_bases": 1, "mde": 2 * sd,
                             "verdict": _verdict(c["mids"], 2 * sd, 0)})
    return pd.DataFrame(rows)


def write_md(path, arch, A, B, noise, summary, burn):
    L = [f"# Year-sensitivity ablations: {arch}", "",
         "Post-hoc; never used for selection. Estimand C = Δ(high year) − Δ(low year), "
         "Δ = arm − base on the same year, metrics oriented + = better. Design and caveats: "
         "`scripts/cross_validation/cv_year_sensitivity.py`, `notes/cv_preregistration.md`.", "",
         "Burned fraction (BurnDate): " + ", ".join(f"{y} {100 * burn[y]:.2f}%" for y in sorted(burn)), "",
         "## Noise floor (seed replicate, one draw)", "",
         "| metric | C_seed | rms Δ_seed | sd(C) used |", "|---|---|---|---|"]
    for m, v in noise.items():
        L.append(f"| {m} | {v['c_seed']:+.4f} | {v['rms_delta_seed']:.4f} | {v['sd_contrast']:.4f} |")
    if not noise:
        L.append("| — | seed replicate not scored | | |")
    L += ["", "## Hypotheses", "", "| analysis | test | metric | component | value | bases | MDE | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        L.append(f"| {r.analysis} | {r.test} | {r.metric} | {r.component} | {r.value:+.4f} | "
                 f"{r.n_bases} | {r.mde:.4f} | {r.verdict} |")
    for name, df in (("A: per base", A), ("B", B)):
        if df.empty:
            continue
        L += ["", f"## Contrasts, {name}", "",
              "| base | arm | metric | component | high | low | Δ high | Δ low | C |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.sort_values(["base_fold", "arm", "metric", "component"]).iterrows():
            L.append(f"| {r.base_fold} | {r.arm} | {r.metric} | {r.component} | {r.high_year} | "
                     f"{r.low_year} | {r.delta_high:+.4f} | {r.delta_low:+.4f} | {r.contrast:+.4f} |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def make_figure(A, B, noise, burn, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("A", A), ("B", B)]
    fig, axes = plt.subplots(len(FIG_METRICS), 2, figsize=(9, 2.6 * len(FIG_METRICS)), squeeze=False)
    labels = {k: lab for k, lab, *_ in METRICS}
    a_roles = probe_roles(burn)
    b_roles = drop_roles(dict(B.drop_duplicates("arm")[["arm", "probe"]].values), burn) if not B.empty else {}
    for row, metric in enumerate(FIG_METRICS):
        sd = noise.get(metric, {}).get("sd_contrast", float("nan"))
        for col, (name, df) in enumerate(panels):
            ax = axes[row][col]
            ax.axhline(0, color=INK2, lw=0.8, zorder=1)
            d = df[df.metric == metric] if not df.empty else df
            if name == "A":
                cats = [f"+{y}" for y in sorted(a_roles, key=lambda y: ["placebo", "high", "low"].index(a_roles[y]))]
                ticks = [f"{c}\n({a_roles[int(c[1:])]})" for c in cats]
                mde = 2 * sd / math.sqrt(2)
            else:
                order = sorted(b_roles, key=lambda a: ["mids", "highs", "lows"].index(b_roles[a]))
                cats = [df[df.arm == a].probe.iloc[0] for a in order]
                ticks = [f"{c.replace('-', '−', 1)}\n({b_roles[a]})" for c, a in zip(cats, order)]
                mde = 2 * sd
            if not math.isnan(mde):
                ax.axhspan(-mde, mde, color=BAND, alpha=0.6, lw=0, zorder=0)
            comps = [c for c in ("total", "network") if not d.empty and (d.component == c).any()]
            for j, comp in enumerate(comps):
                g = d[d.component == comp]
                for i, c in enumerate(cats):
                    v = g[g.probe == c].contrast
                    if v.empty:
                        continue
                    x = i + (j - (len(comps) - 1) / 2) * 0.22
                    if len(v) > 1:
                        ax.scatter([x] * len(v), v, s=22, facecolors="none", edgecolors=SERIES[comp],
                                   linewidths=1.2, zorder=2)
                    ax.scatter([x], [v.mean()], s=64, color=SERIES[comp], edgecolors="white", linewidths=1.5,
                               zorder=3, label={"total": "Total", "network": "Network (base γ)"}[comp]
                               if i == 0 else None)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(ticks, color=INK2, fontsize=8)
            ax.set_xlim(-0.6, len(cats) - 0.4)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8)
            if col == 0:
                ax.set_ylabel(f"{labels[metric]}\nContrast (High − Low)", color=INK, fontsize=9)
            if row == len(FIG_METRICS) - 1:
                ax.set_xlabel("Added Year" if name == "A" else "Removed Years", color=INK, fontsize=9)
    handles, labs = [], []
    for ax in axes.flat:
        h, l_ = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l_):
            if ll not in labs:
                handles.append(hh)
                labs.append(ll)
    if not A.empty and A.base_fold.nunique() > 1:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], ls="none", marker="o", markersize=5, markerfacecolor="none",
                              markeredgecolor=INK2, markeredgewidth=1.2))
        labs.append("Single base (A)")
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND, alpha=0.6, lw=0))
    labs.append("± MDE (seed replicate)")
    if handles:
        fig.legend(handles, labs, loc="upper center", ncol=len(labs), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default=None, help="default: the arch in selection_frozen.json")
    ap.add_argument("--scores", default=SCORES)
    args = ap.parse_args()

    if not os.path.exists(mk.SELECTION_FROZEN):
        sys.exit("GATE CLOSED: out/cv/selection_frozen.json does not exist")
    arch = args.arch or json.load(open(mk.SELECTION_FROZEN))["arch"]
    gate = mk.gate_errors("ablateA", arch)
    if gate:
        sys.exit("GATE CLOSED: " + gate[0])
    scores = pd.read_csv(args.scores)
    burn = burn_fraction()
    A = ablation_contrasts(scores, arch, burn, "ablateA")
    B = ablation_contrasts(scores, arch, burn, "ablateB")
    if A.empty and B.empty:
        sys.exit("no scored ablation arms yet -- run cv_collect_results.py --protocol ablate")
    noise = noise_floor(scores, arch, burn)
    summary = summarize(A, B, noise, burn)

    pd.concat([A, B], ignore_index=True).to_csv(os.path.join(OUT_DIR, "year_sensitivity.csv"), index=False)
    write_md(os.path.join(OUT_DIR, "year_sensitivity.md"), arch, A, B, noise, summary, burn)
    make_figure(A, B, noise, burn, os.path.join(OUT_DIR, "fig_year_sensitivity"))
    print(summary.to_string(index=False))
    print("wrote out/cv/year_sensitivity.{csv,md}, out/cv/fig_year_sensitivity.{png,pdf}")


if __name__ == "__main__":
    main()
