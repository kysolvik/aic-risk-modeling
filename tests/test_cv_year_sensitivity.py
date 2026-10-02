"""Protocol scoring math (cv_collect_results) and post-hoc year sensitivity (gamma swap)."""

import math

import numpy as np
import pandas as pd

import cv_collect_results as cc
import cv_year_sensitivity as ys

# BurnDate burned fraction per year, from the chip panel.
BURN = {2013: .0115, 2014: .0164, 2015: .0200, 2016: .0190, 2017: .0227, 2018: .0112, 2019: .0216,
        2020: .0253, 2021: .0150, 2022: .0182, 2023: .0173, 2024: .0346, 2025: .0110}


def test_gamma_swap_is_exact_and_round_trips():
    rng = np.random.default_rng(0)
    q = rng.uniform(1e-5, 0.9, size=(4, 16, 16))
    back = cc.swap_gamma(cc.swap_gamma(q, 0.3, -0.2), -0.2, 0.3)
    assert np.allclose(back, q, atol=1e-9), "swap does not round-trip"
    assert np.allclose(cc.swap_gamma(q, 0.1, 0.1), q, atol=1e-12), "identity swap moved values"
    assert (cc.swap_gamma(q, 0.25, 0.0) < q).all()
    labels = rng.random((4, 16, 16)) < 0.1
    own = cc.year_totals(q.astype(np.float32), labels, 10.0)
    net = cc.year_totals(q.astype(np.float32), labels, 10.0, g_from=0.25, g_to=0.0)
    assert net["E"] < own["E"] and own["A"] == net["A"]
    assert math.isclose(own["logbias"], math.log(own["E"] / own["A"]))


def test_box_mean_matches_brute_force():
    rng = np.random.default_rng(1)
    a = rng.random((23, 31))
    k, r = 9, 4
    p = np.pad(a, r)
    brute = np.array([[p[i:i + k, j:j + k].mean() for j in range(a.shape[1])] for i in range(a.shape[0])])
    assert np.allclose(cc.box_mean(a, k), brute, atol=1e-5)


def test_pair_metrics():
    tracked = cc.pair_metrics(100.0, 200.0, 50.0, 100.0)
    assert abs(tracked["log_ratio_err"]) < 1e-12 and abs(tracked["amp_frac"] - 1) < 1e-12
    flat = cc.pair_metrics(100.0, 200.0, 50.0, 50.0)
    assert abs(flat["amp_frac"]) < 1e-12 and flat["log_ratio_err"] < 0
    inverted = cc.pair_metrics(100.0, 50.0, 50.0, 60.0)
    assert inverted["amp_frac"] < 0
    small = cc.pair_metrics(100.0, 105.0, 50.0, 80.0)
    assert not small["turn"] and math.isnan(small["amp_frac"]), "flat pair must not get amp_frac"
    a0, a1 = np.array([1., 5., 9.]), np.array([2., 9., 20.])
    r = cc.pair_metrics(15., 31., 15., 31., a0, a1, a0, a1)["r_dE_dA"]
    assert abs(r - 1) < 1e-12


def _fold_scores(arch, skill_bump, lre=0.3, last3_bump=None):
    """Fold-stage scores + pairs for one arch; skill_bump(fold_origin) shifts pr_auc_skill."""
    rows, pairs = [], []
    for t in range(2018, 2023):
        b = skill_bump(t)
        for y in (t, t + 1):
            rows.append({"arch": arch, "fold_id": f"fwdpair_{t}", "stage": "folds", "year": y,
                         "n_train_years": t - 2013, "pr_auc_skill": 0.07 + b,
                         "best": 0.34 + b, "last3": 0.335 + (b if last3_bump is None else last3_bump)})
        turn = t in (2018, 2020)
        pairs.append({"arch": arch, "fold_id": f"fwdpair_{t}", "ref": "", "turn": turn, "log_ratio_err": lre if turn else 0.0})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def test_selection_rule():
    def verdict(bump, lre=0.3, last3_bump=None):
        rs, rp = _fold_scores("ref", lambda t: 0.0)
        xs, xp = _fold_scores("x", bump, lre, last3_bump)
        v = cc.selection_verdicts(pd.concat([rs, xs]), pd.concat([rp, xp]), "ref")
        return v.iloc[0].verdict
    assert verdict(lambda t: 0.03) == "beats reference"
    assert verdict(lambda t: 0.01) == "does not beat reference", "gap inside the 0.02 noise threshold"
    # big mean gain carried by 2 folds, and NOT the largest one -> fails consistency
    assert verdict(lambda t: 0.2 if t in (2018, 2019) else -0.005) == "does not beat reference"
    # skill win but amplitude worse beyond tolerance
    assert verdict(lambda t: 0.03, lre=0.45) == "does not beat reference"
    # amplitude-only win (|lre| 0.3 -> 0.1) with skill flat is a legitimate (b) win...
    assert verdict(lambda t: 0.0, lre=0.1) == "beats reference"
    # ...but not if it costs skill beyond the noise threshold
    assert verdict(lambda t: -0.03, lre=0.1) == "does not beat reference"
    # best says better, last3 says worse -> a max-over-epochs artifact
    assert verdict(lambda t: 0.03, last3_bump=-0.01) == "does not beat reference"


def _synthetic_scores(effect=0.05, seed_noise=0.002, placebo=0.0, gamma_share=0.0):
    """Protocol score rows for arch 'x' with an optional planted regime-matching effect."""
    rng = np.random.default_rng(7)
    rows = []

    def add(fid, stage, base, probe, years, bump):
        for y in years:
            b = bump(y)
            rows.append({"arch": "x", "fold_id": fid, "stage": stage, "base_fold": base, "probe": probe,
                         "year": y, "pr_auc": 0.30 + b, "within_chip_only": 0.13 + b / 2, "chip_r": 0.8 + b,
                         "logbias": -0.30 + b, "logbias_basegamma": -0.30 + (1 - gamma_share) * b,
                         "brier": 0.040 - b / 10, "brier_basegamma": 0.040 - (1 - gamma_share) * b / 10})

    zero = lambda y: 0.0  # noqa: E731
    for base, years in (("fwdpair_2018", (2018, 2019)), ("fwdpair_2020", (2020, 2021))):
        add(base, "folds", "", "", years, zero)
        hi, lo = sorted(years, key=lambda y: -BURN[y])
        add(f"{base}_add2024", "ablateA", base, "+2024", years, lambda y, hi=hi: effect if y == hi else 0.0)
        add(f"{base}_add2025", "ablateA", base, "+2025", years, lambda y, lo=lo: effect if y == lo else 0.0)
        add(f"{base}_add2023", "ablateA", base, "+2023", years, lambda y: placebo)
    add("fwdpair_2020_s55", "seedrep", "fwdpair_2020", "", (2020, 2021),
        lambda y: float(rng.normal(0, seed_noise)))
    add("final", "final", "", "", (2024, 2025), zero)
    add("final_drop2017-2020", "ablateB", "final", "-2017,2020", (2024, 2025),
        lambda y: -effect if y == 2024 else 0.0)
    add("final_drop2013-2018", "ablateB", "final", "-2013,2018", (2024, 2025),
        lambda y: -effect if y == 2025 else 0.0)
    add("final_drop2016-2022", "ablateB", "final", "-2016,2022", (2024, 2025), zero)
    return pd.DataFrame(rows)


def _analyse(scores):
    A = ys.ablation_contrasts(scores, "x", BURN, "ablateA")
    B = ys.ablation_contrasts(scores, "x", BURN, "ablateB")
    noise = ys.noise_floor(scores, "x", BURN)
    return A, B, noise, ys.summarize(A, B, noise, BURN)


def _verdict(summary, analysis, test_prefix, metric, component="total"):
    r = summary[(summary.analysis == analysis) & summary.test.str.startswith(test_prefix)
                & (summary.metric == metric) & (summary.component == component)]
    assert len(r) == 1, (analysis, test_prefix, metric, component, len(r))
    return r.iloc[0]


def test_planted_effect_is_recovered():
    A, B, noise, s = _analyse(_synthetic_scores(effect=0.05))
    assert ys.probe_roles(BURN) == {2024: "high", 2025: "low", 2023: "placebo"}
    for m in ("pr_auc", "magnitude", "brier"):
        h1 = _verdict(s, "A", "H1", m)
        h2 = _verdict(s, "A", "H2", m)
        key = _verdict(s, "A", "key", m)
        plac = _verdict(s, "A", "placebo", m)
        assert h1.value > 0 and h1.verdict == "supported", (m, h1.to_dict())
        assert h2.value < 0 and h2.verdict == "supported", (m, h2.to_dict())
        assert key.verdict == "supported" and plac.verdict.startswith("null"), (m, key.to_dict(), plac.to_dict())
        kb = _verdict(s, "B", "key", m)
        assert kb.value > 0 and kb.verdict == "supported", (m, kb.to_dict())
        assert _verdict(s, "B", "placebo", m).verdict.startswith("null")
    per = A[(A.metric == "pr_auc") & (A.component == "total") & (A.probe == "+2024")]
    assert set(per.base_fold) == {"fwdpair_2018", "fwdpair_2020"} and (per.contrast > 0).all()


def test_flat_world_reads_null():
    _, _, _, s = _analyse(_synthetic_scores(effect=0.0))
    assert not (s.verdict == "supported").any(), s[s.verdict == "supported"]


def test_moving_placebo_is_flagged():
    # Equal help to both years -> contrast 0: a level shift alone must not be flagged.
    _, _, _, s = _analyse(_synthetic_scores(effect=0.05, placebo=0.05))
    assert _verdict(s, "A", "placebo", "pr_auc").verdict.startswith("null")


def test_network_gamma_split_adds_up():
    A, _, _, s = _analyse(_synthetic_scores(effect=0.05, gamma_share=0.75))
    g = A[(A.metric == "magnitude") & (A.arm == "fwdpair_2020_add2024")].set_index("component").contrast
    assert math.isclose(g["total"], g["network"] + g["gamma"], abs_tol=1e-12)
    assert math.isclose(g["gamma"], 0.75 * g["total"], rel_tol=1e-9)
    assert _verdict(s, "A", "H1", "magnitude", "gamma").value > _verdict(s, "A", "H1", "magnitude", "network").value
    assert "network" not in set(A[A.metric == "pr_auc"].component), "ranking metrics must carry no split"
