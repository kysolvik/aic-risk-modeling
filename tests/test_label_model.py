"""fit_label_model: latent-class sensitivity/FPR fit behind confidence weighting."""

import numpy as np

import fit_label_model as flm


def test_recovers_known_parameters():
    counts, pi, sens, fpr = flm.synthetic_tables(seed=3)
    pi_hat, sens_hat, fpr_hat, _ = flm.em_fit(counts, seed=3)
    assert np.abs(pi - pi_hat).max() < 5e-3, (pi, pi_hat)
    assert np.abs(sens - sens_hat).max() < 2e-2, (sens, sens_hat)
    assert np.abs(fpr - fpr_hat).max() < 1e-3, (fpr, fpr_hat)


def test_recovers_a_product_that_is_blind_in_one_stratum():
    """MCD64's tropical-forest omission is the whole point; it must survive."""
    counts, pi, sens, _ = flm.synthetic_tables(seed=5)
    _, sens_hat, _, _ = flm.em_fit(counts, seed=5)
    # column 0 is MCD64; stratum 2 is closed canopy, where truth is 0.09
    assert sens_hat[2, 0] < 0.15
    assert sens_hat[0, 0] > 0.40
    assert sens_hat[2, 0] < sens_hat[2, 2], "VIIRS should outsee MCD64 in forest"


def test_gate_does_not_reject_independent_data():
    counts, *_ = flm.synthetic_tables(seed=7)
    pi, sens, fpr, _ = flm.em_fit(counts, seed=7)
    g2, df, _ = flm.fit_statistic(counts, pi, sens, fpr)
    assert df == 6, f"3 strata should leave 3G-3 = 6 df, got {df}"
    from scipy import stats
    assert stats.chi2.sf(g2, df) > 0.01, f"false rejection: G2={g2}"


def test_gate_rejects_dependent_products():
    counts, *_ = flm.synthetic_tables(seed=7, dependence=0.7)
    pi, sens, fpr, _ = flm.em_fit(counts, seed=7)
    g2, df, expected = flm.fit_statistic(counts, pi, sens, fpr)
    from scipy import stats
    assert stats.chi2.sf(g2, df) < 1e-6, f"no power: G2={g2} on {df} df"

    pairs = flm.pair_corrections(counts, expected)
    worst = min(pairs, key=lambda p: p["both"])
    assert worst["pair"] == "mcd64|mod14", worst
    assert worst["both"] < 0, "excess agreement must produce a DISCOUNT"


def test_exactly_identified_single_stratum_reports_zero_df():
    """Three tests in one population: estimable, but not testable."""
    counts, *_ = flm.synthetic_tables(seed=9)
    pooled = counts.sum(axis=0, keepdims=True)
    pi, sens, fpr, _ = flm.em_fit(pooled, seed=9)
    _, df, _ = flm.fit_statistic(pooled, pi, sens, fpr)
    assert df == 0, f"1 stratum should leave 0 df, got {df}"


def test_posterior_ordering_and_asymmetry():
    counts, *_ = flm.synthetic_tables(seed=11)
    pi, sens, fpr, _ = flm.em_fit(counts, seed=11)
    q = flm.posterior_table(pi, sens, fpr)
    for g in range(q.shape[0]):
        assert q[g, 0] == q[g].min(), "all-negative must be the least evidence"
        assert q[g, 7] == q[g].max(), "three-way agreement must be the most"
    # A single detection is ambiguous: why the hard union stays the eval target.
    assert 0.2 < q[0, 1] < 0.8


def test_weight_scales_hold_mean_loss_weight_at_baseline():
    """Both arms must keep the unweighted mean weight, or the LR shifts."""
    counts, *_ = flm.synthetic_tables(seed=13)
    pi, sens, fpr, _ = flm.em_fit(counts, seed=13)
    q = flm.posterior_table(pi, sens, fpr)
    P = 10.0
    scales = flm.weight_scales(counts, q, P)

    joint = counts / counts.sum()
    union = flm.UNION[None, :]
    for arm, b, denom in (("hard", np.where(union, q, 1 - q),
                           np.where(union, P, 1.0)),
                          ("soft", 1.0, P * q + 1 - q)):
        mean_w = (joint * b * denom * scales[arm]).sum()
        assert np.isclose(mean_w, scales["baseline_mean_weight"], rtol=1e-6), \
            (arm, mean_w, scales["baseline_mean_weight"])


def test_doy_llr_is_positive_for_same_week_co_detection():
    import pandas as pd
    rows = []
    for b, n in enumerate([800, 300, 150, 80, 40]):
        rows.append({"pair": "mcd64|viirs", "stratum": 0, "kind": "observed",
                     "bin": b, "count": n})
    for b, n in enumerate([60, 90, 300, 500, 900]):
        rows.append({"pair": "mcd64|viirs", "stratum": 0, "kind": "null",
                     "bin": b, "count": n})
    terms = flm.doy_terms(pd.DataFrame(rows), [2.0, 8.0, 32.0, 128.0])
    assert len(terms) == 1
    values = terms[0]["values"]
    assert values[0] > 1.0, values
    assert values[-1] < 0.0, values
    assert terms[0]["pairs"] == [[0, 2]], terms[0]["pairs"]
