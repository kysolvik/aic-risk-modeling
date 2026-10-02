"""eval.year_offset: frozen global year offset gamma(t) on the chip-year panel (local panel only)."""

import os

import numpy as np
import pytest

from aic_risk_modeling.eval import year_offset as fyo

PANEL = os.path.join(os.path.dirname(__file__), "..", "out", "chip_panel", "panel.parquet")
pytestmark = pytest.mark.skipif(not os.path.exists(PANEL), reason=f"no local panel at {PANEL}")


def test_panel_regression_and_invariants():
    """The full measured suite: LOYO/forward r_year, b_soi, leak-free, sign, orthogonality."""
    assert fyo.run_checks(PANEL, verbose=False), "fit_year_offset regression checks failed"


def test_prev_burn_coefficient_is_negative_everywhere():
    """Mean-reverting, not persistence."""
    for target in ("bd", "snpp", "union_sum"):
        for prev in ("bd", "snpp", "union_sum"):
            d = fyo.load_panel(PANEL, target=target, prev_burn=prev, space="logit")
            e = fyo.evaluate(d, fyo.BOTH, protocol="forward", exclude_prev_from_clim=True)
            b_prev = e["beta_mean"][2]
            assert b_prev < 0, f"b_prev={b_prev:+.4f} >= 0 for target={target} prev={prev}"
            assert e["turns"] >= 6, f"turns {e['turns']}/{e['n_turns']} for {target}/{prev}"


def test_offsets_are_mean_centered():
    """Centering is what makes gamma independent of pos_weight."""
    d = fyo.load_panel(PANEL, target="union_sum", prev_burn="bd", space="logit")
    fit_years = list(range(2013, 2023))
    beta, _ = fyo.fit_final(d, fit_years)
    offsets, _ = fyo.build_offsets(d, beta, center_years=fit_years)
    mean_fit = np.mean([offsets[y] for y in fit_years])
    assert abs(mean_fit) < 1e-9, f"fit-year offsets not centered: mean={mean_fit:.2e}"
    assert set(offsets) >= set(fit_years) | {2023, 2024, 2025}


def test_gamma_moves_the_right_way_at_the_turns():
    """2024 is an up-year and 2025 a down-year; gamma must have the sign right."""
    d = fyo.load_panel(PANEL, target="union_sum", prev_burn="bd", space="logit")
    fit_years = list(range(2013, 2023))
    beta, _ = fyo.fit_final(d, fit_years)
    offsets, _ = fyo.build_offsets(d, beta, center_years=fit_years)
    assert offsets[2024] > offsets[2023], "gamma must rise into 2024"
    assert offsets[2025] < offsets[2024], "gamma must fall into 2025"
    ratio = np.exp(offsets[2024] - offsets[2023])
    assert ratio > 1.25, f"2024/2023 gamma ratio only {ratio:.3f}x"


def test_spatial_interaction_cannot_change_amplitude():
    """Orthogonality, stated as a property rather than a pinned number."""
    d = fyo.load_panel(PANEL, **fyo.REF)
    base = fyo.evaluate(d, fyo.BOTH)
    for basis in ("zlat", "zlon", "zne"):
        alt = fyo.evaluate(d, fyo.BOTH + [("zsoi", basis)])
        assert abs(alt["r_year"] - base["r_year"]) < 1e-3, f"{basis} moved r_year"
        assert abs(alt["amplitude"] - base["amplitude"]) < 1e-3, f"{basis} moved amplitude"
