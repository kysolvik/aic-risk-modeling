"""download_clim_indices: nodata is interpolated (never zeroed); length and calendar are invariant."""

import numpy as np
import pandas as pd
import pytest

from aic_risk_modeling.preprocess import climate_indices as ci


class _StubReadCsv:
    """Swap pd.read_csv inside the module for a canned frame."""

    def __init__(self, frame):
        self.frame = frame
        self.original = None

    def __enter__(self):
        self.original = ci.pd.read_csv
        ci.pd.read_csv = lambda *a, **k: self.frame.copy()
        return self

    def __exit__(self, *exc):
        ci.pd.read_csv = self.original


def _monthly(year_start, year_end, value=0.5):
    dates = pd.date_range(f"{year_start}-01-01", f"{year_end}-12-01", freq="MS")
    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"),
                         "metric": np.full(len(dates), value)})


def _fetch(frame, name="soi", year_start=2018, year_end=2023):
    with _StubReadCsv(frame):
        return ci.download_clim_indices(name, year_start=year_start, year_end=year_end)


def test_isolated_interior_sentinel_is_interpolated_not_zeroed():
    """The exact SOI 2025-04 case. 0.0 would be a plausible neutral reading."""
    f = _monthly(2016, 2024)
    f.loc[f["Date"] == "2020-04-01", "metric"] = np.nan
    f.loc[f["Date"] == "2020-03-01", "metric"] = 0.88
    f.loc[f["Date"] == "2020-05-01", "metric"] = 0.42
    f["metric"] = f["metric"].fillna(-9999.0)
    out = _fetch(f)
    got = float(out.loc["2020-04-01", "metric"])
    assert 0.42 < got < 0.88, got
    assert got != 0.0
    assert len(out) == 72


def test_length_and_calendar_are_invariant():
    """The property that makes a filter unusable: callers index positionally."""
    f = _monthly(2010, 2024)
    f.loc[f["Date"].isin(["2019-04-01", "2021-09-01"]), "metric"] = -9999.0
    out = _fetch(f, year_start=2018, year_end=2023)
    assert len(out) == 72
    expected = pd.date_range("2018-01-01", "2023-12-01", freq="MS")
    assert list(out.index) == list(expected)
    assert out["metric"].notna().all()
    assert (out["metric"].abs() <= ci.NODATA_ABS).all()


def test_trailing_sentinel_run_raises_instead_of_fabricating():
    f = _monthly(2016, 2023)
    f.loc[f["Date"] >= "2023-07-01", "metric"] = -9999.0
    with pytest.raises(ValueError, match='could not be interpolated') as ei:
        _fetch(f)
    assert "2023-07-01" in str(ei.value)


def test_leading_edge_gap_raises():
    f = _monthly(2016, 2024)
    f.loc[f["Date"] == "2018-01-01", "metric"] = -9999.0
    with pytest.raises(ValueError, match='could not be interpolated'):
        _fetch(f)


def test_long_interior_run_raises():
    f = _monthly(2016, 2024)
    f.loc[f["Date"].isin(["2020-03-01", "2020-04-01", "2020-05-01"]), "metric"] = -9999.0
    with pytest.raises(ValueError, match='MAX_FILL_GAP'):
        _fetch(f)


def test_month_absent_from_source_raises():
    """The AMO .dat is not sentinel-padded; it just ends. Same corruption, no -9999."""
    f = _monthly(2016, 2024)
    f = f[f["Date"] < "2023-09-01"]
    with pytest.raises(ValueError, match='not in the source') as ei:
        _fetch(f)
    assert "2023-09-01" in str(ei.value)


def test_all_header_declared_sentinels_are_caught():
    for sentinel in (-9999.0, -999.0, -99.99, -99.9):
        f = _monthly(2016, 2024)
        f.loc[f["Date"] == "2020-04-01", "metric"] = sentinel
        out = _fetch(f)
        assert abs(float(out.loc["2020-04-01", "metric"]) - 0.5) < 1e-9, sentinel


def test_real_extreme_values_survive():
    """SOI reaches 4.07 and MEI -2.17 in the real record; neither is missing."""
    f = _monthly(2016, 2024)
    f.loc[f["Date"] == "2019-02-01", "metric"] = 4.07
    f.loc[f["Date"] == "2020-08-01", "metric"] = -2.17
    out = _fetch(f)
    assert float(out.loc["2019-02-01", "metric"]) == 4.07
    assert float(out.loc["2020-08-01", "metric"]) == -2.17


def test_duplicate_dates_do_not_change_the_length():
    f = _monthly(2016, 2024)
    f = pd.concat([f, f[f["Date"] == "2020-06-01"]], ignore_index=True)
    out = _fetch(f)
    assert len(out) == 72


def test_unknown_index_name_raises():
    with pytest.raises(ValueError, match='not found'):
        ci.download_clim_indices("enso", 2018, 2023)


def test_caller_contract_values_column_is_positional_and_72_long():
    """Exactly how geebeam_ali_inputs.py consumes it."""
    f = _monthly(2010, 2024, value=0.25)
    f.loc[f["Date"] == "2019-04-01", "metric"] = -9999.0
    vals = _fetch(f, year_start=2018, year_end=2023).values[:, 0]
    assert vals.shape == (72,)
    assert np.isfinite(vals).all()
    # y1 = the last 12 entries = 2023; y1ond = the last 3 = Oct-Dec 2023.
    assert len(vals[60:72]) == 12 and len(vals[69:72]) == 3
