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


# --------------------------------------------------------------------------- CPC sources
# Trimmed copies of the real file layouts (fetched 2026-10-05).
CPC_SOI_TEXT = """\
(STAND TAHITI - STAND DARWIN)  SEA LEVEL PRESS
                        ANOMALY

YEAR   JAN   FEB   MAR   APR   MAY   JUN   JUL   AUG   SEP   OCT   NOV   DEC
2025   0.3   0.9   2.8   0.9   0.7   0.5   1.0   0.7   0.1   1.9   1.8  -0.0
2026   1.8   2.4   2.0  -1.1  -1.5  -2.4  -4.0  -1.8  -3.3-999.9-999.9-999.9

(STAND TAHITI - STAND DARWIN)  SEA LEVEL PRESS
                    STANDARDIZED    DATA

YEAR   JAN   FEB   MAR   APR   MAY   JUN   JUL   AUG   SEP   OCT   NOV   DEC
2025   0.2   0.5   1.7   0.5   0.4   0.3   0.6   0.4   0.0   1.1   1.1  -0.0
2026   1.1   1.4   1.2  -0.6  -0.9  -1.4  -2.4  -1.1  -2.0-999.9-999.9-999.9
"""

CPC_SSTOI_TEXT = """\
YR MON  NINO1+2   ANOM   NINO3    ANOM   NINO4    ANOM NINO3.4    ANOM
2025  12   23.10   -0.50   25.00   -0.60   28.30   -0.40   26.00   -0.55
2026   1   24.28   -0.24   25.84    0.17   28.01   -0.21   26.65    0.08
2026   2   25.38   -0.72   26.26   -0.11   27.99   -0.11   26.54   -0.20
"""


class _StubFetchText:
    def __init__(self, text):
        self.text, self.original = text, None

    def __enter__(self):
        self.original = ci._fetch_text
        ci._fetch_text = lambda url: self.text
        return self

    def __exit__(self, *exc):
        ci._fetch_text = self.original


def test_cpc_soi_reads_the_standardized_table_and_handles_run_together_sentinels():
    with _StubFetchText(CPC_SOI_TEXT):
        out = ci.download_clim_indices("soi_cpc", 2025, 2026, last_month=9)
    assert len(out) == 21
    # standardized table, not the anomaly table above it
    assert float(out.loc["2026-07-01", "metric"]) == -2.4
    assert float(out.loc["2026-09-01", "metric"]) == -2.0
    assert float(out.loc["2025-03-01", "metric"]) == 1.7


def test_cpc_soi_unpublished_months_raise():
    """-999.9 placeholders for Oct-Dec are at the trailing edge: never filled."""
    with _StubFetchText(CPC_SOI_TEXT), pytest.raises(ValueError):
        ci.download_clim_indices("soi_cpc", 2025, 2026)


def test_cpc_sstoi_picks_the_anomaly_column_of_each_region():
    with _StubFetchText(CPC_SSTOI_TEXT):
        n34 = ci.download_clim_indices("nino34", 2026, 2026, last_month=2)
        n4 = ci.download_clim_indices("nino4", 2026, 2026, last_month=2)
    assert list(n34["metric"]) == [0.08, -0.20]
    assert list(n4["metric"]) == [-0.21, -0.11]
