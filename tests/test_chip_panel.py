"""extract_chip_panel: per-chip reductions mask nodata sentinels; paths keep gs://."""

import numpy as np
import pandas as pd
import pytest

import extract_chip_panel as extract

SENTINEL = -2147483648.0


def _blank_record(fire_type=None, burn_date=None, elevation=None):
    rec = {}
    for name in extract.timestepped(extract.MONTHLY_BANDS, extract.MONTHLY_TIMESTEPS):
        rec[name] = np.zeros((128, 128), dtype=np.float32)
    for name in extract.timestepped(extract.ANNUAL_BANDS, extract.ANNUAL_TIMESTEPS):
        rec[name] = np.zeros((128, 128), dtype=np.float32)
    for name in extract.STATIC_BANDS:
        rec[name] = np.ones((128, 128), dtype=np.float32)
    rec["im_fire_type"] = (np.zeros((128, 128), dtype=np.float32)
                           if fire_type is None else fire_type)
    rec["im_BurnDate_0"] = (np.zeros((128, 128), dtype=np.float32)
                            if burn_date is None else burn_date)
    rec["im_viirs_snpp_0"] = np.zeros((128, 128), dtype=np.float32)
    if elevation is not None:
        rec["im_Elevation"] = elevation
    rec["md_id"] = np.array([7], dtype=np.int64)
    rec["md_year"] = np.array([2023], dtype=np.int64)
    rec["md_x"] = np.array([-60.0], dtype=np.float32)
    rec["md_y"] = np.array([-5.0], dtype=np.float32)
    for name in extract.CLIM_INDICES:
        rec[name] = np.arange(72, dtype=np.float32)
    return rec


def test_sentinel_excluded_from_burned_count():
    ft = np.zeros((128, 128), dtype=np.float32)
    ft[:10, :10] = SENTINEL   # 100 nodata pixels
    ft[20:24, 20:25] = 3.0    # 20 burned pixels
    row = extract.reduce_record(_blank_record(fire_type=ft))
    assert row["n_sentinel"] == 100
    assert row["burn_ft"] == 20
    assert row["n_pixels"] == 128 * 128


def test_fire_type_classes_partition_the_burned_count():
    ft = np.zeros((128, 128), dtype=np.float32)
    ft[0, :5] = 1.0
    ft[1, :7] = 2.0
    ft[2, :3] = 3.0
    ft[3, :4] = 4.0
    ft[4, :6] = SENTINEL
    row = extract.reduce_record(_blank_record(fire_type=ft))
    per_class = sum(row[f"burn_ft_c{c}"] for c in (1, 2, 3, 4))
    assert row["burn_ft"] == per_class == 5 + 7 + 3 + 4
    assert row["n_sentinel"] == 6


def test_burn_date_is_a_positive_pixel_count():
    bd = np.zeros((128, 128), dtype=np.float32)
    bd[:3, :4] = 1.0
    row = extract.reduce_record(_blank_record(burn_date=bd))
    assert row["burn_bd"] == 12


def test_static_nodata_excluded_from_mean():
    elev = np.full((128, 128), 200.0, dtype=np.float32)
    elev[:64, :] = -32767.0
    row = extract.reduce_record(_blank_record(elevation=elev))
    assert row["im_Elevation_nodata"] == 64 * 128
    assert abs(row["im_Elevation_mean"] - 200.0) < 1e-6


def test_climate_index_slices_pick_the_right_months():
    # 72 values = Y-6..Y-1 monthly: 60..71 is Y-1, 69..71 is Oct-Dec of Y-1.
    row = extract.reduce_record(_blank_record())
    assert abs(row["md_oni_y1"] - np.arange(60, 72).mean()) < 1e-6
    assert abs(row["md_oni_y1ond"] - np.arange(69, 72).mean()) < 1e-6
    assert abs(row["md_oni_y2"] - np.arange(48, 60).mean()) < 1e-6


def test_excluded_bands_are_not_requested():
    want = set(extract.wanted_features())
    for dead in ("im_gov_type", "im_alert", "im_alertdate"):
        assert dead not in want
    assert not any(w.startswith("im_chirps_cwd_-") for w in want)
    assert "im_chirps_cwd_monthly_-4" in want


def test_join_preserves_the_gs_scheme():
    assert extract._join("out/chip_panel", "2023", "x.parquet") == "out/chip_panel/2023/x.parquet"
    assert extract._join("out/chip_panel/", "2023", "x.parquet") == "out/chip_panel/2023/x.parquet"
    assert extract._join("gs://b/prefix", "2023", "x.parquet") == "gs://b/prefix/2023/x.parquet"
    assert extract._join("gs://b/prefix/", "2023", "x.parquet") == "gs://b/prefix/2023/x.parquet"
    assert extract._join("gs://b", "*", "*.parquet") == "gs://b/*/*.parquet"


def test_parquet_round_trip_local(tmp_path):
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"md_id": [1, 2], "year": [2023, 2023], "burn_bd": [5, 7]})
    path = extract._join(str(tmp_path), "2023", "part.parquet")
    extract._write_parquet(df, path)  # creates the parent dir too
    pd.testing.assert_frame_equal(df, extract._read_parquet(path))
