"""Tile actual-vs-expected burn area and municipality aggregation."""

import math
import os
import tempfile

import numpy as np
import pytest

from aic_risk_modeling.eval.eval import (
    _fire_probability,
    municipality_burn_area_stats,
    tile_burn_area_stats,
)


def test_expected_equals_actual_for_binary_labels():
    rng = np.random.default_rng(0)
    labels = (rng.uniform(size=(256, 256)) < 0.1).astype(np.float64)
    stats = tile_burn_area_stats(labels, labels, tile_size=128)
    assert stats["n_tiles"] == 4  # 256/128 == 2 tiles per side
    assert stats["total_actual"] == stats["total_expected"]
    assert abs(stats["bias"]) < 1e-9
    assert abs(stats["mae"]) < 1e-9
    assert abs(stats["ratio"] - 1.0) < 1e-9
    assert abs(stats["pearson_r"] - 1.0) < 1e-9


def test_known_small_arrays():
    # 2x2 tiles over a 4x4 scene. Tile layout (row,col): (0,0)(0,2)(2,0)(2,2).
    prob = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ])
    gt = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ])
    stats = tile_burn_area_stats(prob, gt, tile_size=2)
    assert stats["n_tiles"] == 4
    # actual per tile: [1, 0, 0, 1]; expected per tile: [1.0, 0, 0, 4.0]
    assert stats["total_actual"] == 2.0
    assert stats["total_expected"] == 5.0
    assert abs(stats["ratio"] - 2.5) < 1e-9
    # errors = expected - actual = [0, 0, 0, 3]; bias mean = 0.75, mae = 0.75
    assert abs(stats["bias"] - 0.75) < 1e-9
    assert abs(stats["mae"] - 0.75) < 1e-9
    assert abs(stats["rmse"] - math.sqrt(9.0 / 4)) < 1e-9
    per_tile = stats["per_tile"]
    assert list(per_tile["actual"]) == [1.0, 0.0, 0.0, 1.0]
    assert list(per_tile["expected"]) == [1.0, 0.0, 0.0, 4.0]


def test_partial_edge_tiles_included():
    # 5x3, tile 2 -> 3*2 = 6 tiles; edge tiles smaller.
    prob = np.zeros((5, 3))
    gt = np.zeros((5, 3), dtype=int)
    stats = tile_burn_area_stats(prob, gt, tile_size=2)
    assert stats["n_tiles"] == math.ceil(5 / 2) * math.ceil(3 / 2)
    n_pixels = sorted(stats["per_tile"]["n_pixels"].tolist())
    assert int(stats["per_tile"]["n_pixels"].sum()) == 5 * 3
    assert min(n_pixels) < max(n_pixels)  # tiles differ in area


def test_multiclass_uses_fire_complement():
    # Band-first multiclass: P(fire) = 1 - band 1.
    num_classes = 5
    height = width = 128
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.0, 1.0, size=(num_classes, height, width))
    scores /= scores.sum(axis=0, keepdims=True)  # softmax-like: sum to 1
    argmax = scores.argmax(axis=0)[None].astype(np.float64)
    predictions = np.concatenate([argmax, scores], axis=0)  # (C+1, H, W)

    prob = _fire_probability(predictions)
    assert np.allclose(prob, 1.0 - scores[0])

    gt = (scores.argmax(axis=0) > 0).astype(int)
    stats = tile_burn_area_stats(predictions, gt, tile_size=128)
    assert stats["n_tiles"] == 1
    assert abs(stats["total_expected"] - float((1.0 - scores[0]).sum())) < 1e-6


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        tile_burn_area_stats(np.zeros((10, 10)), np.zeros((10, 8), dtype=int))


def test_csv_and_plot_written():
    rng = np.random.default_rng(4)
    prob = rng.uniform(size=(256, 256))
    gt = (rng.uniform(size=(256, 256)) < 0.2).astype(int)
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "tiles.csv")
        png_path = os.path.join(d, "tiles.png")
        tile_burn_area_stats(prob, gt, tile_size=128,
                             csv_path=csv_path, plot=png_path)
        assert os.path.exists(csv_path)
        import csv as _csv
        with open(csv_path) as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) == 4
        assert set(rows[0].keys()) == {
            "row", "col", "n_pixels", "actual", "expected", "error"}
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pass
        else:
            assert os.path.exists(png_path)


def _two_municipality_scene():
    """A 4x8 EPSG:4326 grid split into a left ('A') and right ('B') municipality."""
    from rasterio.transform import from_origin
    from shapely.geometry import box
    import geopandas as gpd

    transform = from_origin(0.0, 0.0, 1.0, 1.0)  # origin top-left, 1-deg pixels
    crs = "EPSG:4326"
    # Left half (cols 0-3): expected sum = 4 rows * 4 cols * 0.5 = 8; actual = 4.
    # Right half (cols 4-7): expected = 4 * 4 * 1.0 = 16; actual = 8.
    prob = np.hstack([np.full((4, 4), 0.5), np.full((4, 4), 1.0)]).astype(np.float64)
    gt = np.zeros((4, 8), dtype=int)
    gt[:, 0] = 1        # 4 burned pixels in A
    gt[:, 4:6] = 1      # 8 burned pixels in B (cols 4 and 5)
    gdf = gpd.GeoDataFrame(
        {"cd_mun": [1, 2], "nm_mun": ["A", "B"], "sigla_uf": ["XX", "YY"]},
        geometry=[box(0, -4, 4, 0), box(4, -4, 8, 0)], crs=crs)
    return prob, gt, transform, crs, gdf


def test_municipality_zonal_sums():
    prob, gt, transform, crs, gdf = _two_municipality_scene()
    with tempfile.TemporaryDirectory() as d:
        shp = os.path.join(d, "munis.shp")
        gdf.to_file(shp)
        stats = municipality_burn_area_stats(
            prob, gt, transform, crs, shp, top_n=2)
    assert stats["n_municipalities"] == 2
    per = stats["per_municipality"]
    by_name = dict(zip(per["nm_mun"], zip(per["actual"], per["expected"],
                                          per["n_pixels"])))
    # A: actual = 4 burned pixels, expected = 16 pixels * 0.5 = 8.
    assert by_name["A"] == (4.0, 8.0, 16)
    # B: actual = 8 burned pixels, expected = 16 pixels * 1.0 = 16.
    assert by_name["B"] == (8.0, 16.0, 16)
    assert stats["total_actual"] == 12.0
    assert stats["total_expected"] == 24.0
    assert abs(stats["ratio"] - 2.0) < 1e-9


def test_municipality_no_overlap_returns_none():
    prob, gt, transform, crs, gdf = _two_municipality_scene()
    from shapely.geometry import box
    import geopandas as gpd
    far = gpd.GeoDataFrame(
        {"cd_mun": [9], "nm_mun": ["Far"], "sigla_uf": ["ZZ"]},
        geometry=[box(100, 100, 101, 101)], crs=crs)
    with tempfile.TemporaryDirectory() as d:
        shp = os.path.join(d, "far.shp")
        far.to_file(shp)
        assert municipality_burn_area_stats(prob, gt, transform, crs, shp) is None


def test_municipality_csv_and_plot_written():
    prob, gt, transform, crs, gdf = _two_municipality_scene()
    with tempfile.TemporaryDirectory() as d:
        shp = os.path.join(d, "munis.shp")
        gdf.to_file(shp)
        csv_path = os.path.join(d, "munis_out.csv")
        png_path = os.path.join(d, "munis_out.png")
        municipality_burn_area_stats(
            prob, gt, transform, crs, shp, csv_path=csv_path, plot=png_path)
        assert os.path.exists(csv_path)
        import csv as _csv
        with open(csv_path) as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) == 2
        assert set(rows[0].keys()) == {
            "cd_mun", "nm_mun", "sigla_uf", "n_pixels", "actual", "expected",
            "error"}
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pass
        else:
            assert os.path.exists(png_path)


def test_municipality_highlight_flags_selected():
    prob, gt, transform, crs, gdf = _two_municipality_scene()
    with tempfile.TemporaryDirectory() as d:
        shp = os.path.join(d, "munis.shp")
        gdf.to_file(shp)
        csv_path = os.path.join(d, "munis_out.csv")
        stats = municipality_burn_area_stats(
            prob, gt, transform, crs, shp, csv_path=csv_path,
            highlight_cd_mun=[2, 999])  # 2 -> B present, 999 absent
        per = stats["per_municipality"]
        flagged = dict(zip(per["nm_mun"], per["highlighted"]))
        assert flagged["B"] and not flagged["A"]
        import csv as _csv
        with open(csv_path) as f:
            rows = list(_csv.DictReader(f))
        assert "highlighted" in rows[0]
