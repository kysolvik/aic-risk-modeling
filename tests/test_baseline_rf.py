"""baseline_rf: per-pixel feature extraction for the tabular random-forest baseline."""

import numpy as np

import baseline_rf as rf

P = rf.PATCH_PIXELS

CONFIG = {
    "input_features": {
        "im_annual": {
            "feature_names": ["im_forest", "im_BurnDate"],
            "timesteps": [-2, -1],
            "shape": [128, 128],
            "stack_timesteps": True,
            "transforms": {},
        },
        "im_single_cnn": {
            "feature_names": ["im_Elevation", "im_gov_type"],
            "timesteps": [],
            "shape": [128, 128],
            "stack_timesteps": False,
            "transforms": {"im_gov_type": "gt0"},
        },
        "md_single": {
            "feature_names": ["md_x", "md_y"],
            "timesteps": [],
            "shape": [1],
        },
        "md_monthly": {
            "feature_names": ["md_oni"],
            "timesteps": [],
            "shape": [3],
        },
    },
    "output_features": {
        "feature_names": ["im_BurnDate", "im_viirs_snpp"],
        "timesteps": [0],
    },
}


def _plan():
    return rf.feature_plan(CONFIG)


def test_feature_plan_layout():
    plan = _plan()
    assert [k for k, _ in plan["image_cols"]] == [
        "im_forest_-2", "im_forest_-1", "im_BurnDate_-2", "im_BurnDate_-1",
        "im_Elevation", "im_gov_type"]
    assert plan["vec_cols"] == [("md_oni", 3)]
    assert plan["scalar_cols"] == ["md_x", "md_y"]
    assert plan["label_keys"] == ["im_BurnDate_0", "im_viirs_snpp_0"]
    # 6 image + 3 climate + 2 coord = 11 columns.
    assert len(plan["names"]) == 11, plan["names"]


def test_wanted_keys():
    keys = rf.wanted_keys(_plan())
    assert "im_forest_-2" in keys and "md_oni" in keys
    assert "im_BurnDate_0" in keys and "im_viirs_snpp_0" in keys


def _make_rec():
    rng = np.random.default_rng(0)
    rec = {
        "im_forest_-2": rng.random((128, 128), np.float32),
        "im_forest_-1": rng.random((128, 128), np.float32),
        "im_BurnDate_-2": rng.random((128, 128), np.float32),
        "im_BurnDate_-1": rng.random((128, 128), np.float32),
        "im_Elevation": np.full((128, 128), 250.0, np.float32),
        "im_gov_type": np.zeros((128, 128), np.float32),
        "md_x": np.float32(-60.0),
        "md_y": np.float32(-3.0),
        "md_oni": np.array([0.1, 0.2, 0.3], np.float32),
        "im_BurnDate_0": np.zeros((128, 128), np.float32),
        "im_viirs_snpp_0": np.zeros((128, 128), np.float32),
    }
    rec["im_BurnDate_0"].reshape(-1)[[0, 1, 2]] = 5.0
    rec["im_viirs_snpp_0"].reshape(-1)[[2, 3]] = 1.0     # pixel 2 overlaps
    rec["im_gov_type"].reshape(-1)[10] = 2.0
    rec["im_Elevation"].reshape(-1)[20] = -32767.0
    return rec


def test_chip_matrix_shapes_label_and_transforms():
    plan = _plan()
    X, y = rf.chip_matrix(_make_rec(), plan)
    assert X.shape == (P, 11), X.shape
    assert y.dtype == bool and y.shape == (P,)
    # Union label: pixels 0,1,2 (BurnDate) and 2,3 (snpp) -> {0,1,2,3}.
    assert set(np.flatnonzero(y).tolist()) == {0, 1, 2, 3}

    names = plan["names"]
    gov = X[:, names.index("im_gov_type")]
    assert gov[10] == 1.0 and gov[0] == 0.0           # gt0 boolean-ized
    elev = X[:, names.index("im_Elevation")]
    assert np.isnan(elev[20]) and elev[0] == 250.0    # sentinel -> NaN
    oni0 = X[:, names.index("im_oni_m0") if "im_oni_m0" in names
             else names.index("md_oni_m0")]
    assert np.allclose(oni0, 0.1)                      # climate broadcast
    xcol = X[:, names.index("md_x")]
    assert np.allclose(xcol, -60.0)                   # coord broadcast
