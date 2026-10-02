"""Pyramid max-pool detection metrics (PR AUC / F1 at increasing block sizes)."""

import os
import tempfile

import numpy as np
import pytest

from aic_risk_modeling.eval.eval import (
    _block_max_pool,
    _default_pyramid_blocks,
    pyramid_pool_stats,
)


def test_block_max_pool_exact_divisor():
    arr = np.array([
        [0.1, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.7],
    ])
    pooled = _block_max_pool(arr, 2)
    assert pooled.shape == (2, 2)
    assert np.allclose(pooled, [[0.9, 0.0], [0.0, 0.7]])


def test_block_max_pool_pads_partial_edges():
    # 3x3, block 2 -> zero-padded to 4x4 -> 2x2.
    arr = np.arange(9, dtype=np.float64).reshape(3, 3)
    pooled = _block_max_pool(arr, 2)
    assert pooled.shape == (2, 2)
    # blocks: [[0,1],[3,4]]->4, [[2],[5]]->5, [[6,7]]->7, [[8]]->8
    assert np.allclose(pooled, [[4.0, 5.0], [7.0, 8.0]])


def test_block_size_one_is_identity():
    arr = np.random.default_rng(0).uniform(size=(8, 8))
    assert np.array_equal(_block_max_pool(arr, 1), arr)


def test_default_blocks_powers_of_two():
    assert _default_pyramid_blocks(256, 256) == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert _default_pyramid_blocks(10, 5) == [1, 2, 4]


def test_perfect_prediction_scores_one_at_all_levels():
    rng = np.random.default_rng(1)
    labels = (rng.uniform(size=(64, 64)) < 0.1).astype(np.float64)
    out = pyramid_pool_stats(labels, labels)
    assert [s["block"] for s in out["levels"]] == [1, 2, 4, 8, 16, 32, 64]
    for s in out["levels"]:
        assert abs(s["f1"] - 1.0) < 1e-9
        assert abs(s["pr_auc"] - 1.0) < 1e-9


def test_near_miss_improves_with_scale():
    # Prediction one px right of the burn: miss at 1x1, hit at 2x2.
    prob = np.zeros((4, 4))
    gt = np.zeros((4, 4), dtype=int)
    gt[0, 0] = 1
    prob[0, 1] = 1.0  # predicted next to the actual burn
    out = pyramid_pool_stats(prob, gt, block_sizes=[1, 2])
    by_block = {s["block"]: s for s in out["levels"]}
    assert by_block[1]["f1"] == 0.0            # no pixel overlap
    assert abs(by_block[2]["f1"] - 1.0) < 1e-9  # same 2x2 block -> perfect


def test_pooled_positive_counts_are_maxes():
    # Two separated burns: 2 positives at 1x1 and 2x2, one block at 4x4.
    gt = np.zeros((4, 4), dtype=int)
    gt[0, 0] = 1
    gt[3, 3] = 1
    prob = gt.astype(float)
    out = pyramid_pool_stats(prob, gt, block_sizes=[1, 2, 4])
    n_pos = {s["block"]: s["n_truth"] for s in out["levels"]}
    assert n_pos == {1: 2, 2: 2, 4: 1}


def test_multiclass_uses_fire_complement():
    # Band-first multiclass: P(fire) = 1 - band 1.
    num_classes = 5
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.0, 1.0, size=(num_classes, 32, 32))
    scores /= scores.sum(axis=0, keepdims=True)
    argmax = scores.argmax(axis=0)[None].astype(np.float64)
    predictions = np.concatenate([argmax, scores], axis=0)
    gt = (scores.argmax(axis=0) > 0).astype(int)

    out = pyramid_pool_stats(predictions, gt, block_sizes=[2])
    expected_prob = _block_max_pool(1.0 - scores[0], 2)
    assert 0.0 <= out["levels"][0]["pr_auc"] <= 1.0
    assert out["levels"][0]["grid_h"] == expected_prob.shape[0]


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pyramid_pool_stats(np.zeros((10, 10)), np.zeros((10, 8), dtype=int))


def test_csv_and_plot_written():
    rng = np.random.default_rng(4)
    prob = rng.uniform(size=(64, 64))
    gt = (rng.uniform(size=(64, 64)) < 0.2).astype(int)
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "pyramid.csv")
        png_path = os.path.join(d, "pyramid.png")
        out = pyramid_pool_stats(prob, gt, csv_path=csv_path, plot=png_path)
        assert os.path.exists(csv_path)
        import csv as _csv
        with open(csv_path) as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) == len(out["levels"])
        assert set(rows[0].keys()) == {
            "block", "grid_h", "grid_w", "n_blocks", "n_truth", "n_pred",
            "precision", "recall", "f1", "pr_auc"}
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pass
        else:
            assert os.path.exists(png_path)
