"""Prediction-time stats resolution matches training; container template path."""

import os

import pytest

pytest.importorskip("torch")
from aic_risk_modeling.predict.core import resolve_stats_path  # noqa: E402

_REPO = os.path.join(os.path.dirname(__file__), "..")
_PREDICT = os.path.join(_REPO, "scripts", "predict", "predict.py")


def test_explicit_flag_wins():
    got = resolve_stats_path("gs://b/explicit.json", {"stats_path": "gs://b/cfg.json"}, "d/")
    assert got == "gs://b/explicit.json", got


def test_config_stats_path_is_used():
    got = resolve_stats_path(None, {"stats_path": "gs://b/pooled.json"}, "gs://b/allpreds_2025/")
    assert got == "gs://b/pooled.json", got


def test_legacy_fallback_preserved():
    """Configs without a stats_path fall back to <data_dir>/stats.pbtxt."""
    got = resolve_stats_path(None, {}, "../data/allpreds_2023/")
    assert got == "../data/allpreds_2023/stats.pbtxt", got


def test_no_double_slash_on_gcs():
    """'gs://b/d//stats.pbtxt' is a DIFFERENT object from 'gs://b/d/stats.pbtxt'."""
    assert resolve_stats_path(None, {}, "gs://b/d/") == "gs://b/d/stats.pbtxt"
    assert resolve_stats_path(None, {}, "gs://b/d") == "gs://b/d/stats.pbtxt"


def test_profile_template_resolves_off_file_not_cwd():
    """The container runs from /app, so the template path must not be cwd-relative."""
    src = open(_PREDICT).read()
    assert "DEFAULT_PROFILE_TEMPLATE" in src and "__file__" in src
    assert os.path.isfile(os.path.join(_REPO, "assets", "example_v3.tif")), \
        "assets/example_v3.tif missing (check the !assets/*.tif negation in .gitignore)"
