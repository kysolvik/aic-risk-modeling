"""Prediction-time stats resolution matches training; container template path."""

import os

_REPO = os.path.join(os.path.dirname(__file__), "..")
_PREDICT = os.path.join(_REPO, "scripts", "predict", "predict.py")


def _load_from_predict(*names):
    """Exec just the named top-level defs/assignments out of predict.py."""
    src = open(_PREDICT).read()
    ns = {"os": os}
    for name in names:
        marker = f"def {name}"
        start = src.index(marker)
        end = src.index("\ndef ", start + len(marker))
        exec(src[start:end], ns)
    return ns


def test_explicit_flag_wins():
    r = _load_from_predict("resolve_stats_path")["resolve_stats_path"]
    got = r("gs://b/explicit.json", {"stats_path": "gs://b/cfg.json"}, "d/")
    assert got == "gs://b/explicit.json", got


def test_config_stats_path_is_used():
    r = _load_from_predict("resolve_stats_path")["resolve_stats_path"]
    got = r(None, {"stats_path": "gs://b/pooled.json"}, "gs://b/allpreds_2025/")
    assert got == "gs://b/pooled.json", got


def test_legacy_fallback_preserved():
    """quickrun_preds.sh relies on this: v32-v36 configs carry no stats_path."""
    r = _load_from_predict("resolve_stats_path")["resolve_stats_path"]
    got = r(None, {}, "../data/allpreds_2023/")
    assert got == "../data/allpreds_2023/stats.pbtxt", got


def test_no_double_slash_on_gcs():
    """'gs://b/d//stats.pbtxt' is a DIFFERENT object from 'gs://b/d/stats.pbtxt'."""
    r = _load_from_predict("resolve_stats_path")["resolve_stats_path"]
    assert r(None, {}, "gs://b/d/") == "gs://b/d/stats.pbtxt"
    assert r(None, {}, "gs://b/d") == "gs://b/d/stats.pbtxt"


def test_profile_template_resolves_off_file_not_cwd():
    """The container runs from /app; the old './out/example.tif' was cwd-relative."""
    src = open(_PREDICT).read()
    assert "'./out/example.tif'" not in src, "cwd-relative template path is back"
    assert "DEFAULT_PROFILE_TEMPLATE" in src
    for name in ("example.tif", "example_v3.tif"):
        assert os.path.isfile(os.path.join(_REPO, "assets", name)), \
            f"assets/{name} missing (check the !assets/*.tif negation in .gitignore)"
