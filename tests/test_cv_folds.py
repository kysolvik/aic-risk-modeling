"""cv_make_folds: fold structure and leakage guards of the evaluation protocol."""

import copy
import csv
import json
import os
import tempfile

import pytest

import cv_make_folds as mk

REPO = os.path.join(os.path.dirname(__file__), "..")

FIN = set(mk.FINAL_TEST_YEARS)


def _by_id():
    return {s["fold_id"]: s for s in mk.protocol_specs()}


def test_fold_structure():
    specs = _by_id()
    for t in range(2018, 2023):
        s = specs[f"fwdpair_{t}"]
        assert s["train"] == list(range(2013, t)), s
        assert s["val"] == s["eval"] == [t, t + 1], s
    f = specs["final_all"]
    assert f["train"] == list(range(2013, 2024)) and f["val"] == [] and f["eval"] == [2024, 2025]
    for drop in mk.DROP_SETS:
        arm = specs[f"final_drop{drop[0]}-{drop[1]}"]
        assert arm["base_fold"] == "final_all" and arm["val"] == [], arm
    p = specs["lead1_2023"]
    assert p["train"] == list(range(2013, 2023)) and p["val"] == p["eval"] == [2023], p
    assert {s["stage"] for s in specs.values()} == set(mk.STAGES)
    counts = {st: sum(s["stage"] == st for s in specs.values()) for st in mk.STAGES}
    assert counts == {"folds": 5, "seedrep": 1, "final": 1, "ablateA": 6, "ablateB": 3,
                      "posthoc": 1}, counts


def test_no_fold_touches_the_final_years():
    for s in mk.protocol_specs():
        assert mk.protocol_violations(s) == [], (s["fold_id"], mk.protocol_violations(s))
        used = set(s["train"]) | set(s["val"])
        if s["stage"] in ("folds", "seedrep", "posthoc"):
            assert not (used | set(s["eval"])) & FIN, s["fold_id"]
        if s["stage"] in ("final", "ablateB"):
            assert not used & FIN, s["fold_id"]
        if s["stage"] == "ablateA":
            assert not (set(s["eval"]) | set(s["val"])) & FIN, s["fold_id"]


def test_leakage_guards_fire():
    specs = _by_id()
    bad = copy.deepcopy(specs["fwdpair_2020"])
    bad["train"].append(2024)
    assert mk.protocol_violations(bad), "fold training on 2024 not caught"
    bad = copy.deepcopy(specs["final_all"])
    bad["val"] = [2024]
    assert mk.protocol_violations(bad), "final early-stopping on a test year not caught"
    bad = copy.deepcopy(specs["lead1_2023"])
    bad["val"] = bad["eval"] = [2023, 2024]
    assert mk.protocol_violations(bad), "post-hoc fold early-stopping on 2024 not caught"
    bad = copy.deepcopy(specs["fwdpair_2018_add2024"])
    bad["eval"] = [2024, 2025]
    bad["val"] = [2024, 2025]
    assert mk.protocol_violations(bad), "ablation A evaluating on the final years not caught"
    bad = copy.deepcopy(specs["fwdpair_2019"])
    bad["train"].append(2019)
    assert mk.protocol_violations(bad), "train/eval overlap not caught"
    bad = copy.deepcopy(specs["fwdpair_2019"])
    bad["val"] = [2018]
    assert mk.protocol_violations(bad), "fold with val != eval not caught"


def test_arms_differ_from_base_by_exactly_the_intended_years():
    specs = _by_id()
    for s in specs.values():
        if s["base_fold"]:
            assert mk.arm_diff_errors(s, specs[s["base_fold"]]) == [], s["fold_id"]
    arm = copy.deepcopy(specs["fwdpair_2020_add2024"])
    arm["train"].append(2023)
    assert mk.arm_diff_errors(arm, specs["fwdpair_2020"]), "extra added year not caught"
    arm = copy.deepcopy(specs["final_drop2017-2020"])
    arm["train"].remove(2013)
    assert mk.arm_diff_errors(arm, specs["final_all"]), "extra dropped year not caught"
    rep = copy.deepcopy(specs["fwdpair_2020_s55"])
    rep["seed"] = specs["fwdpair_2020"]["seed"]
    assert mk.arm_diff_errors(rep, specs["fwdpair_2020"]), "seed replicate with the base seed not caught"
    arm = copy.deepcopy(specs["fwdpair_2018_add2023"])
    arm["seed"] = 99
    assert mk.arm_diff_errors(arm, specs["fwdpair_2018"]), "ablation arm changing the seed not caught"


def test_stats_and_gamma_names():
    specs = mk.protocol_specs()
    res = mk.protocol_resources(specs)
    for s in specs:
        fid = s["fold_id"]
        assert res[fid]["gamma_name"] == fid, "gamma is refit per job"
        if s["stage"] in ("ablateA", "ablateB"):
            assert res[fid]["stats_name"] == res[s["base_fold"]]["stats_name"], "arm must keep base stats"
        else:
            assert res[fid]["stats_name"] == fid


def test_model_paths_unique_and_nested():
    specs = mk.protocol_specs()
    paths = [mk.arch_paths(a, s["fold_id"])["model_gs"] for a in ("arch_a", "arch_b") for s in specs]
    assert len(paths) == len(set(paths)), "model_output_path collision"
    p = mk.arch_paths("unet_v3p_union4", "final_all")
    assert p["config_local"] == "configs/cv/unet_v3p_union4/final_all.json"
    assert p["model_gs"].endswith("/models/cv/unet_v3p_union4/final_all.pt")


def test_config_guards(repo_config):
    base = repo_config("factored_v3p_union4_monthlyattn_wide_yeargain")
    assert mk.config_guard_errors(base) == []
    leak = copy.deepcopy(base)
    leak["input_features"]["im_annual"]["feature_names"].append("im_lossyear")
    assert mk.config_guard_errors(leak), "im_lossyear input not caught"
    year_in = copy.deepcopy(base)
    year_in["decoder_config"]["year_group"] = None
    assert mk.config_guard_errors(year_in), "md_year as a network input not caught"
    routed = copy.deepcopy(base)
    routed["decoder_config"]["pixel_groups"].append("md_year")
    assert mk.config_guard_errors(routed), "md_year routed into the network not caught"
    for name in ("vit_test_v3p_union4", "mlp_v3p_union4_flat"):
        cfg = repo_config(name)
        assert mk.config_guard_errors(cfg) == [], name
    flat = repo_config("mlp_v3p_union4_flat")
    flat["input_features"]["im_all"]["feature_names"].append("im_BurnDate_viirs_0")
    assert mk.config_guard_errors(flat), "step-0 target band as a flat input not caught"
    stepped = copy.deepcopy(base)
    stepped["input_features"]["im_annual"]["timesteps"].append(0)
    assert mk.config_guard_errors(stepped), "timestep 0 in a stacked group not caught"


def test_gates():
    with tempfile.TemporaryDirectory() as d:
        frozen, scored = os.path.join(d, "selection_frozen.json"), os.path.join(d, "final_scored.json")
        g = lambda st, arch: mk.gate_errors(st, arch, frozen=frozen, scored=scored)  # noqa: E731
        assert g("folds", "a") == []
        for st in ("final", "seedrep", "ablateA", "ablateB"):
            assert g(st, "a"), f"{st} open without selection_frozen.json"
        json.dump({"arch": "a"}, open(frozen, "w"))
        assert g("final", "a") == [] and g("final", "b") == [], "final is for every reported arch"
        assert g("seedrep", "a") == [] and g("seedrep", "b"), "seedrep is selected-arch only"
        assert g("ablateA", "a") and g("ablateB", "a"), "ablations open before the final is scored"
        json.dump({"scored_at": "now"}, open(scored, "w"))
        assert g("ablateA", "a") == [] and g("ablateB", "a") == []
        assert g("ablateA", "b"), "ablations open for a non-selected arch"
        assert g("bogus", "a"), "unknown stage accepted"
    with tempfile.TemporaryDirectory() as d:
        frozen, scored = os.path.join(d, "selection_frozen.json"), os.path.join(d, "final_scored.json")
        early = lambda st: mk.gate_errors(st, "a", frozen=frozen, scored=scored,  # noqa: E731
                                          allow_early_final=True)
        assert early("final") == [], "early-final override did not open the final"
        for st in ("seedrep", "ablateA", "ablateB"):
            assert early(st), f"early-final override opened {st}"


def test_no_val_final_config(repo_config):
    base = repo_config("factored_v3p_union4_monthlyattn_wide_yeargain")
    base["val_cache_dir"] = "/tmp/vc"
    spec = _by_id()["final_all"]
    with tempfile.TemporaryDirectory() as d:
        paths = {"config_local": os.path.join(d, "final_all.json"), "model_gs": "gs://x/final_all.pt"}
        cfg = mk.make_protocol_config(base, spec, paths, "gs://x/stats.json", "gs://x/g.json")
    assert cfg["checkpoint_metric"] == "last", cfg["checkpoint_metric"]
    for k in mk.NO_VAL_DROP:
        assert k not in cfg, f"no-val config still sets {k}"
    assert cfg["epochs"] == mk.PROTOCOL_BUDGET["epochs"]
    fold = mk.make_protocol_config(base, _by_id()["fwdpair_2020"], dict(paths), "s", "g")
    assert fold["checkpoint_metric"] == "pr_auc" and fold["early_stopping_patience"] == 8
    assert len(fold["val_data_dirs"]) == 2


def test_generated_configs_respect_the_protocol():
    manifest = os.path.join(REPO, "out", "cv", "protocol.csv")
    if not os.path.exists(manifest):
        pytest.skip("run cv_make_folds.py first")
    for r in csv.DictReader(open(manifest)):
        cfg = json.load(open(os.path.join(REPO, r["config_local"])))
        yrs = lambda dirs: {int(x.rstrip("/").rsplit("_", 1)[1]) for x in dirs}  # noqa: E731
        train, val = yrs(cfg["data_dirs"]), yrs(cfg.get("val_data_dirs", []))
        if r["stage"] in ("folds", "seedrep", "final", "ablateB"):
            assert not (train | val) & FIN, f"{r['arch']}/{r['fold_id']} config touches 2024/2025"
        assert cfg["model_output_path"] == r["model_gs"]
        want = mk.job_budget(mk.PROTOCOL_BUDGET, bool(val))
        if r.get("steps_per_epoch"):                 # per-fold value (--chips_per_year)
            want["steps_per_epoch"] = int(r["steps_per_epoch"])
        for k, v in want.items():
            assert cfg.get(k) == v, f"{r['arch']}/{r['fold_id']}: {k}={cfg.get(k)}"
