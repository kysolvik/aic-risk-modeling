"""label_agreement_stats: contingency-table reduce/combine on synthetic chips."""

import os
import shutil
import tempfile

import numpy as np

import label_agreement_stats as las

EDGES = [0.25, 0.75]
NO_DILATE = {n: 0 for n in las.PRODUCTS}


def _chip(seed=0, forest=0.9, density=(0.02, 0.015, 0.03)):
    """A 128x128 record of day-of-year bands plus the forest-fraction band."""
    rng = np.random.default_rng(seed)
    rec = {}
    for name, p in zip(las.PRODUCTS, density):
        doy = np.where(rng.random((128, 128)) < p,
                       rng.integers(1, 367, (128, 128)), 0)
        rec[name] = doy.astype(np.float32)
    rec[las.STRAT_BAND] = np.full((128, 128), forest, dtype=np.float32)
    return rec


def _table(rows, mode):
    t = np.zeros(8, dtype=np.int64)
    for r in rows:
        if r["mode"] == mode:
            t[r["pattern"]] += r["count"]
    return t


def test_counts_are_complete_and_match_the_marginals():
    rec = _chip(seed=1)
    rows, _ = las.reduce_record(rec, EDGES, NO_DILATE, np.random.default_rng(0))
    exact = _table(rows, "exact")
    assert exact.sum() == 128 * 128, "every pixel must land in exactly one cell"

    for r, name in enumerate(las.PRODUCTS):
        from_table = sum(exact[k] for k in range(8) if k & (1 << r))
        assert from_table == int((rec[name] > 0).sum()), name


def test_exact_union_equals_the_frozen_combine_any_label():
    rec = _chip(seed=2)
    rows, _ = las.reduce_record(rec, EDGES, NO_DILATE, np.random.default_rng(0))
    exact = _table(rows, "exact")
    union_from_table = exact.sum() - exact[0]

    stacked = np.stack([rec[n] > 0 for n in las.PRODUCTS], axis=-1)
    assert union_from_table == int(stacked.any(axis=-1).sum())


def test_dilation_only_moves_the_tolerant_table():
    rec = _chip(seed=3)
    rows, _ = las.reduce_record(rec, EDGES, {"im_BurnDate_0": 0,
                                             "im_mod14_0": 1,
                                             "im_viirs_snpp_0": 1},
                                np.random.default_rng(0))
    exact, tolerant = _table(rows, "exact"), _table(rows, "tolerant")
    assert exact.sum() == tolerant.sum() == 128 * 128
    # Dilation only adds detections: all-negative shrinks, co-detection grows.
    assert tolerant[0] < exact[0]
    assert tolerant[7] >= exact[7]
    # MCD64 is not dilated, so its marginal is untouched.
    assert (sum(tolerant[k] for k in range(8) if k & 1)
            == sum(exact[k] for k in range(8) if k & 1))


def test_stratum_assignment_follows_forest_fraction():
    for forest, want in ((0.1, 0), (0.5, 1), (0.9, 2)):
        rows, _ = las.reduce_record(_chip(seed=4, forest=forest), EDGES,
                                    NO_DILATE, np.random.default_rng(0))
        assert {r["stratum"] for r in rows} == {want}, forest


def test_delta_rows_carry_an_observed_and_a_null_arm():
    rec = _chip(seed=5, density=(0.25, 0.25, 0.25))
    _, deltas = las.reduce_record(rec, EDGES, NO_DILATE,
                                  np.random.default_rng(0))
    assert deltas, "dense chips must produce co-detections"
    kinds = {d["kind"] for d in deltas}
    assert kinds == {"observed", "null"}, kinds
    assert {d["pair"] for d in deltas} == {"mcd64|mod14", "mcd64|viirs",
                                           "mod14|viirs"}
    for d in deltas:
        assert 0 <= d["bin"] <= len(las.DOY_EDGES)


def test_combine_sums_shards_and_prints():
    """The --combine path, end to end, on a local directory."""
    import pandas as pd
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "shards"))
        totals = np.zeros(8, dtype=np.int64)
        for i in range(3):
            rows, _ = las.reduce_record(_chip(seed=10 + i), EDGES, NO_DILATE,
                                        np.random.default_rng(i))
            df = (pd.DataFrame(rows)
                  .groupby(["mode", "stratum", "pattern"], as_index=False)["count"].sum())
            df["year"] = 2019
            df.to_parquet(os.path.join(tmp, "shards", f"2019_{i}_patterns.parquet"),
                          index=False)
            totals += _table(rows, "exact")

        las.combine(tmp, EDGES)
        out = pd.read_parquet(os.path.join(tmp, "patterns.parquet"))
        got = (out[out["mode"] == "exact"].groupby("pattern")["count"].sum()
               .reindex(range(8), fill_value=0).to_numpy())
        assert (got == totals).all(), (got, totals)
        assert out["count"].sum() == 2 * 3 * 128 * 128  # two modes, three chips
    finally:
        shutil.rmtree(tmp)
