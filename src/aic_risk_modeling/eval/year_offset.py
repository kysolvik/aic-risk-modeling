"""Offline fit of the factored model's frozen global year offset:

    gamma(t) = b0 + b_soi * z(SOI_{Oct-Dec, Y-1}) + b_prev * z(log basin burn_{Y-1})

Emitted mean-centered over the fit years, so gamma is independent of pos_weight."""

import numpy as np
import pandas as pd

TARGETS = {"bd": ["burn_bd"], "snpp": ["burn_snpp"], "ft": ["burn_ft"],
           "union_sum": ["burn_bd", "burn_snpp"],
           "union3": ["burn_bd", "burn_snpp", "burn_mod14"]}
PREV_BANDS = {"bd": ["im_BurnDate_-1_mean"],
              "snpp": ["im_viirs_snpp_-1_mean"],
              "union_sum": ["im_BurnDate_-1_mean", "im_viirs_snpp_-1_mean"],
              "union3": ["im_BurnDate_-1_mean", "im_viirs_snpp_-1_mean", "im_mod14_-1_mean"]}
SOI_COL = "md_soi_y1ond"
CHIP_PIXELS = 128 * 128


def _z(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std()


def load_panel(path, target="bd", prev_burn="union_sum", space="log1p"):
    """Chip-year frame with response `y` and year regressors zsoi/zprev (fullgrid chip panel)."""
    d = pd.read_parquet(path)
    cols = ["md_id", "year", "md_x", "md_y", SOI_COL] + list(TARGETS[target])
    cols += [c for c in PREV_BANDS[prev_burn] if c not in cols]
    d = d[list(dict.fromkeys(cols))].dropna().copy()

    burn = d[list(TARGETS[target])].sum(axis=1).to_numpy(dtype=float)
    if space == "log1p":
        d["y"] = np.log1p(burn)
    elif space == "logit":
        denom = CHIP_PIXELS * len(TARGETS[target])
        p = (burn + 0.5) / (denom + 1.0)
        d["y"] = np.log(p / (1.0 - p))
    else:
        raise ValueError(f"unknown space {space!r}")

    soi = d.groupby("year")[SOI_COL].first()
    d["zsoi"] = d.year.map((soi - soi.mean()) / soi.std())
    prev = sum(d.groupby("year")[c].mean() for c in PREV_BANDS[prev_burn])
    lprev = np.log1p(prev)
    d["zprev"] = d.year.map((lprev - lprev.mean()) / lprev.std())

    d["zlat"] = _z(d.md_y)
    d["zlon"] = _z(d.md_x)
    d["zne"] = _z(d.md_y + d.md_x)              # NE-SW axis
    return d


def load_target_panel(path, target="bd", space="logit", emit_through=None):
    """load_panel for the long targets-only panel; prev-burn is the chip's lagged MCD64 count.

    emit_through adds predict-only years to d.attrs["per_year"]; they never enter a fit."""
    if space != "logit":
        raise ValueError("the target panel only supports space='logit'")
    raw = pd.read_parquet(path)
    col = f"burn_{target}"
    d = raw.dropna(subset=[col, "prev_bd", SOI_COL]).copy()
    p = (d[col] + 0.5) / (CHIP_PIXELS + 1.0)
    d["y"] = np.log(p / (1.0 - p))
    d["burn_w"] = d[col].astype(float)

    per_year = d.groupby("year").agg(soi=(SOI_COL, "first"), prev=("prev_bd", "mean"))
    last = int(per_year.index.max())
    if emit_through is not None and emit_through > last:
        from aic_risk_modeling.preprocess.climate_indices import download_clim_indices
        s = download_clim_indices("soi", last - 1, emit_through - 1)["metric"]
        basin = raw.groupby("year")["burn_bd"].mean()
        for y in range(last + 1, emit_through + 1):
            ond = s[(s.index.year == y - 1) & (s.index.month >= 10)]
            if len(ond) != 3 or (y - 1) not in basin.index:
                raise ValueError(f"cannot emit {y}: needs SOI Oct-Dec {y - 1} and MCD64 {y - 1}")
            per_year.loc[y] = {"soi": float(ond.mean()), "prev": float(basin.loc[y - 1])}
        ond_last = s[(s.index.year == last - 1) & (s.index.month >= 10)].mean()
        if not np.isclose(ond_last, per_year.loc[last, "soi"], atol=1e-9):
            raise ValueError("NOAA SOI Oct-Dec disagrees with the panel -- calendar misaligned")
    lprev = np.log1p(per_year.prev)
    per_year["zsoi"] = (per_year.soi - per_year.soi.mean()) / per_year.soi.std()
    per_year["zprev"] = (lprev - lprev.mean()) / lprev.std()
    d["zsoi"] = d.year.map(per_year.zsoi)
    d["zprev"] = d.year.map(per_year.zprev)
    d.attrs["per_year"] = per_year.rename_axis("year").reset_index()
    return d


def _chip_weights(tr, weighting):
    """Per-chip weights from training rows ('equal' -> None, 'burn' -> mean burn), summing to 1."""
    if weighting == "equal":
        return None
    if weighting != "burn":
        raise ValueError(f"unknown weighting {weighting!r}")
    if "burn_w" not in tr:
        raise ValueError("weighting='burn' needs the target panel (load_target_panel)")
    w = tr.groupby("md_id")["burn_w"].mean()
    if w.sum() <= 0:
        raise ValueError("no burn in the training years; cannot burn-weight")
    return w / w.sum()


def _lstsq(A, y, row_w=None):
    if row_w is None:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    else:
        sw = np.sqrt(row_w)
        beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return beta


def _wmean(v, w):
    return float(v.mean()) if w is None else float((v * w).sum() / w.sum())


def _design(frame, terms):
    cols = [np.ones(len(frame))]
    for a, b in terms:
        cols.append(frame[a].to_numpy() * frame[b].to_numpy() if b else frame[a].to_numpy())
    return np.column_stack(cols)


def evaluate(d, terms, protocol="loyo", exclude_prev_from_clim=False, min_train_years=5,
             weighting="equal"):
    """Out-of-sample fit of the year effect ('loyo' or 'forward').

    Chip climatology uses training years only (in-sample climatology leaks the held-out year)."""
    years = sorted(d.year.unique())
    pred_year, act_year, betas = {}, {}, []
    for i, t in enumerate(years):
        if protocol == "forward" and i < min_train_years:
            continue
        drop = {t} | ({t - 1} if exclude_prev_from_clim else set())
        tr_years = [y for y in years if y not in drop and (protocol != "forward" or y < t)]
        tr, te = d[d.year.isin(tr_years)], d[d.year == t]

        clim = tr.groupby("md_id")["y"].mean()
        tr_res = (tr.y - tr.md_id.map(clim)).to_numpy()
        te = te[te.md_id.isin(clim.index)]
        te_res = (te.y - te.md_id.map(clim)).to_numpy()

        w = _chip_weights(tr, weighting)
        tr_w = None if w is None else tr.md_id.map(w).to_numpy()
        te_w = None if w is None else te.md_id.map(w).to_numpy()
        beta = _lstsq(_design(tr, terms), tr_res, tr_w)
        betas.append(beta)
        pred_year[t] = _wmean(_design(te, terms) @ beta, te_w)
        act_year[t] = _wmean(te_res, te_w)

    ks = sorted(pred_year)
    P = np.array([pred_year[k] for k in ks])
    A = np.array([act_year[k] for k in ks])
    turns = int(sum(np.sign(A[i] - A[i - 1]) == np.sign(P[i] - P[i - 1]) for i in range(1, len(ks))))
    B = np.array(betas)
    return {
        "years": ks,
        "pred_year": P, "act_year": A,
        "r_year": float(np.corrcoef(P, A)[0, 1]),
        "amplitude": float(np.exp(P.max() - P.min())),
        "actual_amplitude": float(np.exp(A.max() - A.min())),
        "turns": turns, "n_turns": len(ks) - 1,
        "beta_mean": B.mean(0), "beta_sd": B.std(0),
        "sign_consistent": bool(np.all(np.sign(B[:, 1:]) == np.sign(B[0, 1:]), axis=0).all()),
    }


def jackknife_r(result):
    """(min, max, most-influential year) of r_year under leave-one-year-out."""
    P, A, years = result["pred_year"], result["act_year"], result["years"]
    rs = [(float(np.corrcoef(np.delete(P, i), np.delete(A, i))[0, 1]), int(years[i]))
          for i in range(len(years))]
    return min(r for r, _ in rs), max(r for r, _ in rs), min(rs)[1]


SOI = [("zsoi", None)]
PREV = [("zprev", None)]
BOTH = [("zsoi", None), ("zprev", None)]


def fit_final(d, fit_years, terms=BOTH, weighting="equal"):
    """In-sample fit over fit_years -> (beta, chip climatology)."""
    tr = d[d.year.isin(fit_years)]
    clim = tr.groupby("md_id")["y"].mean()
    res = (tr.y - tr.md_id.map(clim)).to_numpy()
    w = _chip_weights(tr, weighting)
    beta = _lstsq(_design(tr, terms), res, None if w is None else tr.md_id.map(w).to_numpy())
    return beta, clim


def build_offsets(d, beta, terms=BOTH, center_years=None):
    """Per-year gamma mean-centered over center_years -> ({year: gamma}, mean)."""
    per_year = d.attrs.get("per_year")
    if per_year is None:
        per_year = d.groupby("year").first().reset_index()
    vals = _design(per_year, terms) @ beta
    out = dict(zip(per_year.year.astype(int), vals))
    ref = center_years if center_years is not None else sorted(out)
    mu = float(np.mean([out[y] for y in ref if y in out]))
    return {int(y): float(v - mu) for y, v in out.items()}, mu


# Regression references from out/chip_panel/panel.parquet; changing a number means the fit changed.
REF = dict(target="bd", prev_burn="bd", space="log1p")


def _chk(results, name, got, want, tol):
    ok = abs(got - want) <= tol
    results.append((ok, f"{name}: {got:+.4f} (expect {want:+.4f} +/- {tol})"))
    return ok


def run_checks(panel, verbose=True):
    d = load_panel(panel, **REF)
    r = []

    loyo = {k: evaluate(d, t) for k, t in [("soi", SOI), ("prev", PREV), ("both", BOTH)]}
    _chk(r, "LOYO r_year SOI-only", loyo["soi"]["r_year"], 0.278, 0.01)
    _chk(r, "LOYO r_year prev-only", loyo["prev"]["r_year"], 0.422, 0.01)
    _chk(r, "LOYO r_year both", loyo["both"]["r_year"], 0.500, 0.01)

    b_soi_only = evaluate(d, SOI)
    _chk(r, "LOYO b_soi mean", b_soi_only["beta_mean"][1], -0.152, 0.005)
    _chk(r, "LOYO b_soi sd", b_soi_only["beta_sd"][1], 0.029, 0.005)
    r.append((b_soi_only["sign_consistent"], "LOYO b_soi sign consistent across all folds"))

    fwd = {k: evaluate(d, t, protocol="forward", exclude_prev_from_clim=True)
           for k, t in [("soi", SOI), ("both", BOTH)]}
    _chk(r, "forward-chained r_year both", fwd["both"]["r_year"], 0.775, 0.01)
    _chk(r, "forward-chained r_year SOI-only", fwd["soi"]["r_year"], 0.422, 0.01)
    r.append((fwd["both"]["turns"] == 6 and fwd["both"]["n_turns"] == 7,
              f"forward-chained turns both: {fwd['both']['turns']}/{fwd['both']['n_turns']} (expect 6/7)"))
    r.append((fwd["soi"]["turns"] == 5 and fwd["soi"]["n_turns"] == 7,
              f"forward-chained turns SOI-only: {fwd['soi']['turns']}/{fwd['soi']['n_turns']} (expect 5/7)"))

    # Leak-free: dropping t-1 from the climatology must not move b_prev.
    incl = evaluate(d, BOTH, exclude_prev_from_clim=False)["beta_mean"][2]
    excl = evaluate(d, BOTH, exclude_prev_from_clim=True)["beta_mean"][2]
    _chk(r, "b_prev with t-1 in climatology", incl, -0.1650, 0.005)
    _chk(r, "b_prev with t-1 excluded", excl, -0.1648, 0.005)
    r.append((abs(incl - excl) < 0.005,
              f"leak-free invariant: |b_prev shift| = {abs(incl - excl):.5f} < 0.005"))

    # Sign guard: b_prev > 0 would be persistence, which lags every turn.
    r.append((excl < 0, f"sign guard: b_prev = {excl:+.4f} < 0 (mean-reverting, not persistence)"))

    # A mean-zero spatial basis times a year scalar can't move the year effect.
    base = evaluate(d, BOTH)
    for basis in ("zlat", "zlon", "zne"):
        alt = evaluate(d, BOTH + [("zsoi", basis)])
        same = (round(alt["r_year"], 3) == round(base["r_year"], 3)
                and round(alt["amplitude"], 3) == round(base["amplitude"], 3))
        r.append((same, f"orthogonality SOI x {basis}: r_year {alt['r_year']:.3f} vs "
                        f"{base['r_year']:.3f}, amp {alt['amplitude']:.3f} vs {base['amplitude']:.3f}"))

    if verbose:
        for ok, msg in r:
            print(f"  [{'ok ' if ok else 'FAIL'}] {msg}")
    n_bad = sum(1 for ok, _ in r if not ok)
    print(f"\n{len(r) - n_bad}/{len(r)} checks passed")
    return n_bad == 0
