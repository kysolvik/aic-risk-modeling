#!/usr/bin/env python
"""Fit a latent-class measurement model to the three fire products.

Consumes the contingency tables from label_agreement_stats.py and estimates, per
land-cover stratum g and product r:

    pi_g   = P(the cell burned | g)                       prevalence
    s_rg   = P(product r detects | burned, g)             sensitivity
    f_r    = P(product r detects | not burned)            false-positive rate

by EM, then emits the `sample_weight` block that data_loader.build_confidence_*
consumes, with pi/s/f already turned into the log-likelihood ratios the pipeline
sums per pixel.

WHY THIS IS IDENTIFIABLE. Three binary tests in one population give 7 degrees of
freedom for 7 parameters -- exactly identified, so s and f come out but nothing
is left over to check conditional independence with. Holding f_r shared across
strata (the Hui-Walter 1980 constraint) while letting pi_g and s_rg vary gives
7G df against 4G + 3 parameters, so G strata leave 3G - 3 df for a real
goodness-of-fit test. G=3 forest-fraction bins leave 6.

WHAT THE FIT STATISTIC IS FOR. Two dependencies are known a priori and both
inflate agreement: MCD64A1's Collection 6 algorithm seeds its burned training
samples and priors from a cumulative MOD14 active-fire composite (Giglio et al.
2018), and MOD14/VIIRS share an early-afternoon overpass, so their omissions are
driven by the same cloud and fire-timing draws. If G^2 rejects, the residual
excess over the independence model is emitted as first-order `pair_llr`
discounts -- and a large residual after that is the signal to fit proper
covariance terms, or to collapse MOD14 into the MCD64 channel, rather than to
ship the numbers.

STRATIFYING BY YEAR IS NOT COSMETIC. notes/v43_union_target.txt found the VIIRS
label drifts denser across 2013-2022 while MODIS holds flat. Absorbed into a
year-varying s_VIIRS, that drift lives in the measurement model; left alone it
lives in the label, where a model reads "the label got denser" as "more fire".

Run directly (no cloud, recovers known parameters from simulated tables):
    .venv/bin/python scripts/analysis/fit_label_model.py --synthetic

On real tables:
    .venv/bin/python scripts/analysis/fit_label_model.py \
        --patterns out/label_model/patterns.parquet \
        --deltas out/label_model/deltas.parquet \
        --mode tolerant --pos_weight 10 --out out/label_model/label_model_v3.json
"""

import argparse
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

PRODUCTS = ["im_BurnDate_0", "im_mod14_0", "im_viirs_snpp_0"]
SHORT = ["mcd64", "mod14", "viirs"]
N_PROD = 3
N_PATTERN = 1 << N_PROD

# pattern k has product r detecting iff bit r is set
BITS = np.array([[(k >> r) & 1 for r in range(N_PROD)] for k in range(N_PATTERN)],
                dtype=np.float64)
UNION = BITS.any(axis=1)

_EPS = 1e-12


def _clip(x, lo=1e-6, hi=1 - 1e-6):
    return np.clip(x, lo, hi)


def _pattern_likelihood(rates):
    """P(pattern | rates) for each pattern. rates: (..., N_PROD) detection probs."""
    rates = rates[..., None, :]                      # (..., 1, N_PROD)
    per = np.where(BITS > 0, rates, 1.0 - rates)     # (..., N_PATTERN, N_PROD)
    return per.prod(axis=-1)                         # (..., N_PATTERN)


def em_fit(counts, max_iter=2000, tol=1e-11, n_init=8, seed=0):
    """EM for pi_g, s_rg and a shared f_r from a (G, 8) count table.

    Returns (pi, sens, fpr, loglik). Multiple random inits, best likelihood kept:
    the likelihood is not concave and a bad start can park s below f, which would
    make a detection evidence AGAINST fire.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n_strata = counts.shape[0]
    totals = counts.sum(axis=1)
    rng = np.random.default_rng(seed)
    best = None

    # Moment start: union prevalence is a lower bound on pi (every detector can
    # only miss), so scale it up; sensitivities follow from the marginals.
    union_prev = (counts[:, UNION].sum(axis=1) / np.maximum(totals, 1))
    marg = (counts @ BITS) / np.maximum(totals, 1)[:, None]

    for attempt in range(n_init):
        if attempt == 0:
            pi = _clip(union_prev * 1.6, 1e-4, 0.5)
            sens = _clip(marg / pi[:, None], 0.02, 0.95)
            fpr = _clip(marg.mean(axis=0) * 0.05, 1e-5, 0.05)
        else:
            pi = _clip(union_prev * rng.uniform(1.05, 3.0, n_strata), 1e-4, 0.6)
            sens = _clip(marg / pi[:, None] * rng.uniform(0.5, 1.5, marg.shape),
                         0.02, 0.95)
            fpr = _clip(marg.mean(axis=0) * rng.uniform(0.01, 0.3), 1e-5, 0.1)

        prev = -np.inf
        for _ in range(max_iter):
            like1 = pi[:, None] * _pattern_likelihood(sens)          # (G, 8)
            like0 = (1.0 - pi)[:, None] * _pattern_likelihood(
                np.broadcast_to(fpr, (n_strata, N_PROD)))
            mix = like1 + like0
            loglik = float((counts * np.log(np.maximum(mix, _EPS))).sum())
            resp = like1 / np.maximum(mix, _EPS)                     # P(z=1|g,k)

            wpos = counts * resp
            wneg = counts * (1.0 - resp)
            pi = _clip(wpos.sum(axis=1) / np.maximum(totals, _EPS))
            sens = _clip((wpos @ BITS) / np.maximum(wpos.sum(axis=1), _EPS)[:, None])
            fpr = _clip((wneg.sum(axis=0) @ BITS) / max(wneg.sum(), _EPS))

            if loglik - prev < tol * max(abs(prev), 1.0):
                break
            prev = loglik

        if (sens <= fpr[None, :]).any():
            continue  # degenerate: a detection would argue against fire
        if best is None or loglik > best[-1]:
            best = (pi.copy(), sens.copy(), fpr.copy(), loglik)

    if best is None:
        raise RuntimeError(
            "EM never reached a solution with sensitivity above the "
            "false-positive rate; the products may not be measuring a common "
            "latent event in this stratification")
    return best


def fit_statistic(counts, pi, sens, fpr):
    """Likelihood-ratio G^2 against the saturated table, and its df.

    df = 7G observed free cells - (G prevalences + 3G sensitivities + 3 shared
    false-positive rates) = 3G - 3. Zero or negative df means the model cannot
    be tested, only estimated.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n_strata = counts.shape[0]
    totals = counts.sum(axis=1)
    probs = (pi[:, None] * _pattern_likelihood(sens)
             + (1.0 - pi)[:, None] * _pattern_likelihood(
                 np.broadcast_to(fpr, (n_strata, N_PROD))))
    expected = probs * totals[:, None]
    nz = counts > 0
    g2 = 2.0 * float((counts[nz] * np.log(counts[nz] / np.maximum(expected[nz], _EPS))).sum())
    return g2, 3 * n_strata - 3, expected


def pair_corrections(counts, expected):
    """First-order LLR discounts for pairs the independence model mis-predicts.

    A positively dependent pair co-detects more often than independence implies,
    so seeing both is worth LESS than the sum of two independent detections;
    the discount is minus the log excess. This is a residual correction, not a
    fitted covariance -- if it is large, fit the covariance properly instead of
    leaning on this.
    """
    counts = np.asarray(counts, dtype=np.float64)
    out = []
    for i in range(N_PROD):
        for j in range(i + 1, N_PROD):
            both = (BITS[:, i] > 0) & (BITS[:, j] > 0)
            neither = (BITS[:, i] == 0) & (BITS[:, j] == 0)
            obs_b, exp_b = counts[:, both].sum(), expected[:, both].sum()
            obs_n, exp_n = counts[:, neither].sum(), expected[:, neither].sum()
            out.append({
                "a": i, "b": j, "pair": f"{SHORT[i]}|{SHORT[j]}",
                "both": -math.log(max(obs_b, 1.0) / max(exp_b, _EPS)),
                "neither": -math.log(max(obs_n, 1.0) / max(exp_n, _EPS)),
                "obs_both": int(obs_b), "exp_both": float(exp_b),
            })
    return out


def doy_terms(deltas, edges, min_count=500):
    """|dDOY| log-likelihood ratios per product pair, observed against the null.

    The null shuffles detection dates within a stratum, so it holds "this cell
    burns a lot" fixed and removes only the pairing. A positive LLR in the first
    bins therefore means the two products saw the SAME fire, which is much
    stronger corroboration than two detections months apart.
    """
    if deltas is None or deltas.empty:
        return []
    terms = []
    for pair, grp in deltas.groupby("pair"):
        obs = grp[grp["kind"] == "observed"].groupby("bin")["count"].sum()
        null = grp[grp["kind"] == "null"].groupby("bin")["count"].sum()
        n_bins = len(edges) + 1
        obs = obs.reindex(range(n_bins), fill_value=0).to_numpy(dtype=float)
        null = null.reindex(range(n_bins), fill_value=0).to_numpy(dtype=float)
        if obs.sum() < min_count or null.sum() < min_count:
            continue
        obs_p = (obs + 0.5) / (obs.sum() + 0.5 * n_bins)
        null_p = (null + 0.5) / (null.sum() + 0.5 * n_bins)
        a, b = [SHORT.index(x) for x in pair.split("|")]
        terms.append({"pair": pair, "pairs": [[a, b]], "edges": list(edges),
                      "values": np.log(obs_p / null_p).round(4).tolist()})
    return terms


def posterior_table(pi, sens, fpr, pair_llr=()):
    """q for every (stratum, pattern), the way the pipeline computes it."""
    n_strata = len(pi)
    llr = np.log(pi / (1 - pi))[:, None] + np.zeros((n_strata, N_PATTERN))
    for r in range(N_PROD):
        pos = np.log(sens[:, r] / fpr[r])[:, None]
        neg = np.log((1 - sens[:, r]) / (1 - fpr[r]))[:, None]
        llr += np.where(BITS[None, :, r] > 0, pos, neg)
    for pair in pair_llr:
        i, j = pair["a"], pair["b"]
        both = (BITS[:, i] > 0) & (BITS[:, j] > 0)
        neither = (BITS[:, i] == 0) & (BITS[:, j] == 0)
        llr += np.where(both, pair["both"], np.where(neither, pair["neither"], 0.0))[None, :]
    return 1.0 / (1.0 + np.exp(-llr))


def weight_scales(counts, q, pos_weight):
    """Scale factors that hold mean loss weight at the unweighted baseline.

    Without this an arm can win or lose purely by changing the effective step
    size of a tuned cosine schedule, which would be indistinguishable from a
    real effect.
    """
    counts = np.asarray(counts, dtype=np.float64)
    joint = counts / counts.sum()
    union = UNION[None, :]
    base = float((joint * np.where(union, pos_weight, 1.0)).sum())

    b_hard = np.where(union, q, 1.0 - q)
    w_hard = float((joint * b_hard * np.where(union, pos_weight, 1.0)).sum())
    w_soft = float((joint * (pos_weight * q + 1.0 - q)).sum())
    return {"hard": base / max(w_hard, _EPS), "soft": base / max(w_soft, _EPS),
            "baseline_mean_weight": base}


def bootstrap(counts, n_boot, seed):
    """Percentile CIs by multinomial resampling of each stratum's table."""
    counts = np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    pis, senses, fprs = [], [], []
    for b in range(n_boot):
        draw = np.stack([rng.multinomial(int(row.sum()), row / max(row.sum(), _EPS))
                         for row in counts])
        try:
            pi, sens, fpr, _ = em_fit(draw, n_init=3, seed=seed + b + 1)
        except RuntimeError:
            continue
        pis.append(pi); senses.append(sens); fprs.append(fpr)
    if not pis:
        return None
    pct = lambda a: [np.percentile(a, 2.5, axis=0).tolist(),
                     np.percentile(a, 97.5, axis=0).tolist()]
    return {"n_ok": len(pis), "pi": pct(np.array(pis)),
            "sens": pct(np.array(senses)), "fpr": pct(np.array(fprs))}


def build_block(pi, sens, fpr, pair_llr, doy, edges, scales, soft_label,
                strat_band, forest_edges):
    """The `sample_weight` config block the training pipeline consumes."""
    block = {
        "mode": "confidence",
        "soft_label": bool(soft_label),
        "prior": [round(float(p), 6) for p in pi],
        "products": [
            {"name": name, "dilate": d,
             "sens": [round(float(s), 6) for s in sens[:, r]],
             "fpr": round(float(fpr[r]), 8)}
            for r, (name, d) in enumerate(zip(PRODUCTS, edges["dilate"]))
        ],
        "confidence": {
            "floor": 0.05,
            "weight_scale": round(scales["soft" if soft_label else "hard"], 6),
        },
    }
    if strat_band:
        block["stratify"] = {"feature_name": strat_band,
                             "edges": list(forest_edges)}
    keep = [p for p in pair_llr if abs(p["both"]) > 0.05]
    if keep:
        block["pair_llr"] = [{"a": p["a"], "b": p["b"],
                              "both": round(p["both"], 4),
                              "neither": round(p["neither"], 4)} for p in keep]
    if doy:
        block["doy_llr"] = [{"pairs": t["pairs"], "edges": t["edges"],
                             "values": t["values"]} for t in doy]
    return block


def synthetic_tables(seed=0, n_per_stratum=4_000_000, dependence=0.0,
                     dep_pair=(0, 1)):
    """Simulate tables from known parameters, for recovery and power checks.

    `dependence` tilts the joint detection probability of `dep_pair` within the
    burned class by exp(lambda) -- the Ising-style excess agreement that MCD64A1
    being seeded by MOD14 would produce. At 0 the data satisfy conditional
    independence, so G^2 should NOT reject; above 0 it should.
    """
    rng = np.random.default_rng(seed)
    pi = np.array([0.018, 0.045, 0.085])
    sens = np.array([[0.48, 0.28, 0.58],     # open: MCD64 sees scars well
                     [0.30, 0.26, 0.54],
                     [0.09, 0.22, 0.47]])    # closed canopy: MCD64 nearly blind
    fpr = np.array([0.0030, 0.0012, 0.0020])
    i, j = dep_pair
    tilt = np.exp(dependence * ((BITS[:, i] > 0) & (BITS[:, j] > 0)))
    counts = []
    for g in range(len(pi)):
        burned = _pattern_likelihood(sens[g]) * tilt
        burned /= burned.sum()
        probs = pi[g] * burned + (1 - pi[g]) * _pattern_likelihood(fpr)
        counts.append(rng.multinomial(n_per_stratum, probs / probs.sum()))
    return np.array(counts), pi, sens, fpr


def write_configs(base_path, prefix, payload, pos_weight):
    """Emit the two training configs, with the fitted block already inlined.

    Configs are GENERATED rather than hand-written so there is never a
    launchable file holding placeholder sensitivities: the measurement model has
    to have been fitted for the config to exist at all.

    Arm A (hard) keeps today's union label and only re-weights it -- the
    label-dependent-cost estimator. Arm B (soft) trains against the posterior.
    Both keep `output_features` untouched, so the evaluation label stays the
    frozen 3-product union and every PR-AUC remains comparable.
    """
    with open(base_path) as fh:
        base = json.load(fh)
    if base.get("pos_weight") != pos_weight:
        raise SystemExit(
            f"--pos_weight {pos_weight} does not match {base_path}'s "
            f"{base.get('pos_weight')}; the weight scales are computed for one "
            "specific pos_weight and would be wrong for the other")
    written = []
    for arm, key in (("a", "sample_weight_hard"), ("b", "sample_weight_soft")):
        cfg = json.loads(json.dumps(base))
        cfg["sample_weight"] = payload[key]
        name = f"{prefix}_{arm}"
        cfg["model_output_path"] = f"gs://aic-amazon/models/{name}.pt"
        out = os.path.join("configs", f"{name}.json")
        with open(out, "w") as fh:
            json.dump(cfg, fh, indent=2)
        written.append(out)
    return written


def _load(path):
    if path.startswith("gs://"):
        import tensorflow as tf
        with tf.io.gfile.GFile(path, "rb") as f:
            return pd.read_parquet(io.BytesIO(f.read()))
    return pd.read_parquet(path)


def tables_from_parquet(path, mode, by_year=False):
    df = _load(path)
    df = df[df["mode"] == mode]
    if df.empty:
        raise SystemExit(f"no rows with mode={mode!r} in {path}")
    keys = ["stratum", "year"] if by_year else ["stratum"]
    grouped = df.groupby(keys + ["pattern"])["count"].sum().reset_index()
    labels = sorted({tuple(r) for r in grouped[keys].to_numpy().tolist()})
    table = np.zeros((len(labels), N_PATTERN))
    index = {lab: i for i, lab in enumerate(labels)}
    for _, row in grouped.iterrows():
        table[index[tuple(row[k] for k in keys)], int(row["pattern"])] = row["count"]
    return table, [dict(zip(keys, lab)) for lab in labels]


def report(counts, pi, sens, fpr, g2, df, q, labels):
    print(f"\n{'stratum':>24} {'pixels':>13} {'prevalence':>11} "
          + " ".join(f"{'s_' + s:>8}" for s in SHORT))
    for i, lab in enumerate(labels):
        print(f"{str(lab):>24} {int(counts[i].sum()):>13,} {pi[i]:>11.4f} "
              + " ".join(f"{sens[i, r]:>8.3f}" for r in range(N_PROD)))
    print(f"\nshared false-positive rates: "
          + ", ".join(f"{s}={fpr[r]:.5f}" for r, s in enumerate(SHORT)))

    print("\nevidence per product (log-likelihood ratio, stratum 0):")
    print(f"{'product':>10} {'detection':>11} {'non-detection':>15}")
    for r, s in enumerate(SHORT):
        print(f"{s:>10} {math.log(sens[0, r] / fpr[r]):>11.3f} "
              f"{math.log((1 - sens[0, r]) / (1 - fpr[r])):>15.3f}")

    if df > 0:
        try:
            from scipy import stats
            p = stats.chi2.sf(g2, df)
            verdict = ("conditional independence NOT rejected"
                       if p > 0.01 else "conditional independence REJECTED")
            print(f"\nfit: G2={g2:.1f} on {df} df, p={p:.3g} -- {verdict}")
        except ImportError:
            print(f"\nfit: G2={g2:.1f} on {df} df")
    else:
        print(f"\nfit: {df} df -- exactly identified, cannot be tested "
              "(add strata to get a test)")

    print("\nposterior q by detection pattern (stratum 0):")
    print(f"{'mcd64':>6} {'mod14':>6} {'viirs':>6} {'q':>8}")
    for k in range(N_PATTERN):
        print(f"{int(BITS[k, 0]):>6} {int(BITS[k, 1]):>6} {int(BITS[k, 2]):>6} "
              f"{q[0, k]:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patterns", help="patterns.parquet from label_agreement_stats.py")
    parser.add_argument("--deltas", default=None)
    parser.add_argument("--mode", default="tolerant", choices=["exact", "tolerant"])
    parser.add_argument("--by_year", action="store_true",
                        help="stratify by (land cover, year) as well")
    parser.add_argument("--pos_weight", type=float, default=10.0)
    parser.add_argument("--dilate", type=int, nargs=3, default=[0, 1, 1],
                        help="radii the tolerant table was built with; copied "
                             "into the emitted config so training matches the fit")
    parser.add_argument("--forest_edges", type=float, nargs="*", default=[0.25, 0.75])
    parser.add_argument("--strat_band", default="im_forest_-1")
    parser.add_argument("--doy_edges", type=float, nargs="*",
                        default=[2.0, 8.0, 32.0, 128.0])
    parser.add_argument("--n_boot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--base_config", default=None,
                        help="training config to derive the two arms from, "
                             "e.g. configs/mlp_v3_mod14.json")
    parser.add_argument("--config_prefix", default="mlp_v3_confw",
                        help="writes configs/<prefix>_a.json (hard arm) and "
                             "configs/<prefix>_b.json (soft arm)")
    parser.add_argument("--synthetic", action="store_true",
                        help="recover known parameters from simulated tables; "
                             "no cloud access, no input files")
    parser.add_argument("--synthetic_dependence", type=float, default=0.0,
                        help="with --synthetic, tilt the mcd64/mod14 joint to "
                             "check the G^2 gate can actually reject")
    args = parser.parse_args()

    truth = None
    if args.synthetic:
        counts, *truth = synthetic_tables(
            seed=args.seed, dependence=args.synthetic_dependence)
        labels = [{"stratum": i} for i in range(counts.shape[0])]
        deltas = None
    else:
        if not args.patterns:
            parser.error("--patterns is required unless --synthetic")
        counts, labels = tables_from_parquet(args.patterns, args.mode, args.by_year)
        deltas = _load(args.deltas) if args.deltas else None

    pi, sens, fpr, loglik = em_fit(counts, seed=args.seed)
    g2, df, expected = fit_statistic(counts, pi, sens, fpr)
    pairs = pair_corrections(counts, expected)
    q = posterior_table(pi, sens, fpr, pairs)
    report(counts, pi, sens, fpr, g2, df, q, labels)

    print("\npairwise residuals (positive 'both' = independence over-counts "
          "their agreement):")
    for p in pairs:
        print(f"  {p['pair']:>13}  both={p['both']:+.3f}  "
              f"observed={p['obs_both']:,}  expected={p['exp_both']:,.0f}")

    if truth:
        t_pi, t_sens, t_fpr = truth
        print("\nrecovery against the simulated truth:")
        print(f"  max |pi  - pi_hat|   = {np.abs(t_pi - pi).max():.5f}")
        print(f"  max |s   - s_hat|    = {np.abs(t_sens - sens).max():.5f}")
        print(f"  max |f   - f_hat|    = {np.abs(t_fpr - fpr).max():.6f}")

    if args.n_boot:
        ci = bootstrap(counts, args.n_boot, args.seed)
        print(f"\nbootstrap: {ci['n_ok']}/{args.n_boot} refits converged")
    else:
        ci = None

    scales = weight_scales(counts, q, args.pos_weight)
    doy = doy_terms(deltas, args.doy_edges)
    edges = {"dilate": args.dilate}
    strat = args.strat_band if counts.shape[0] > 1 and not args.by_year else None
    payload = {
        "fit": {"loglik": loglik, "g2": g2, "df": df,
                "n_pixels": int(counts.sum()), "mode": args.mode,
                "strata": labels, "bootstrap": ci},
        "weight_scales": scales,
        "pair_residuals": pairs,
        "sample_weight_hard": build_block(pi, sens, fpr, pairs, doy, edges,
                                          scales, False, strat, args.forest_edges),
        "sample_weight_soft": build_block(pi, sens, fpr, pairs, doy, edges,
                                          scales, True, strat, args.forest_edges),
    }
    if args.base_config:
        if args.synthetic:
            raise SystemExit("refusing to write launchable configs from "
                             "simulated tables")
        for path in write_configs(args.base_config, args.config_prefix,
                                  payload, args.pos_weight):
            print(f"wrote {path}")
        print(f"\nnext: cv_make_folds.py --arch {args.config_prefix}_a "
              f"--base_config configs/{args.config_prefix}_a.json  (and _b)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.out}")
    elif not args.base_config:
        print("\n--- sample_weight (soft arm) ---")
        print(json.dumps(payload["sample_weight_soft"], indent=2))


if __name__ == "__main__":
    main()
