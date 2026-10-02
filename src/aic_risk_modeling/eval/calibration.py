"""Calibration: reliability bins, ECE, and Platt/isotonic fitting."""

import numpy as np

_PROB_EPS = 1e-6


def _bin_edges(scores, n_bins, strategy):
    """Bin edges on [0, 1]: 'uniform' width or 'quantile' (equal count, ties merged)."""
    if strategy == 'uniform':
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == 'quantile':
        edges = np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
        return edges
    raise ValueError(f"unknown binning strategy {strategy!r} (use 'uniform' or 'quantile')")


def reliability_bins(scores, labels, n_bins=15, strategy='uniform'):
    """Per-bin bin_lo/bin_hi, conf, freq (NaN if empty) and count for a reliability diagram."""
    scores = np.clip(np.asarray(scores).reshape(-1).astype(np.float64), 0.0, 1.0)
    labels = np.asarray(labels).reshape(-1).astype(np.float64)

    edges = _bin_edges(scores, n_bins, strategy)
    nb = edges.size - 1
    bin_idx = np.clip(np.searchsorted(edges, scores, side='right') - 1, 0, nb - 1)

    count = np.bincount(bin_idx, minlength=nb).astype(np.float64)
    score_sum = np.bincount(bin_idx, weights=scores, minlength=nb)
    pos_sum = np.bincount(bin_idx, weights=labels, minlength=nb)

    nonempty = count > 0
    conf = np.full(nb, np.nan)
    freq = np.full(nb, np.nan)
    conf[nonempty] = score_sum[nonempty] / count[nonempty]
    freq[nonempty] = pos_sum[nonempty] / count[nonempty]

    return {
        'bin_lo': edges[:-1],
        'bin_hi': edges[1:],
        'conf': conf,
        'freq': freq,
        'count': count,
    }


def expected_calibration_error(scores, labels, n_bins=15, strategy='uniform'):
    """(ece, mce, bins): count-weighted mean and max |conf - freq| over non-empty bins."""
    bins = reliability_bins(scores, labels, n_bins=n_bins, strategy=strategy)
    count = bins['count']
    total = count.sum()
    nonempty = count > 0
    if total == 0 or not nonempty.any():
        return 0.0, 0.0, bins
    gap = np.abs(bins['conf'][nonempty] - bins['freq'][nonempty])
    weights = count[nonempty] / total
    ece = float(np.sum(weights * gap))
    mce = float(np.max(gap))
    return ece, mce, bins


def _probs_to_logits(scores, eps=_PROB_EPS):
    """log(p / (1 - p)) with p clipped eps from 0/1 so saturated scores stay finite."""
    p = np.clip(np.asarray(scores).reshape(-1).astype(np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid_stable(s):
    return np.where(s >= 0, 1.0 / (1.0 + np.exp(-s)), np.exp(s) / (1.0 + np.exp(s)))


def fit_platt(scores, labels, eps=_PROB_EPS):
    """Fit sigmoid(a*logit(p) + b) by unregularized logistic regression; returns (a, b)."""
    from sklearn.linear_model import LogisticRegression

    z = _probs_to_logits(scores, eps=eps).reshape(-1, 1)
    y = np.asarray(labels).reshape(-1).astype(int)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(scores, a, b, eps=_PROB_EPS):
    s = a * _probs_to_logits(scores, eps=eps) + b
    return _sigmoid_stable(s).reshape(np.shape(scores))


def fit_isotonic(scores, labels):
    """Fit a monotonic IsotonicRegression from scores to probabilities."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(scores).reshape(-1).astype(np.float64),
            np.asarray(labels).reshape(-1).astype(np.float64))
    return iso


def apply_isotonic(scores, iso):
    out = iso.predict(np.asarray(scores).reshape(-1).astype(np.float64))
    return out.reshape(np.shape(scores))


def fit_calibrator(method, scores, labels):
    """Fit 'platt' or 'isotonic'; returns (transform_fn, info_str)."""
    method = method.lower()
    if method == 'platt':
        a, b = fit_platt(scores, labels)
        return (lambda s: apply_platt(s, a, b)), f"platt (a={a:.4f}, b={b:.4f})"
    if method == 'isotonic':
        iso = fit_isotonic(scores, labels)
        return (lambda s: apply_isotonic(s, iso)), "isotonic (non-parametric)"
    raise ValueError(
        f"unknown calibration method {method!r} "
        "(use 'platt' or 'isotonic')")


def deflate(q, pos_weight):
    """Invert the weighted-BCE optimum q = w*p/(w*p+1-p) back to p."""
    return q / (pos_weight - (pos_weight - 1.0) * q)


def inflate(p, pos_weight):
    """Inverse of deflate."""
    return pos_weight * p / (pos_weight * p + 1.0 - p)


def to_prob(q, pos_weight, level=1.0, cal=None):
    """Model score -> burn probability: frozen calibrator if given, else deflate / level."""
    return cal(q) if cal is not None else deflate(q, pos_weight) / level


def load_calibrator(path):
    """Frozen calibrator from a calibrated_year_totals.py npz (platt, or isotonic if no `method`)."""
    d = np.load(path)
    method = str(d["method"]) if "method" in d.files else "isotonic"
    if method == "platt":
        a, b = float(d["a"]), float(d["b"])
        return lambda q: apply_platt(q, a, b)
    x, y = d["x"], d["y"]
    return lambda q: np.interp(q, x, y)


def reliability_table_str(bins):
    lines = [f"{'bin':>11}  {'conf':>6}  {'freq':>6}  {'count':>12}"]
    for lo, hi, conf, freq, count in zip(
            bins['bin_lo'], bins['bin_hi'], bins['conf'], bins['freq'],
            bins['count']):
        conf_s = f"{conf:.3f}" if np.isfinite(conf) else "-"
        freq_s = f"{freq:.3f}" if np.isfinite(freq) else "-"
        lines.append(
            f"{lo:>4.2f}-{hi:<4.2f}  {conf_s:>6}  {freq_s:>6}  {int(count):>12}")
    return "\n".join(lines)


def plot_reliability_diagram(bins, ece, mce, out_path, title=None):
    """Save a reliability diagram PNG; returns False if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed; skipping reliability plot ({out_path}).")
        return False

    conf = bins['conf']
    freq = bins['freq']
    count = bins['count']
    centers = 0.5 * (bins['bin_lo'] + bins['bin_hi'])
    widths = bins['bin_hi'] - bins['bin_lo']
    nonempty = count > 0

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(5, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]})

    ax_cal.plot([0, 1], [0, 1], '--', color='gray', label='perfect')
    ax_cal.plot(conf[nonempty], freq[nonempty], 'o-', color='C0', label='model')
    ax_cal.set_ylabel('empirical frequency')
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.legend(loc='upper left')
    ax_cal.set_title(title or f"Reliability (ECE={ece:.4f}, MCE={mce:.4f})")

    ax_hist.bar(centers, count, width=widths * 0.9, color='C0')
    ax_hist.set_ylabel('count')
    ax_hist.set_xlabel('predicted probability')

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True
