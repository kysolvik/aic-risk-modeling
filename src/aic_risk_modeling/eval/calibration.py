"""Calibration metrics for binary fire-risk predictions.

Quantifies model overconfidence. With hard-label BCE the model is pushed to emit
probabilities near 0 and 1, but fire is stochastic, so a well-calibrated model's
predicted probability should match the empirical burn frequency. We bin
predictions by predicted probability (the same binning ``SegmentationMetrics``
uses for PR/ROC AUC in ``train/metrics.py``) and compare each bin's mean
predicted confidence to the observed positive rate:

  - Expected Calibration Error (ECE): count-weighted mean |confidence - frequency|.
  - Maximum Calibration Error (MCE): worst-case bin |confidence - frequency|.
  - Reliability diagram: confidence (x) vs frequency (y) against the y=x diagonal.

It also provides post-hoc *temperature scaling* (Guo et al. 2017): a single
scalar T > 1 softens an overconfident model by dividing the logits before the
sigmoid. Because the saved predictions are already sigmoid probabilities (the
model applies the sigmoid in ``forward``), we recover the logits as
``log(p / (1 - p))``, divide by T, and re-apply the sigmoid. Temperature scaling
is monotonic and fixes the p=0.5 crossing, so it leaves hard-label metrics and
PR AUC untouched and only moves the calibration curve.

Plotting uses matplotlib, imported lazily so it stays an optional dependency.
"""

import numpy as np

# Clip probabilities this far from {0, 1} before taking the logit, so saturated
# predictions give finite logits (|logit| <= ~13.8 at 1e-6) instead of +/-inf.
_PROB_EPS = 1e-6


def _bin_edges(scores, n_bins, strategy):
    """Bin edges over [0, 1] for ECE binning.

    ``uniform`` returns ``n_bins`` equal-width bins. ``quantile`` returns bins
    with roughly equal *count* by placing edges at score quantiles, so the
    sparse high-probability region gets real weight instead of being swallowed
    by one giant near-zero bin (the usual failure of equal-width ECE under heavy
    class imbalance). Tied quantiles are de-duplicated, which can yield fewer
    than ``n_bins`` bins; the [0, 1] endpoints are always included.
    """
    if strategy == 'uniform':
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == 'quantile':
        edges = np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)  # drop degenerate (tied) edges
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
        return edges
    raise ValueError(f"unknown binning strategy {strategy!r} (use 'uniform' or 'quantile')")


def reliability_bins(scores, labels, n_bins=15, strategy='uniform'):
    """Bin predictions by predicted probability for a reliability diagram.

    Args:
        scores: predicted probabilities in [0, 1], any shape (flattened here).
        labels: ground-truth labels (0/1 or bool), same shape as ``scores``.
        n_bins: number of bins.
        strategy: ``'uniform'`` (equal-width) or ``'quantile'`` (equal-count).

    Returns:
        dict of per-bin numpy arrays: ``bin_lo``/``bin_hi`` (bin edges), ``conf``
        (mean predicted probability in the bin, NaN if empty), ``freq``
        (empirical positive rate in the bin, NaN if empty), and ``count`` (number
        of pixels in the bin). Length is ``n_bins`` for uniform, possibly fewer
        for quantile when score ties collapse edges.
    """
    scores = np.clip(np.asarray(scores).reshape(-1).astype(np.float64), 0.0, 1.0)
    labels = np.asarray(labels).reshape(-1).astype(np.float64)

    edges = _bin_edges(scores, n_bins, strategy)
    nb = edges.size - 1
    # Bin index in [0, nb-1]; the right edge (score == 1.0) lands in the last bin.
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
    """Expected and maximum calibration error for binary predictions.

    Returns:
        (ece, mce, bins) where ``ece`` is the count-weighted mean of
        |confidence - frequency| over non-empty bins, ``mce`` is the maximum
        over non-empty bins, and ``bins`` is the ``reliability_bins`` dict.
    """
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
    """Recover logits from saved sigmoid probabilities: ``log(p / (1 - p))``.

    Probabilities are clipped ``eps`` away from 0 and 1 first, so predictions
    that saturated to exactly 0 or 1 in the float32 raster yield large but
    finite logits rather than infinities. Note this caps the recoverable
    confidence: a model that truly saturated has lost the information temperature
    scaling would need, and gets pinned at ``|logit| <= log((1 - eps) / eps)``.
    """
    p = np.clip(np.asarray(scores).reshape(-1).astype(np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def apply_temperature(scores, temperature, eps=_PROB_EPS):
    """Temperature-scale saved sigmoid probabilities.

    Recovers logits from ``scores``, divides by ``temperature`` (T > 1 softens an
    overconfident model toward 0.5; T < 1 sharpens), and re-applies the sigmoid.
    Returns probabilities with the same shape as ``scores``.
    """
    if temperature is None or temperature <= 0:
        raise ValueError(f"temperature must be a positive float, got {temperature!r}")
    logits = _probs_to_logits(scores, eps=eps) / float(temperature)
    # Stable sigmoid (avoids overflow in exp for large-magnitude logits).
    out = np.where(logits >= 0,
                   1.0 / (1.0 + np.exp(-logits)),
                   np.exp(logits) / (1.0 + np.exp(logits)))
    return out.reshape(np.shape(scores))


def fit_temperature(scores, labels, eps=_PROB_EPS, bounds=(1e-2, 1e2)):
    """Fit the NLL-minimizing temperature for saved sigmoid probabilities.

    Recovers logits from ``scores`` and finds the scalar T that minimizes the
    binary cross-entropy of ``sigmoid(logit / T)`` against ``labels`` (the
    standard temperature-scaling objective of Guo et al. 2017). The NLL is
    computed directly from logits via softplus for numerical stability.

    Fit T on a held-out calibration split and apply it to the test set; fitting
    and reporting on the same set (in-sample). Returns the fitted temperature
    as a float (1.0 if there are no usable samples).
    """
    from scipy.optimize import minimize_scalar

    logits = _probs_to_logits(scores, eps=eps)
    y = np.asarray(labels).reshape(-1).astype(np.float64)
    if logits.size == 0:
        return 1.0

    def nll(temperature):
        s = logits / temperature
        # -log-likelihood per sample = softplus(s) - y * s; mean over pixels.
        return float(np.mean(np.logaddexp(0.0, s) - y * s))

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(result.x)


def _sigmoid_stable(s):
    """Numerically stable elementwise sigmoid."""
    return np.where(s >= 0, 1.0 / (1.0 + np.exp(-s)), np.exp(s) / (1.0 + np.exp(s)))


def fit_platt(scores, labels, eps=_PROB_EPS):
    """Fit Platt (logistic) scaling on saved sigmoid probabilities.

    Recovers logits ``z`` from ``scores`` and fits ``sigmoid(a*z + b)`` by
    logistic regression of ``labels`` on ``z`` (effectively unregularized). The
    intercept ``b`` is what temperature scaling lacks: it shifts the whole
    reliability curve, correcting a systematic bias/base-rate offset that a
    single temperature (which pins the p=0.5 crossing) cannot. Temperature
    scaling is the ``a = 1/T, b = 0`` special case.

    Returns ``(a, b)``. Fit on a held-out calibration split.
    """
    from sklearn.linear_model import LogisticRegression

    z = _probs_to_logits(scores, eps=eps).reshape(-1, 1)
    y = np.asarray(labels).reshape(-1).astype(int)
    # C large => negligible regularization, i.e. plain logistic calibration.
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(scores, a, b, eps=_PROB_EPS):
    """Apply Platt scaling: ``sigmoid(a * logit(scores) + b)``."""
    s = a * _probs_to_logits(scores, eps=eps) + b
    return _sigmoid_stable(s).reshape(np.shape(scores))


def fit_isotonic(scores, labels):
    """Fit isotonic regression mapping scores -> calibrated probabilities.

    Non-parametric and monotonic: it can bend the reliability curve onto the
    diagonal at *any* probability level (unlike the single global squash of
    temperature scaling), so it handles arbitrary monotonic miscalibration. With
    pixel-scale data there are plenty of samples to fit it; the main risk is
    overfitting the sparse high-probability tail, so fit it on a held-out split.

    Returns a fitted ``sklearn.isotonic.IsotonicRegression``.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(scores).reshape(-1).astype(np.float64),
            np.asarray(labels).reshape(-1).astype(np.float64))
    return iso


def apply_isotonic(scores, iso):
    """Apply a fitted isotonic calibrator, preserving ``scores`` shape."""
    out = iso.predict(np.asarray(scores).reshape(-1).astype(np.float64))
    return out.reshape(np.shape(scores))


def fit_calibrator(method, scores, labels):
    """Fit a post-hoc calibrator and return ``(transform_fn, info_str)``.

    ``transform_fn`` maps a probability array to calibrated probabilities;
    ``info_str`` summarizes the fitted parameters for logging. ``method`` is one
    of ``'temperature'``, ``'platt'``, ``'isotonic'``.

    Temperature scaling is monotonic *and* fixes the p=0.5 crossing, so it leaves
    threshold-0.5 hard metrics unchanged; Platt and isotonic are monotonic (so PR
    AUC is preserved) but can move the 0.5 operating point, so hard-label metrics
    may shift.
    """
    method = method.lower()
    if method == 'temperature':
        t = fit_temperature(scores, labels)
        return (lambda s: apply_temperature(s, t)), f"temperature (T={t:.4f})"
    if method == 'platt':
        a, b = fit_platt(scores, labels)
        return (lambda s: apply_platt(s, a, b)), f"platt (a={a:.4f}, b={b:.4f})"
    if method == 'isotonic':
        iso = fit_isotonic(scores, labels)
        return (lambda s: apply_isotonic(s, iso)), "isotonic (non-parametric)"
    raise ValueError(
        f"unknown calibration method {method!r} "
        "(use 'temperature', 'platt', or 'isotonic')")


def reliability_table_str(bins):
    """Format a reliability table (bin range, confidence, frequency, count)."""
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
    """Save a reliability diagram PNG (calibration curve + count histogram).

    matplotlib is imported lazily; if it is not installed, prints a notice and
    returns False instead of raising, so calibration stays usable without a
    plotting dependency.

    Returns:
        True if the figure was written, False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend, no display required
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
