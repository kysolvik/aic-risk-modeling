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


def reliability_bins(scores, labels, n_bins=15):
    """Bin predictions by predicted probability for a reliability diagram.

    Args:
        scores: predicted probabilities in [0, 1], any shape (flattened here).
        labels: ground-truth labels (0/1 or bool), same shape as ``scores``.
        n_bins: number of equal-width bins over [0, 1].

    Returns:
        dict of per-bin numpy arrays (length ``n_bins``): ``bin_lo``/``bin_hi``
        (bin edges), ``conf`` (mean predicted probability in the bin, NaN if
        empty), ``freq`` (empirical positive rate in the bin, NaN if empty), and
        ``count`` (number of pixels in the bin).
    """
    scores = np.clip(np.asarray(scores).reshape(-1).astype(np.float64), 0.0, 1.0)
    labels = np.asarray(labels).reshape(-1).astype(np.float64)

    # Bin index in [0, n_bins-1]; the right edge (score == 1.0) lands in the last bin.
    bin_idx = np.minimum((scores * n_bins).astype(int), n_bins - 1)

    count = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    score_sum = np.bincount(bin_idx, weights=scores, minlength=n_bins)
    pos_sum = np.bincount(bin_idx, weights=labels, minlength=n_bins)

    nonempty = count > 0
    conf = np.full(n_bins, np.nan)
    freq = np.full(n_bins, np.nan)
    conf[nonempty] = score_sum[nonempty] / count[nonempty]
    freq[nonempty] = pos_sum[nonempty] / count[nonempty]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return {
        'bin_lo': edges[:-1],
        'bin_hi': edges[1:],
        'conf': conf,
        'freq': freq,
        'count': count,
    }


def expected_calibration_error(scores, labels, n_bins=15):
    """Expected and maximum calibration error for binary predictions.

    Returns:
        (ece, mce, bins) where ``ece`` is the count-weighted mean of
        |confidence - frequency| over non-empty bins, ``mce`` is the maximum
        over non-empty bins, and ``bins`` is the ``reliability_bins`` dict.
    """
    bins = reliability_bins(scores, labels, n_bins=n_bins)
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
    and reporting on the same set is optimistic. Returns the fitted temperature
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
