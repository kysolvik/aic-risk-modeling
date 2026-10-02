"""Binary metrics for fire-probability predictions vs ground truth."""

import numpy as np
from sklearn import metrics

from .calibration import (
    expected_calibration_error,
    fit_calibrator,
    plot_reliability_diagram,
    reliability_table_str,
)


def _print_metrics(title, stats):
    """Pretty-print a one-vs-rest metrics dict (as returned by binary_metrics)."""
    print(title)
    print(f"  Accuracy: {stats['accuracy']:.4f}")
    print(f"  Precision: {stats['precision']:.4f}")
    print(f"  Recall: {stats['recall']:.4f}")
    print(f"  F1 Score: {stats['f1']:.4f}")
    print(f"  Cohen's Kappa: {stats['kappa']:.4f}")
    print(f"  PR AUC: {stats['pr_auc']:.4f}")
    if stats.get('ece') is not None:
        print(f"  ECE: {stats['ece']:.4f}")
        print(f"  MCE: {stats['mce']:.4f}")
    print(f"  N (truth): {stats['n_truth']}")
    print(f"  N (pred): {stats['n_pred']}")


def binary_metrics(gt_binary, pred_binary, scores=None):
    """One-vs-rest metrics for a single class/label.

    ``pred_binary`` is the hard prediction (a thresholded score for binary
    models, or ``argmax == class`` for multiclass). ``scores`` are the
    continuous scores used for PR AUC; if ``None``, PR AUC is skipped.
    """
    gt_flat = np.asarray(gt_binary).flatten().astype(bool)
    pred_flat = np.asarray(pred_binary).flatten().astype(bool)
    n_truth = int(gt_flat.sum())
    n_pred = int(pred_flat.sum())
    stats = {k: 0.0 for k in ("accuracy", "precision", "recall", "f1", "kappa", "pr_auc")}
    stats["n_truth"] = n_truth
    stats["n_pred"] = n_pred
    if n_truth == 0:
        print("Warning: No positive cases in ground truth, metrics undefined.")
        return stats
    if n_pred == 0:
        print("Warning: No positive cases in predictions; precision/recall/F1 will be 0.")
    # Hard-label metrics from the thresholded/argmax predictions.
    stats["accuracy"] = metrics.accuracy_score(gt_flat, pred_flat)
    stats["precision"] = metrics.precision_score(gt_flat, pred_flat, zero_division=0)
    stats["recall"] = metrics.recall_score(gt_flat, pred_flat, zero_division=0)
    stats["f1"] = metrics.f1_score(gt_flat, pred_flat, zero_division=0)
    stats["kappa"] = metrics.cohen_kappa_score(gt_flat, pred_flat)
    # PR AUC uses the continuous class scores, not the hard labels.
    if scores is not None:
        stats["pr_auc"] = metrics.average_precision_score(gt_flat, np.asarray(scores).flatten())
    return stats


def calc_stats(predictions, ground_truth, grouped=False, threshold=0.5,
               reliability_plot=None, calibration_bins=15,
               calibration_binning='uniform', calibration_method='none',
               calibration_fit=None):
    """Calculate stats for predictions vs ground truth.

    Computes binary stats by thresholding the scores, and reports calibration:
    Expected/Maximum Calibration Error plus a reliability table, and writes a
    reliability-diagram PNG to ``reliability_plot`` when given.

    ``calibration_binning`` selects ``'uniform'`` (equal-width) or ``'quantile'``
    (equal-count) ECE bins; quantile binning keeps the sparse high-probability
    region from being swamped by the near-zero mass under heavy class imbalance.

    Post-hoc calibration: ``calibration_method`` is one of
    ``'none'``/``'platt'``/``'isotonic'``. The calibrator is fit on
    ``calibration_fit`` (a ``(scores, ground_truth)`` held-out pair, an
    out-of-sample fit) if given, else in-sample on this set. When a calibrator
    is applied, the pre-calibration ECE/MCE are also printed.

    Returns ``(stats, calibrated)``: ``stats`` is the metrics dict, and
    ``calibrated`` is the calibrated prediction array (same shape as
    ``predictions``) when a calibrator was applied, else ``None``.
    """
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    labels = ground_truth > 0

    # Post-hoc calibration. Fitting and reporting on the same set (in-sample);
    # pass calibration_fit (a held-out split) for an out-of-sample estimate.
    transform = None
    cal_info = None
    if calibration_method != 'none':
        if calibration_fit is not None:
            fit_scores, fit_gt = calibration_fit
            fit_scores, fit_labels = np.asarray(fit_scores), np.asarray(fit_gt) > 0
            fit_src = "held-out calibration set"
        else:
            fit_scores, fit_labels = predictions, labels
            fit_src = "this set (in-sample)"
        transform, cal_info = fit_calibrator(calibration_method, fit_scores, fit_labels)
        cal_info += f", fit on {fit_src}"

    if transform is not None:
        ece_raw, mce_raw, _ = expected_calibration_error(
            predictions, labels, n_bins=calibration_bins, strategy=calibration_binning)
        predictions = transform(predictions)
        print(f"Applied post-hoc calibration: {cal_info}")
        print(f"  pre-calibration:  ECE={ece_raw:.4f}  MCE={mce_raw:.4f}")

    if grouped:
        unique_labels = np.unique(ground_truth)
        for label in unique_labels:
            if label != 0:  # Skip background
                label_mask = ground_truth == label
                stats = binary_metrics(label_mask, predictions > threshold, scores=predictions)
                _print_metrics(f"Stats for group {label}:", stats)

    overall = binary_metrics(labels, predictions > threshold, scores=predictions)

    # Calibration uses the continuous scores (not the thresholded labels).
    ece, mce, bins = expected_calibration_error(
        predictions, labels, n_bins=calibration_bins, strategy=calibration_binning)
    overall["ece"] = ece
    overall["mce"] = mce
    if cal_info is not None:
        overall["calibration"] = cal_info

    _print_metrics("Overall Stats:", overall)
    print(f"Reliability ({calibration_binning} bins, predicted probability vs "
          f"empirical frequency):")
    print(reliability_table_str(bins))
    if reliability_plot is not None:
        plot_reliability_diagram(bins, ece, mce, reliability_plot)

    return overall, (predictions if transform is not None else None)


def pr_auc(labels, scores):
    return float(binary_metrics(labels, scores >= 0.5, scores=scores)["pr_auc"])


def decompose(scores, labels, chip):
    """Split pooled PR-AUC into chip-level intensity vs within-chip localisation.

    `chip` holds each pixel's chip index. oracle_chip_only = true chip rate flat
    inside each chip; within_chip_only = scores / own chip mean (pure
    localisation); oracle_chip_rescale = chip means set to truth; chip_r =
    corr(model chip mean, true chip rate).
    """
    n_chips = int(chip.max()) + 1
    count = np.bincount(chip, minlength=n_chips).astype(np.float64)
    true_rate = np.bincount(chip, weights=labels, minlength=n_chips) / count
    pred_rate = np.maximum(np.bincount(chip, weights=scores, minlength=n_chips) / count, 1e-9)

    rescaled = np.clip(scores * (true_rate / pred_rate)[chip], 0.0, 1.0)

    # The complement of oracle_chip_rescale: PERFECT within-chip ranking at the
    # model's OWN chip intensity. This is the ceiling for a better-localising
    # architecture, and it is what oracle_chip_rescale does not tell you -- that
    # column holds localisation fixed and perfects the chip scalar; this one holds
    # the chip scalar fixed and perfects localisation.
    #
    # Built as a WITHIN-CHIP PERMUTATION of the model's own scores: inside each
    # chip the score multiset is untouched and simply re-assigned so the largest
    # values land on the positives. Chip mean, chip variance and the model's
    # within-chip dynamic range are therefore all preserved exactly, and only the
    # ordering is perfected.
    #
    # Do NOT instead nudge positives above negatives by a tiny epsilon. That
    # collapses within-chip dynamic range to ~0, so the global ranking degenerates
    # to chip order alone and pooled AP FALLS (measured: MLP 0.3566 -> 0.2332).
    # The result is an artifact of the construction, not a statement about
    # localisation.
    # `labels` arrives as a bool array; unary minus is undefined on numpy bools, so
    # cast before negating to get descending (positives-first) order.
    by_score = np.lexsort((-scores, chip))                    # per chip, score desc
    by_label = np.lexsort((-labels.astype(np.int8), chip))    # per chip, positives first
    oracle_within = np.empty_like(scores)
    oracle_within[by_label] = scores[by_score]

    return {
        "pr_auc": pr_auc(labels, scores),
        "oracle_chip_only": pr_auc(labels, true_rate[chip]),
        "within_chip_only": pr_auc(labels, scores / pred_rate[chip]),
        "oracle_chip_rescale": pr_auc(labels, rescaled),
        "oracle_within_chip": pr_auc(labels, oracle_within),
        "chip_r": float(np.corrcoef(pred_rate, true_rate)[0, 1]),
        "prevalence": float(labels.mean()),
        "n_chips": n_chips,
        "n_pixels": int(labels.size),
    }


def fit_stats(x, y):
    """OLS slope/intercept/R^2 plus Spearman rho."""
    if x.std() == 0 or y.std() == 0:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, rho=np.nan)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    rho = float(np.corrcoef(xr, yr)[0, 1])
    return dict(slope=float(slope), intercept=float(intercept), r2=float(r2), rho=rho)


def best_f1(labels, scores):
    """Best achievable F1 over all thresholds, plus the P/R/threshold there.

    Models here are trained with weighted BCE (pos_weight 10), so their
    probabilities are inflated relative to a natural-scale baseline like
    climatology. Scoring every series at one fixed threshold would compare
    calibration, not ranking -- climatology loses on recall simply because its
    values rarely exceed 0.5. Sweeping the threshold removes that confound and
    asks what each score field could achieve at its own best operating point.
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(labels, scores)
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1), 0.0)
    i = int(np.argmax(f1))
    # precision_recall_curve returns len(thr) == len(prec) - 1
    t = float(thr[i]) if i < len(thr) else float("inf")
    return float(f1[i]), float(prec[i]), float(rec[i]), t


def best_kappa(labels, scores):
    """Best achievable Cohen's kappa over all thresholds, plus that threshold.

    Kappa is threshold-dependent (unlike PR-AUC / ROC-AUC), so -- exactly as for
    `best_f1` -- the threshold is swept rather than fixed at 0.5, otherwise the
    weighted-BCE models (inflated probabilities) and the natural-scale baselines
    would be compared on calibration, not agreement. Vectorised over the sorted
    scores: for every distinct cut, predict-positive = {score >= cut} and read
    kappa = (p_o - p_e) / (1 - p_e) off the running confusion counts.
    """
    y = (np.asarray(labels) > 0).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n = y.size
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0, float("inf")  # kappa undefined with one class; no skill = 0
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    tp = np.cumsum(y[order]).astype(np.float64)   # positives among the top k
    k = np.arange(1, n + 1, dtype=np.float64)      # predicted-positive count
    tn = n_neg - (k - tp)
    p_o = (tp + tn) / n
    p_e = (k / n) * (n_pos / n) + ((n - k) / n) * (n_neg / n)
    denom = 1.0 - p_e
    kappa = np.where(denom > 0, (p_o - p_e) / np.where(denom > 0, denom, 1.0), 0.0)
    # Only cuts at a distinct-score boundary are real operating points (ties must
    # move together); predicting the all-negative side always scores kappa 0.
    boundary = np.ones(n, dtype=bool)
    boundary[:-1] = s_sorted[1:] != s_sorted[:-1]
    kappa_at_cut = np.where(boundary, kappa, -np.inf)
    i = int(np.argmax(kappa_at_cut))
    best = float(kappa[i])
    if best <= 0.0:
        return 0.0, float("inf")
    return best, float(s_sorted[i])
