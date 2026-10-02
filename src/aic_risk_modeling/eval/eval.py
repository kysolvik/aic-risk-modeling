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
    """Pretty-print a one-vs-rest metrics dict (as returned by _binary_metrics)."""
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


def _binary_metrics(gt_binary, pred_binary, scores=None):
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
                stats = _binary_metrics(label_mask, predictions > threshold, scores=predictions)
                _print_metrics(f"Stats for group {label}:", stats)

    overall = _binary_metrics(labels, predictions > threshold, scores=predictions)

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


def _block_max_pool(arr, block):
    """Non-overlapping ``block`` x ``block`` max-pool of a 2-D array.

    When H/W are not multiples of ``block`` the array is zero-padded on the
    bottom/right first. Zero padding is safe for both the fire-probability and
    the 0/1 label fields because 0 is the minimum possible value, so a partial
    edge block's max is effectively taken over its real pixels only.
    """
    block = int(block)
    if block <= 1:
        return np.asarray(arr)
    arr = np.asarray(arr)
    height, width = arr.shape
    pad_h, pad_w = (-height) % block, (-width) % block
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=0)
    padded_h, padded_w = arr.shape
    return arr.reshape(padded_h // block, block,
                       padded_w // block, block).max(axis=(1, 3))


