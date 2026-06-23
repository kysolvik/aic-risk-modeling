"""Helpers for evaluating fire model outputs.

Predictions can be binary (single band of scores) or multiclass. For
multiclass, the prediction raster follows the layout written by
``scripts/predict/predict.py``: band 1 is the argmax (hard predicted class),
and the remaining bands are the per-class scores (softmax probabilities), so
band ``2 + c`` holds the score for class ``c``. Ground truth is a single band
of integer class labels.
"""

import numpy as np
from sklearn import metrics

from .calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
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


def _is_multiclass(predictions):
    """Multiclass predictions are band-first (>1 band): argmax + class scores."""
    return np.ndim(predictions) == 3 and np.shape(predictions)[0] > 1


def calc_stats_multiclass(predictions, ground_truth, class_names=None):
    """Per-class accuracy stats for multiclass predictions.

    Hard-label metrics (accuracy/precision/recall/F1/kappa) come from the
    argmax band; PR AUC uses the raw per-class scores. Also reports overall
    multiclass accuracy and kappa across all pixels.

    Args:
        predictions: array of shape ``(num_classes + 1, H, W)`` where band 0 is
            the argmax class and bands ``1 + c`` are the score for class ``c``.
        ground_truth: array of integer class labels.
        class_names: optional sequence mapping class index -> display name.

    Returns:
        dict with ``"overall"`` and ``"per_class"`` (keyed by class index) stats.
    """
    predictions = np.asarray(predictions)
    pred_labels = predictions[0].astype(int)  # argmax band
    scores = predictions[1:]                  # (num_classes, H, W)
    num_classes = scores.shape[0]

    gt_flat = np.asarray(ground_truth).flatten().astype(int)
    pred_flat = pred_labels.flatten()

    overall = {
        "accuracy": metrics.accuracy_score(gt_flat, pred_flat),
        "kappa": metrics.cohen_kappa_score(gt_flat, pred_flat),
        "macro_f1": metrics.f1_score(gt_flat, pred_flat, average="macro", zero_division=0),
    }
    print("Overall multiclass stats:")
    print(f"  Accuracy: {overall['accuracy']:.4f}")
    print(f"  Cohen's Kappa: {overall['kappa']:.4f}")
    print(f"  Macro F1: {overall['macro_f1']:.4f}")

    per_class = {}
    for c in range(num_classes):
        name = class_names[c] if class_names is not None else c
        stats = _binary_metrics(gt_flat == c, pred_flat == c, scores=scores[c])
        per_class[c] = stats
        _print_metrics(f"Stats for class {name}:", stats)

    return {"overall": overall, "per_class": per_class}


def calc_stats(predictions, ground_truth, grouped=False, threshold=0.5,
               class_names=None, reliability_plot=None, calibration_bins=15,
               temperature=None, fit_temp=False):
    """Calculate stats for predictions vs ground truth.

    Dispatches to per-class multiclass stats when ``predictions`` is a
    multi-band (band-first) array; otherwise computes binary stats by
    thresholding the single-band scores. For binary predictions, also reports
    calibration: Expected/Maximum Calibration Error plus a reliability table,
    and writes a reliability-diagram PNG to ``reliability_plot`` when given.

    Post-hoc temperature scaling (binary only): pass ``fit_temp=True`` to fit the
    NLL-minimizing temperature on this set, or ``temperature=T`` to apply a known
    T (e.g. one fitted on a held-out calibration split). Scaling is monotonic and
    fixes the p=0.5 crossing, so hard-label metrics and PR AUC are unchanged; only
    the calibration metrics/diagram move. Returns the fitted/applied T in the
    stats dict under ``"temperature"``.
    """
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    if _is_multiclass(predictions):
        if fit_temp or temperature is not None:
            print("Warning: temperature scaling is binary-only; ignoring for multiclass.")
        return calc_stats_multiclass(predictions, ground_truth, class_names=class_names)

    labels = ground_truth > 0

    # Post-hoc temperature scaling. Fitting and applying on the same set is
    # optimistic; fit on a held-out split (printed T) and apply via temperature=T.
    if fit_temp:
        temperature = fit_temperature(predictions, labels)
        print(f"Fitted temperature: T = {temperature:.4f} "
              f"(NLL-minimizing, in-sample on this set)")
    if temperature is not None and temperature != 1.0:
        ece_raw, mce_raw, _ = expected_calibration_error(
            predictions, labels, n_bins=calibration_bins)
        predictions = apply_temperature(predictions, temperature)
        print(f"Applied temperature scaling T = {temperature:.4f}  "
              f"(pre-scaling calibration: ECE={ece_raw:.4f} MCE={mce_raw:.4f})")

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
        predictions, labels, n_bins=calibration_bins)
    overall["ece"] = ece
    overall["mce"] = mce
    if temperature is not None:
        overall["temperature"] = float(temperature)

    _print_metrics("Overall Stats:", overall)
    print("Reliability (predicted probability vs empirical frequency):")
    print(reliability_table_str(bins))
    if reliability_plot is not None:
        plot_reliability_diagram(bins, ece, mce, reliability_plot)

    return overall


def load_preprocess_inputs(predictions_path, ground_truth_path):
    """Load and preprocess inputs for evaluation.

    For tif predictions, all bands are read. A single-band raster is squeezed to
    2D (binary scores); a multi-band raster is returned band-first
    ``(num_bands, H, W)`` (argmax band + class scores) and triggers the
    multiclass code path in ``calc_stats``.
    """
    if predictions_path.endswith('.csv'):
        if not ground_truth_path.endswith('.csv'):
            raise ValueError("Both predictions and ground truth must be in the same format (both csv or both tif)")
        import pandas as pd
        predictions = pd.read_csv(predictions_path)['pred'].values
        ground_truth = pd.read_csv(ground_truth_path)['label'].values
    else:
        if not ground_truth_path.endswith('.tif') and not predictions_path.endswith('.tif'):
            raise ValueError("Both predictions and ground truth must be in the same format (both csv or both tif")
        import rasterio as rio
        predictions = rio.open(predictions_path).read()  # (bands, H, W)
        if predictions.shape[0] == 1:
            predictions = predictions[0]  # binary: squeeze to (H, W)
        ground_truth = rio.open(ground_truth_path).read(1)
    return predictions, ground_truth
