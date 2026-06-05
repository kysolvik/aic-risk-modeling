"""Helpers for evaluating fire model outputs"""

import numpy as np
from sklearn import metrics

def _get_metrics(preds, gt, threshold=0.5):
    """Calculate metrics for predictions vs ground truth"""
    # Flatten arrays
    preds_flat = preds.flatten()
    gt_flat = gt.flatten()
    # Apply threshold
    preds_binary = (preds_flat > threshold).astype(bool)
    n_truth = np.sum(gt_flat)
    n_pred = np.sum(preds_binary)
    print(n_truth, n_pred)
    if n_truth == 0 or n_pred == 0:
        print("Warning: No positive cases in ground truth or predictions, metrics undefined.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, n_truth, n_pred
    else:
        # Calculate metrics
        pr_auc = metrics.average_precision_score(gt_flat, preds_flat)
        accuracy = metrics.accuracy_score(gt_flat, preds_binary)
        precision = metrics.precision_score(gt_flat, preds_binary, zero_division=0)
        recall = metrics.recall_score(gt_flat, preds_binary, zero_division=0)
        f1 = metrics.f1_score(gt_flat, preds_binary, zero_division=0)
        kappa = metrics.cohen_kappa_score(gt_flat, preds_binary)
        return accuracy, precision, recall, f1, kappa, pr_auc, n_truth, n_pred


def calc_stats(predictions, ground_truth, grouped=False, threshold=0.5):
    """Calculate stats for predictions vs ground truth"""
    if grouped:
        unique_labels = np.unique(ground_truth)
        for label in unique_labels:
            if label != 0:  # Skip background
                label_mask = ground_truth == label
                # Calculate stats for this group
                accuracy, precision, recall, f1, kappa, pr_auc, n_truth, n_pred = _get_metrics(predictions, label_mask, threshold=threshold)
                print(f"Stats for group {label}:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  Cohen's Kappa: {kappa:.4f}")
                print(f"  PR AUC: {pr_auc:.4f}")
                print(f"  N (truth): {n_truth}")
                print(f"  N (pred): {n_pred}")

    accuracy, precision, recall, f1, kappa, pr_auc, n_truth, n_pred = _get_metrics(predictions, ground_truth>0, threshold=threshold)
    print("Overall Stats:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"  N (truth): {n_truth}")
    print(f"  N (pred): {n_pred}")
    return accuracy, precision, recall, f1, kappa, pr_auc, n_truth, n_pred

def load_preprocess_inputs(predictions_path, ground_truth_path):
    """Load and preprocess inputs for evaluation"""
    # Load predictions and ground truth (this is a placeholder, replace with actual loading code)
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
        predictions = rio.open(predictions_path).read(1)  # Replace with actual loading code
        ground_truth = rio.open(ground_truth_path).read(1)  # Replace with actual loading code
    return predictions, ground_truth
