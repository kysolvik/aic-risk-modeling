"""Streaming metrics for binary segmentation.

ROC AUC and PR AUC are computed from histograms of the predicted
probabilities over a fixed threshold grid (the same approximation
keras.metrics.AUC uses), so memory stays constant regardless of how many
batches are accumulated.
"""

import torch


class SegmentationMetrics:
    """Accumulates binary IoU, ROC AUC, and PR AUC over batches.

    `update` takes ground truth (any shape, bool or 0/1) and predicted
    probabilities of the same shape. Histograms live on the same device as
    the predictions, so no per-batch device transfers happen.
    """

    def __init__(self, num_thresholds=512, iou_threshold=0.5):
        self.num_thresholds = num_thresholds
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.pos_hist = None
        self.neg_hist = None

    @torch.no_grad()
    def update(self, y_true, y_pred):
        y_pred = y_pred.detach().float().flatten().clamp(0.0, 1.0)
        y_true = y_true.detach().bool().flatten()
        pos = torch.histc(y_pred[y_true], bins=self.num_thresholds,
                          min=0.0, max=1.0)
        neg = torch.histc(y_pred[~y_true], bins=self.num_thresholds,
                          min=0.0, max=1.0)
        if self.pos_hist is None:
            self.pos_hist = pos
            self.neg_hist = neg
        else:
            self.pos_hist += pos
            self.neg_hist += neg

    @torch.no_grad()
    def compute(self):
        """Returns {'binary_iou', 'roc_auc', 'pr_auc'} as python floats."""
        if self.pos_hist is None:
            return {'binary_iou': 0.0, 'roc_auc': 0.0, 'pr_auc': 0.0}

        # tp[k]/fp[k]: counts predicted positive at threshold k/num_thresholds,
        # i.e. reverse cumulative sums of the histograms (k = 0..num_thresholds).
        zero = self.pos_hist.new_zeros(1)
        tp = torch.cat([self.pos_hist.flip(0).cumsum(0).flip(0), zero])
        fp = torch.cat([self.neg_hist.flip(0).cumsum(0).flip(0), zero])
        total_pos = tp[0]
        total_neg = fp[0]
        if total_pos == 0 or total_neg == 0:
            return {'binary_iou': 0.0, 'roc_auc': 0.0, 'pr_auc': 0.0}

        tpr = tp / total_pos
        fpr = fp / total_neg
        precision = torch.where(tp + fp > 0, tp / (tp + fp),
                                torch.ones_like(tp))

        # flip so x runs ascending for trapezoidal integration
        roc_auc = torch.trapz(tpr.flip(0), fpr.flip(0))
        pr_auc = torch.trapz(precision.flip(0), tpr.flip(0))

        # IoU on class 1 at iou_threshold: tp / (tp + fp + fn), fn = P - tp
        k = min(int(round(self.iou_threshold * self.num_thresholds)),
                self.num_thresholds)
        binary_iou = tp[k] / (total_pos + fp[k])

        return {
            'binary_iou': float(binary_iou),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
        }


class MulticlassSegmentationMetrics:
    """Accumulates a confusion matrix for multi-class segmentation.

    `update` takes integer ground-truth class labels (any shape) and predicted
    class probabilities with a trailing class axis of size num_classes (the
    decoder's softmax output). The confusion matrix lives on the prediction
    device, so no per-batch device transfers happen.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = None

    @torch.no_grad()
    def update(self, y_true, y_pred):
        n = self.num_classes
        y_true = y_true.detach().long().flatten()
        y_pred = y_pred.detach().argmax(dim=-1).long().flatten()
        # Row = true class, column = predicted class.
        index = y_true * n + y_pred
        cm = torch.bincount(index, minlength=n * n).reshape(n, n).float()
        if self.confusion is None:
            self.confusion = cm
        else:
            self.confusion += cm

    @torch.no_grad()
    def compute(self):
        """Returns overall accuracy, per-class IoU ('iou_class{c}'), mean IoU
        over all classes, and 'fire_iou' (mean IoU over the non-background
        classes 1..C-1) as python floats."""
        n = self.num_classes
        if self.confusion is None:
            result = {'accuracy': 0.0, 'mean_iou': 0.0, 'fire_iou': 0.0}
            result.update({f'iou_class{c}': 0.0 for c in range(n)})
            return result

        cm = self.confusion
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
        total = cm.sum()
        accuracy = tp.sum() / total if total > 0 else tp.new_zeros(())

        result = {
            'accuracy': float(accuracy),
            'mean_iou': float(iou.mean()),
            'fire_iou': float(iou[1:].mean()) if n > 1 else float(iou.mean()),
        }
        for c in range(n):
            result[f'iou_class{c}'] = float(iou[c])
        return result
