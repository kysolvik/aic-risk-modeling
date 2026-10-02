"""PyTorch loss functions.

Losses are called as `loss(y_true, y_pred)` where `y_pred` holds sigmoid
probabilities, not logits.
"""

import torch

_EPSILON = 1e-7


def _bce_elementwise(y_true, y_pred):
    y_pred = y_pred.clamp(_EPSILON, 1.0 - _EPSILON)
    return -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))


def weighted_bce(pos_weight):
    """Weighted BCE loss for 2D segmentation problems.

    Positive pixels are weighted by `pos_weight` and negatives by 1.0.
    """
    def loss(y_true, y_pred):
        y_true = y_true.float()
        weights = y_true * pos_weight + (1.0 - y_true)
        return (_bce_elementwise(y_true, y_pred) * weights).mean()
    return loss


def deflate_probs(y_pred, pos_weight):
    """Invert the probability inflation caused by a weighted BCE.

    Training with `pos_weight` w drives predictions toward the pointwise
    optimum q = w*p / (w*p + 1 - p), an inflated version of the calibrated
    probability p. This maps q back to p = q / (w - (w-1)*q) exactly
    (identity when pos_weight is 1), so sums of deflated probabilities are
    comparable to actual positive-pixel counts.
    """
    return y_pred / (pos_weight - (pos_weight - 1.0) * y_pred)


# Losses whose predictions are inflated by pos_weight (see deflate_probs);
# consumers (e.g. the area_ratio metric) should deflate before summing.
POS_WEIGHT_LOSSES = frozenset({'weighted_binary_crossentropy'})


def get_loss(loss_name, pos_weight=9.0):
    """Loss function by config name; only 'weighted_binary_crossentropy' is supported."""
    if loss_name != 'weighted_binary_crossentropy':
        raise ValueError(f"Could not interpret loss name: {loss_name}. "
                         "Current options: ['weighted_binary_crossentropy']")
    return weighted_bce(pos_weight)
