"""Losses, called as loss(y_true, y_pred) with y_pred as probabilities (not logits)."""

import torch

_EPSILON = 1e-7


def _bce_elementwise(y_true, y_pred):
    y_pred = y_pred.clamp(_EPSILON, 1.0 - _EPSILON)
    return -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))


def weighted_bce(pos_weight):
    """BCE with positives weighted by pos_weight."""
    def loss(y_true, y_pred):
        y_true = y_true.float()
        weights = y_true * pos_weight + (1.0 - y_true)
        return (_bce_elementwise(y_true, y_pred) * weights).mean()
    return loss


def deflate_probs(y_pred, pos_weight):
    """Map a weighted-BCE optimum q = wp/(wp+1-p) back to the calibrated p."""
    return y_pred / (pos_weight - (pos_weight - 1.0) * y_pred)


# Predictions from these losses must be deflated before summing.
POS_WEIGHT_LOSSES = frozenset({'weighted_binary_crossentropy'})


def get_loss(loss_name, pos_weight=9.0):
    """Loss function by config name; only 'weighted_binary_crossentropy' is supported."""
    if loss_name != 'weighted_binary_crossentropy':
        raise ValueError(f"Could not interpret loss name: {loss_name}. "
                         "Current options: ['weighted_binary_crossentropy']")
    return weighted_bce(pos_weight)
