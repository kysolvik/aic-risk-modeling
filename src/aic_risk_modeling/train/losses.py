"""PyTorch loss functions.

All losses are called as `loss(y_true, y_pred)` where `y_pred` holds
probabilities (after sigmoid), not logits.
"""

import torch

_EPSILON = 1e-7


def _bce_elementwise(y_true, y_pred):
    y_pred = y_pred.clamp(_EPSILON, 1.0 - _EPSILON)
    return -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))


def binary_crossentropy(y_true, y_pred):
    return _bce_elementwise(y_true.float(), y_pred).mean()


def weighted_bce(pos_weight):
    """Weighted BCE loss for 2D segmentation problems."""
    def loss(y_true, y_pred):
        y_true = y_true.float()
        bce = _bce_elementwise(y_true, y_pred)
        weights = y_true * pos_weight + (1.0 - y_true)
        return (bce * weights).mean()
    return loss


def dice(y_true, y_pred):
    y_true = y_true.float()
    intersection = (y_true * y_pred).sum()
    return 1.0 - (2.0 * intersection + _EPSILON) / (
        y_true.sum() + y_pred.sum() + _EPSILON)


def focal(alpha=0.25, gamma=2.0):
    """Class-balanced binary focal crossentropy."""
    def loss(y_true, y_pred):
        y_true = y_true.float()
        bce = _bce_elementwise(y_true, y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return (alpha_t * (1.0 - p_t) ** gamma * bce).mean()
    return loss


def weighted_bce_dice(pos_weight):
    """Weighted BCE and Dice loss combined"""
    wbce = weighted_bce(pos_weight)

    def loss(y_true, y_pred):
        return wbce(y_true, y_pred) + dice(y_true, y_pred)
    return loss


LOSSES_DICT = {
    'binary_crossentropy': binary_crossentropy,
    'weighted_binary_crossentropy': weighted_bce(9.0),
    'dice': dice,
    'focal': focal(),
    'weighted_bce_dice': weighted_bce_dice(9.0),
}


def get_loss(loss_name):
    """Retrieves a loss function by name.

    Args:
        loss_name: None, the string name of a loss in LOSSES_DICT, or a
            callable (returned as-is).

    Returns:
        A loss function called as loss(y_true, y_pred).
    """
    if loss_name is None:
        return None
    elif isinstance(loss_name, str):
        obj = LOSSES_DICT.get(loss_name, None)
    else:
        obj = loss_name

    if callable(obj):
        return obj
    else:
        raise ValueError(f"Could not interpret loss name: {loss_name}. "
                         f"Current options: {list(LOSSES_DICT.keys())}")
