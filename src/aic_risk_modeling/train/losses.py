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


def _true_class_prob(y_true, y_pred):
    """Probability the model assigned to each pixel's true class.

    y_true holds integer class indices (any shape); y_pred holds softmax
    probabilities with a trailing class axis of matching shape.
    """
    y_pred = y_pred.clamp(_EPSILON, 1.0 - _EPSILON)
    y_true = y_true.long().unsqueeze(-1)
    return torch.gather(y_pred, -1, y_true).squeeze(-1)


def categorical_crossentropy(y_true, y_pred):
    """Multi-class cross-entropy over softmax probabilities."""
    return -torch.log(_true_class_prob(y_true, y_pred)).mean()


def weighted_categorical_crossentropy(class_weights):
    """Multi-class cross-entropy weighting each pixel by its true-class weight."""
    weights = torch.as_tensor(class_weights, dtype=torch.float32)

    def loss(y_true, y_pred):
        true_p = _true_class_prob(y_true, y_pred)
        pixel_weights = weights.to(y_pred.device)[y_true.long()]
        return (-torch.log(true_p) * pixel_weights).mean()
    return loss


LOSSES_DICT = {
    'binary_crossentropy': binary_crossentropy,
    'weighted_binary_crossentropy': weighted_bce(9.0),
    'dice': dice,
    'focal': focal(),
    'weighted_bce_dice': weighted_bce_dice(9.0),
}


def get_loss(loss_name, num_classes=1, class_weights=None):
    """Retrieves a loss function by name.

    Args:
        loss_name: None, the string name of a loss, or a callable (returned
            as-is).
        num_classes: number of output classes. >1 selects the multi-class
            losses, whose predictions are softmax probabilities with a trailing
            class axis and whose labels are integer class indices.
        class_weights: optional per-class weights (length num_classes) for
            'weighted_categorical_crossentropy'; defaults to all ones.

    Returns:
        A loss function called as loss(y_true, y_pred).
    """
    if loss_name is None:
        return None
    if callable(loss_name):
        return loss_name

    if num_classes > 1:
        if class_weights is None:
            class_weights = [1.0] * num_classes
        elif len(class_weights) != num_classes:
            raise ValueError(
                f"class_weights has {len(class_weights)} entries but "
                f"num_classes is {num_classes}")
        multiclass_losses = {
            'categorical_crossentropy': categorical_crossentropy,
            'weighted_categorical_crossentropy':
                weighted_categorical_crossentropy(class_weights),
        }
        obj = multiclass_losses.get(loss_name, None)
        if not callable(obj):
            raise ValueError(
                f"Could not interpret multi-class loss name: {loss_name}. "
                f"Current options: {list(multiclass_losses.keys())}")
        return obj

    obj = LOSSES_DICT.get(loss_name, None)
    if callable(obj):
        return obj
    else:
        raise ValueError(f"Could not interpret loss name: {loss_name}. "
                         f"Current options: {list(LOSSES_DICT.keys())}")
