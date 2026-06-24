"""PyTorch loss functions.

All losses are called as `loss(y_true, y_pred, sample_weight=None)` where
`y_pred` holds probabilities (after sigmoid for binary, softmax for
multi-class), not logits. `sample_weight`, when given, is a per-pixel weight
tensor (same shape as the loss's elementwise term) multiplied in before the
mean. It is built upstream in the data pipeline (see
`data_loader.build_type_weight_map`) to up-weight specific fire types while
keeping the model binary.
"""

import torch

_EPSILON = 1e-7


def _bce_elementwise(y_true, y_pred):
    y_pred = y_pred.clamp(_EPSILON, 1.0 - _EPSILON)
    return -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))


def _weighted_mean(values, sample_weight):
    """Mean of `values`, optionally weighted per-element by `sample_weight`."""
    if sample_weight is None:
        return values.mean()
    return (values * sample_weight).mean()


def binary_crossentropy(y_true, y_pred, sample_weight=None):
    return _weighted_mean(_bce_elementwise(y_true.float(), y_pred), sample_weight)


def weighted_bce(pos_weight):
    """Weighted BCE loss for 2D segmentation problems.

    Without a `sample_weight`, positive pixels are weighted by `pos_weight` and
    negatives by 1.0. When a `sample_weight` is supplied it is treated as the
    authoritative per-pixel weight (it already encodes `pos_weight` for fire and
    any per-type overrides), so the internal class weighting is skipped.
    """
    def loss(y_true, y_pred, sample_weight=None):
        y_true = y_true.float()
        bce = _bce_elementwise(y_true, y_pred)
        if sample_weight is not None:
            return (bce * sample_weight).mean()
        weights = y_true * pos_weight + (1.0 - y_true)
        return (bce * weights).mean()
    return loss


def dice(y_true, y_pred, sample_weight=None):
    # Dice is a set-overlap measure, so per-pixel sample weights don't apply;
    # the argument is accepted only to keep a uniform loss call signature.
    y_true = y_true.float()
    intersection = (y_true * y_pred).sum()
    return 1.0 - (2.0 * intersection + _EPSILON) / (
        y_true.sum() + y_pred.sum() + _EPSILON)


def focal(alpha=0.25, gamma=2.0):
    """Class-balanced binary focal crossentropy."""
    def loss(y_true, y_pred, sample_weight=None):
        y_true = y_true.float()
        bce = _bce_elementwise(y_true, y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return _weighted_mean(alpha_t * (1.0 - p_t) ** gamma * bce, sample_weight)
    return loss


def weighted_bce_dice(pos_weight):
    """Weighted BCE and Dice loss combined"""
    wbce = weighted_bce(pos_weight)

    def loss(y_true, y_pred, sample_weight=None):
        # sample_weight applies to the BCE term only; Dice is set-based.
        return wbce(y_true, y_pred, sample_weight) + dice(y_true, y_pred)
    return loss


def _true_class_prob(y_true, y_pred):
    """Probability the model assigned to each pixel's true class.

    y_true holds integer class indices (any shape); y_pred holds softmax
    probabilities with a trailing class axis of matching shape.
    """
    y_pred = y_pred.clamp(_EPSILON, 1.0 - _EPSILON)
    y_true = y_true.long().unsqueeze(-1)
    return torch.gather(y_pred, -1, y_true).squeeze(-1)


def categorical_crossentropy(y_true, y_pred, sample_weight=None):
    """Multi-class cross-entropy over softmax probabilities."""
    return _weighted_mean(-torch.log(_true_class_prob(y_true, y_pred)), sample_weight)


def weighted_categorical_crossentropy(class_weights):
    """Multi-class cross-entropy weighting each pixel by its true-class weight."""
    weights = torch.as_tensor(class_weights, dtype=torch.float32)

    def loss(y_true, y_pred, sample_weight=None):
        true_p = _true_class_prob(y_true, y_pred)
        pixel_weights = weights.to(y_pred.device)[y_true.long()]
        return _weighted_mean(-torch.log(true_p) * pixel_weights, sample_weight)
    return loss


LOSSES_DICT = {
    'binary_crossentropy': binary_crossentropy,
    'weighted_binary_crossentropy': weighted_bce(9.0),
    'dice': dice,
    'focal': focal(),
    'weighted_bce_dice': weighted_bce_dice(9.0),
}


def get_loss(loss_name, num_classes=1, class_weights=None, pos_weight=9.0):
    """Retrieves a loss function by name.

    Args:
        loss_name: None, the string name of a loss, or a callable (returned
            as-is).
        num_classes: number of output classes. >1 selects the multi-class
            losses, whose predictions are softmax probabilities with a trailing
            class axis and whose labels are integer class indices.
        class_weights: optional per-class weights (length num_classes) for
            'weighted_categorical_crossentropy'; defaults to all ones.
        pos_weight: positive-class weight for the binary weighted losses
            ('weighted_binary_crossentropy', 'weighted_bce_dice'). Defaults to
            9.0 to reproduce the historical hard-coded value.

    Returns:
        A loss function called as loss(y_true, y_pred, sample_weight=None).
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

    # Binary losses. Rebuild the pos_weight-dependent ones from the configured
    # pos_weight; the rest are reused from LOSSES_DICT unchanged.
    binary_losses = dict(LOSSES_DICT)
    binary_losses['weighted_binary_crossentropy'] = weighted_bce(pos_weight)
    binary_losses['weighted_bce_dice'] = weighted_bce_dice(pos_weight)

    obj = binary_losses.get(loss_name, None)
    if callable(obj):
        return obj
    else:
        raise ValueError(f"Could not interpret loss name: {loss_name}. "
                         f"Current options: {list(binary_losses.keys())}")
