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


def deflate_probs(y_pred, pos_weight):
    """Invert the probability inflation caused by a weighted BCE.

    Training with `pos_weight` w drives predictions toward the pointwise
    optimum q = w*p / (w*p + 1 - p), an inflated version of the calibrated
    probability p. This maps q back to p = q / (w - (w-1)*q) exactly
    (identity when pos_weight is 1), so sums of deflated probabilities are
    comparable to actual positive-pixel counts.
    """
    return y_pred / (pos_weight - (pos_weight - 1.0) * y_pred)


def area_log_ratio(pos_weight=1.0, block_size=None):
    """Aggregate burn-area term: squared log-ratio of expected vs actual count.

    Per batch element, compares the expected positive-pixel count (sum of
    deflated predicted probabilities, see `deflate_probs`) against the actual
    count, as (log1p(expected) - log1p(actual))**2 — scale-free and finite for
    empty chips. With `block_size` K the counts are compared per K x K block
    (K must divide the chip size) instead of per whole chip.

    Per-pixel sample weights don't apply to an aggregate count; the argument
    is accepted only to keep a uniform loss call signature.
    """
    def loss(y_true, y_pred, sample_weight=None):
        y_true = y_true.float()
        p_hat = deflate_probs(y_pred.clamp(_EPSILON, 1.0 - _EPSILON),
                              pos_weight)
        if block_size:
            scale = float(block_size * block_size)
            expected = torch.nn.functional.avg_pool2d(
                p_hat.unsqueeze(1), block_size) * scale
            actual = torch.nn.functional.avg_pool2d(
                y_true.unsqueeze(1), block_size) * scale
        else:
            expected = p_hat.flatten(1).sum(1)
            actual = y_true.flatten(1).sum(1)
        return (torch.log1p(expected) - torch.log1p(actual)).pow(2).mean()
    return loss


def weighted_bce_area(pos_weight, area_weight=1.0, area_block_size=None):
    """Weighted BCE plus an aggregate expected-burn-area term.

    The area term ties the summed prediction to the summed label so the
    aggregate burn level (e.g. year-to-year severity) carries explicit loss
    pressure instead of only the diffuse per-pixel signal. It always deflates
    predictions by `pos_weight` first (`deflate_probs`), anchoring the
    calibrated count so it doesn't fight the WBCE optimum. The trainer calls
    losses outside autocast on float32 predictions, which the log-ratio term
    relies on.

    sample_weight applies to the BCE term only; note that a per-type
    sample_weight map makes the effective per-pixel pos_weight vary, so the
    scalar deflation is then only approximate.
    """
    wbce = weighted_bce(pos_weight)
    area = area_log_ratio(pos_weight, area_block_size)

    def loss(y_true, y_pred, sample_weight=None):
        return (wbce(y_true, y_pred, sample_weight)
                + area_weight * area(y_true, y_pred))
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
    'weighted_bce_area': weighted_bce_area(9.0),
}

# Losses whose predictions are inflated by pos_weight (see deflate_probs);
# consumers (e.g. the area_ratio metric) should deflate before summing.
POS_WEIGHT_LOSSES = frozenset({
    'weighted_binary_crossentropy',
    'weighted_bce_dice',
    'weighted_bce_area',
})


def get_loss(loss_name, num_classes=1, class_weights=None, pos_weight=9.0,
             area_weight=1.0, area_block_size=None):
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
            ('weighted_binary_crossentropy', 'weighted_bce_dice',
            'weighted_bce_area'). Defaults to 9.0 to reproduce the historical
            hard-coded value.
        area_weight: weight of the aggregate burn-area term in
            'weighted_bce_area'.
        area_block_size: optional K to compare expected-vs-actual counts per
            K x K block in 'weighted_bce_area' instead of per whole chip.

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
    binary_losses['weighted_bce_area'] = weighted_bce_area(
        pos_weight, area_weight, area_block_size)

    obj = binary_losses.get(loss_name, None)
    if callable(obj):
        return obj
    else:
        raise ValueError(f"Could not interpret loss name: {loss_name}. "
                         f"Current options: {list(binary_losses.keys())}")
