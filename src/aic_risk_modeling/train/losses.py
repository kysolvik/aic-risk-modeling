import inspect
import keras


def weighted_bce(pos_weight):
    """Weighted BCE loss for 2D segmentation problems."""
    def loss(y_true, y_pred):
        y_true = keras.ops.cast(y_true, dtype='float32')
        bce = keras.losses.binary_crossentropy(y_true,
                                               y_pred, axis=0)
        weights = (y_true * pos_weight + (1 - y_true))
        return keras.ops.mean(bce * weights)
    return loss

def weighted_bce_dice(pos_weight):
    """Weighted BCE and Dice loss combined (mean)"""
    def loss(y_true, y_pred):
        y_true = keras.ops.cast(y_true, dtype='float32')
        bce = keras.losses.binary_crossentropy(y_true, y_pred, axis=0)
        weights = y_true * pos_weight + (1 - y_true)
        bce = keras.ops.mean(bce*weights)
        dice = keras.losses.dice(y_true, y_pred)
        return bce + dice
    return loss

LOSSES_DICT = {
    'binary_crossentropy': keras.losses.BinaryCrossentropy(),
    'weighted_binary_crossentropy': weighted_bce(9.0),
    'dice': keras.losses.Dice(),
    'focal': keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True),
    'weighted_bce_dice': weighted_bce_dice(9.0),
}

def get_loss(loss_name):
    """Retrieves loss as a `function`/`Loss` class instance.

    The `loss_name` is the lowercase name of loss.

    >>> loss = losses.get("weighted_binary_crossentropy")
    >>> type(loss)
    <class 'function'>

    Args:
        loss_name: A loss name. One of None or string name of a loss
            function/class or loss configuration dictionary or a loss function
            or a loss class instance.

    Returns:
        A Keras loss as a `function`/ `Loss` class instance.
    """
    if loss_name is None:
        return None
    elif isinstance(loss_name, str):
        obj = LOSSES_DICT.get(loss_name, None)
    else:
        obj = loss_name

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(f"Could not interpret loss name: {loss_name}"
                         f"Current options: {LOSSES_DICT.keys()}")