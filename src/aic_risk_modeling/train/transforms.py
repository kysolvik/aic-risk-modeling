"""Per-feature transforms applied inside the tf.data pipeline."""

import tensorflow as tf

def gt0(tensor):
    return tf.cast(tensor > 0, tf.float32)

def gt0_bool(tensor):
    return tf.cast(tensor > 0, tf.bool)


def lag1(tensor):
    """Shift a monthly vector one step later along the last axis, edge-padding slot 0.

    For md_oni: PSL labels each 3-month mean by its centre month, so the raw Dec(Y-1)
    value includes Jan(Y). lag1 makes every slot a trailing mean (no lookahead).
    """
    return tf.concat([tensor[..., :1], tensor[..., :-1]], axis=-1)


transform_registry = {
    "gt0": gt0,
    "gt0_bool": gt0_bool,
    "lag1": lag1,
}

# Value-preserving transforms: these features are still z-scored, other transformed ones aren't.
NORMALIZE_THROUGH_TRANSFORMS = {"lag1"}
