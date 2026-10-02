"""Per-feature transforms applied inside the tf.data pipeline."""

import tensorflow as tf

def gt0(tensor):
    return tf.cast(tensor > 0, tf.float32)

def gt0_bool(tensor):
    return tf.cast(tensor > 0, tf.bool)


def lag1(tensor):
    """Shift a monthly vector one step later along its last axis.

    out[..., k] = in[..., k-1]; the oldest slot repeats its own value (edge pad),
    so the length is unchanged and no artificial value enters the sequence.

    For `md_oni`: PSL's oni.csv labels each 3-month mean by its CENTRE month, so
    the export's Dec(Y-1) value is Nov(Y-1)-Jan(Y) -- one month past the
    January-forecast lag. After lag1 every slot is the trailing 3-month mean
    ending that month (last slot = Oct-Dec(Y-1)), aligned with SOI/TNA/AMO/MEI.
    Value-preserving, so the feature is still normalized (see
    NORMALIZE_THROUGH_TRANSFORMS).
    """
    return tf.concat([tensor[..., :1], tensor[..., :-1]], axis=-1)


transform_registry = {
    "gt0": gt0,
    "gt0_bool": gt0_bool,
    "lag1": lag1,
}

# Transforms that only re-index values. Unlike the binarizing transforms above,
# a feature using one of these is still z-scored from the stats
# (data_norm.get_normalize_list skips every other transformed feature).
NORMALIZE_THROUGH_TRANSFORMS = {"lag1"}
