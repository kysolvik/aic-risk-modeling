"""Per-feature transforms applied inside the tf.data pipeline."""

import tensorflow as tf

def gt0(tensor):
    return tf.cast(tensor > 0, tf.float32)

def gt0_bool(tensor):
    return tf.cast(tensor > 0, tf.bool)

def gt2_bool(tensor):
    return tf.cast(tensor > 2, tf.bool)

def gt1_bool(tensor):
    return tf.cast(tensor > 1, tf.bool)

def eq2(tensor):
    return tf.cast(tensor == 2, tf.float32)

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

def normalize_mcwd(tensor):
    """Normalize MCWD values using precomputed mean and stddev."""
    mean = tf.constant(650.0, dtype=tf.float32)
    std = tf.constant(425.0, dtype=tf.float32)

    return (tf.cast(tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_def(tensor):
    """Normalize MCWD values using precomputed mean and stddev."""
    mean = tf.constant(200.0, dtype=tf.float32)
    std = tf.constant(50.0, dtype=tf.float32)

    out_tensor = tf.where(tensor < -10000.0,
                          mean,
                          tensor)
    return (tf.cast(out_tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_pdsi(tensor):
    """Normalize PDSI values using precomputed mean and stddev."""
    mean = tf.constant(-100.0, dtype=tf.float32)
    std = tf.constant(50.0, dtype=tf.float32)

    out_tensor = tf.where(tensor < -10000.0,
                          mean,
                          tensor)

    return (tf.cast(out_tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_tmmn(tensor):
    """Normalize TMMN values using precomputed mean and stddev."""
    mean = tf.constant(165.0, dtype=tf.float32)
    std = tf.constant(50.0, dtype=tf.float32)

    out_tensor = tf.where(tensor < -10000.0,
                          mean,
                          tensor)
    return (tf.cast(out_tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_tmmx(tensor):
    """Normalize TMMX values using precomputed mean and stddev."""
    mean = tf.constant(276.0, dtype=tf.float32)
    std = tf.constant(50.0, dtype=tf.float32)

    out_tensor = tf.where(tensor < -10000.0,
                          mean,
                          tensor)
    return (tf.cast(out_tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_evi(tensor):
    """Normalize evi values using precomputed mean and stddev."""
    mean = tf.constant(4000.0, dtype=tf.float32)
    std = tf.constant(1200.0, dtype=tf.float32)

    out_tensor = tf.where(tensor < 0.0,
                          mean,
                          tensor)
    return (tf.cast(out_tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_vpd(tensor):
    """Normalize vpd values using precomputed mean and stddev."""
    mean = tf.constant(60.0, dtype=tf.float32)
    std = tf.constant(50.0, dtype=tf.float32)

    out_tensor = tf.where(tensor < -10000.0,
                          mean,
                          tensor)
    return (tf.cast(out_tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_ndvi(tensor):
    """Normalize ndvi values using precomputed mean and stddev."""
    mean = tf.constant(6750.0, dtype=tf.float32)
    std = tf.constant(1500.0, dtype=tf.float32)

    out_tensor = tf.where(tensor < 0.0,
                          mean,
                          tensor)
    return (tf.cast(out_tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_classification(tensor):
    """Normalize classification values using precomputed mean and stddev."""
    mean = tf.constant(0.85, dtype=tf.float32)
    std = tf.constant(0.3, dtype=tf.float32)

    return (tf.cast(tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_lat(tensor):
    """Normalize latitude values using precomputed mean and stddev."""
    mean = tf.constant(-5.0, dtype=tf.float32)
    std = tf.constant(10.0, dtype=tf.float32)

    return (tf.cast(tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_lon(tensor):
    """Normalize longitude values using precomputed mean and stddev."""
    mean = tf.constant(-50.0, dtype=tf.float32)
    std = tf.constant(10.0, dtype=tf.float32)

    return (tf.cast(tensor, tf.float32) - mean) / (std + 1e-7)

transform_registry = {
    "gt0": gt0,
    "gt0_bool": gt0_bool,
    "gt2_bool": gt2_bool,
    "gt1_bool": gt1_bool,
    "eq2": eq2,
    "normalize_mcwd": normalize_mcwd,
    "normalize_def": normalize_def,
    "normalize_pdsi": normalize_pdsi,
    "normalize_tmmn": normalize_tmmn,
    "normalize_tmmx": normalize_tmmx,
    "normalize_vpd": normalize_vpd,
    "normalize_evi": normalize_evi,
    "normalize_ndvi": normalize_ndvi,
    "normalize_classification": normalize_classification,
    "normalize_lat": normalize_lat,
    "normalize_lon": normalize_lon,
    "none": lambda x: x,
    "lag1": lag1,
}

# Transforms that only re-index values. Unlike the binarizing / hand-normalizing
# transforms above, a feature using one of these is still z-scored from the stats
# (data_norm.get_normalize_list skips every other transformed feature).
NORMALIZE_THROUGH_TRANSFORMS = {"lag1"}
