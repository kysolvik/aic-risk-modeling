
import tensorflow as tf

def gt0(tensor):
    return tf.cast(tensor > 0, tf.float32)

def gt0_bool(tensor):
    return tf.cast(tensor > 0, tf.bool)

def eq2(tensor):
    return tf.cast(tensor == 2, tf.float32)

def normalize_mcwd(tensor):
    """Normalize MCWD values using precomputed mean and stddev."""
    mean = tf.constant(600.0, dtype=tf.float32)
    std = tf.constant(400.0, dtype=tf.float32)

    return (tf.cast(tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_evi(tensor):
    """Normalize evi values using precomputed mean and stddev."""
    mean = tf.constant(6250.0, dtype=tf.float32)
    std = tf.constant(1200.0, dtype=tf.float32)

    return (tf.cast(tensor, tf.float32) - mean) / (std + 1e-7)

def normalize_ndvi(tensor):
    """Normalize ndvi values using precomputed mean and stddev."""
    mean = tf.constant(8600.0, dtype=tf.float32)
    std = tf.constant(1100.0, dtype=tf.float32)

    return (tf.cast(tensor, tf.float32) - mean) / (std + 1e-7)

transform_registry = {
    "gt0": gt0,
    "gt0_bool": gt0_bool,
    "eq2": eq2,
    "normalize_mcwd": normalize_mcwd,
    "normalize_evi": normalize_evi,
    "normalize_ndvi": normalize_ndvi,
    "none": lambda x: x
}
