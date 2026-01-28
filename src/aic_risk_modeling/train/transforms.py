
import tensorflow as tf

def gt0(tensor):
    return tensor > 0

def eq2(tensor):
    return tf.cast(tensor == 2, tf.float32)

def normalize_mcwd(tensor):
    """Normalize MCWD values using precomputed mean and stddev."""
    mean = tf.constant(889.8259811401367, dtype=tf.float32)
    std = tf.constant(507.33071914299626, dtype=tf.float32)

    return (tf.cast(tensor, tf.float32) - mean) / (std + 1e-7)

transform_registry = {
    "gt0": gt0,
    "eq2": eq2,
    "normalize_mcwd": normalize_mcwd,
    "none": lambda x: x
}