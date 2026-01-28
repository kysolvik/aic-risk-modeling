
import tensorflow as tf

def gt0(tensor):
    return tf.cast(tensor > 0, tf.float32)

def eq2(tensor):
    return tf.cast(tensor == 2, tf.float32)

transform_registry = {
    "gt0": gt0,
    "eq2": eq2,
    "none": lambda x: x
}