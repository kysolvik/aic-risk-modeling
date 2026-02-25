
import keras

def gt0(tensor):
    return keras.ops.cast(tensor > 0, 'float32')

def gt0_bool(tensor):
    return keras.ops.cast(tensor > 0, 'bool')

def eq2(tensor):
    return keras.ops.cast(tensor == 2, 'float32')

def normalize_mcwd(tensor):
    """Normalize MCWD values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(650.0, dtype='float32')
    std = keras.ops.convert_to_tensor(425.0, dtype='float32')

    return (keras.ops.cast(tensor, 'float32') - mean) / (std + 1e-7)

def normalize_evi(tensor):
    """Normalize evi values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(4250.0, dtype='float32')
    std = keras.ops.convert_to_tensor(750.0, dtype='float32')

    return (keras.ops.cast(tensor, 'float32') - mean) / (std + 1e-7)

def normalize_ndvi(tensor):
    """Normalize ndvi values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(6750.0, dtype='float32')
    std = keras.ops.convert_to_tensor(1200.0, dtype='float32')

    return (keras.ops.cast(tensor, 'float32') - mean) / (std + 1e-7)

def normalize_classification(tensor):
    """Normalize ndvi values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(0.85, dtype='float32')
    std = keras.ops.convert_to_tensor(0.3, dtype='float32')

    return (keras.ops.cast(tensor, 'float32') - mean) / (std + 1e-7)

transform_registry = {
    "gt0": gt0,
    "gt0_bool": gt0_bool,
    "eq2": eq2,
    "normalize_mcwd": normalize_mcwd,
    "normalize_evi": normalize_evi,
    "normalize_ndvi": normalize_ndvi,
    "normalize_classification": normalize_classification,
    "none": lambda x: x
}
