
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

def normalize_def(tensor):
    """Normalize MCWD values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(200.0, dtype='float32')
    std = keras.ops.convert_to_tensor(50.0, dtype='float32')

    out_tensor = keras.ops.where(tensor < -10000.0,
                                  mean,
                                  tensor)
    return (keras.ops.cast(out_tensor, 'float32') - mean) / (std + 1e-7)

def normalize_pdsi(tensor):
    """Normalize PDSI values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(-100.0, dtype='float32')
    std = keras.ops.convert_to_tensor(50.0, dtype='float32')

    out_tensor = keras.ops.where(tensor < -10000.0,
                                  mean,
                                  tensor)

    return (keras.ops.cast(out_tensor, 'float32') - mean) / (std + 1e-7)

def normalize_tmmn(tensor):
    """Normalize TMMN values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(165.0, dtype='float32')
    std = keras.ops.convert_to_tensor(50.0, dtype='float32')

    out_tensor = keras.ops.where(tensor < -10000.0,
                                  mean,
                                  tensor)
    return (keras.ops.cast(out_tensor, 'float32') - mean) / (std + 1e-7)

def normalize_tmmx(tensor):
    """Normalize TMMX values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(276.0, dtype='float32')
    std = keras.ops.convert_to_tensor(50.0, dtype='float32')

    out_tensor = keras.ops.where(tensor < -10000.0,
                                  mean,
                                  tensor)
    return (keras.ops.cast(out_tensor, 'float32') - mean) / (std + 1e-7)

def normalize_evi(tensor):
    """Normalize evi values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(4000.0, dtype='float32')
    std = keras.ops.convert_to_tensor(1200.0, dtype='float32')

    out_tensor = keras.ops.where(tensor < 0.0,
                                  mean,
                                  tensor)
    return (keras.ops.cast(out_tensor, 'float32') - mean) / (std + 1e-7)

def normalize_vpd(tensor):
    """Normalize vpd values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(60.0, dtype='float32')
    std = keras.ops.convert_to_tensor(50.0, dtype='float32')

    out_tensor = keras.ops.where(tensor < -10000.0,
                                  mean,
                                  tensor)
    return (keras.ops.cast(out_tensor, 'float32') - mean) / (std + 1e-7)

def normalize_ndvi(tensor):
    """Normalize ndvi values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(6750.0, dtype='float32')
    std = keras.ops.convert_to_tensor(1500.0, dtype='float32')

    out_tensor = keras.ops.where(tensor < 0.0,
                                  mean,
                                  tensor)
    return (keras.ops.cast(out_tensor, 'float32') - mean) / (std + 1e-7)

def normalize_classification(tensor):
    """Normalize ndvi values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(0.85, dtype='float32')
    std = keras.ops.convert_to_tensor(0.3, dtype='float32')

    return (keras.ops.cast(tensor, 'float32') - mean) / (std + 1e-7)

def normalize_lat(tensor):
    """Normalize ndvi values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(-5, dtype='float32')
    std = keras.ops.convert_to_tensor(10, dtype='float32')

    return (keras.ops.cast(tensor, 'float32') - mean) / (std + 1e-7)

def normalize_lon(tensor):
    """Normalize ndvi values using precomputed mean and stddev."""
    mean = keras.ops.convert_to_tensor(-50, dtype='float32')
    std = keras.ops.convert_to_tensor(10, dtype='float32')

    return (keras.ops.cast(tensor, 'float32') - mean) / (std + 1e-7)

transform_registry = {
    "gt0": gt0,
    "gt0_bool": gt0_bool,
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
    "none": lambda x: x
}
