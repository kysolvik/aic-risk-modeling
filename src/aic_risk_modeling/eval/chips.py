"""Prediction-chip I/O and raster helpers shared by the scoring and figure scripts."""

import glob
import os

import numpy as np


def chip_pairs(directory):
    """[(out_path, mask_path)] for the chips directly in `directory`, sorted."""
    out_paths = sorted(glob.glob(os.path.join(directory, "out_*.tif")))
    if not out_paths:
        raise FileNotFoundError(f"no out_*.tif in {directory}")
    pairs = []
    for p in out_paths:
        m = os.path.join(os.path.dirname(p),
                         os.path.basename(p).replace("out_", "mask_", 1))
        if not os.path.exists(m):
            raise FileNotFoundError(f"missing mask for {p}")
        pairs.append((p, m))
    return pairs


def read_window(src, bounds, shape):
    """Band 1 of open raster `src` over a chip's `bounds`; must come out as `shape`."""
    from rasterio.windows import from_bounds
    win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
    v = src.read(1, window=win)
    if v.shape != tuple(shape):
        raise ValueError(f"window {v.shape} != chip {tuple(shape)}")
    return v


def basin_mask(shp_path, crs, transform, shape):
    """(inside, gdf): the RAISG outline in `crs` and its rasterised mask on the grid."""
    import geopandas as gpd
    from rasterio.features import rasterize
    gdf = gpd.read_file(shp_path).to_crs(crs)
    inside = rasterize(((g, 1) for g in gdf.geometry), out_shape=shape,
                       transform=transform, fill=0, dtype="uint8") > 0
    return inside, gdf


def block_nanmean(arr, b):
    """Mean over b x b blocks of a NaN-masked array (all-NaN blocks -> NaN)."""
    h, w = (arr.shape[0] // b) * b, (arr.shape[1] // b) * b
    v = arr[:h, :w].reshape(h // b, b, w // b, b)
    n = np.isfinite(v).sum(axis=(1, 3))
    s = np.nansum(v, axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)
