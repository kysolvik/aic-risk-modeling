#!/usr/bin/env python
"""Rasterize VIIRS S-NPP hotspot detection points into annual geotiffs.

Reads a shapefile of VIIRS hotspot detections and burns them onto a raster
grid, one geotiff per year, where each cell holds the *minimum* (earliest)
date of burn among the detections that fall in it. Output projection,
resolution, and extent are specified on the command line.

Run python preprocess_viirs_snpp.py --help for options.

Date encoding (`--date-encoding`):
  doy       day-of-year, 1-366 (default; matches MODIS MCD64A1 BurnDate)
  yyyymmdd  integer calendar date, e.g. 20230115

Cells with no detection are set to `--nodata` (default 0).
"""

import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_origin

# Encoding name -> raster dtype able to hold the encoded values.
DATE_ENCODINGS = {
    "doy": "uint16",       # 1-366
    "yyyymmdd": "int32",   # e.g. 20230115 exceeds uint16
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to the input VIIRS hotspot shapefile")
    parser.add_argument(
        "output_dir", help="Directory to write the annual geotiffs into"
    )
    parser.add_argument(
        "--crs",
        required=True,
        help="Output projection, e.g. 'EPSG:4326' or a PROJ/WKT string",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        required=True,
        help="Output cell size, in units of --crs (square pixels)",
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        required=True,
        help="Output extent, in units of --crs",
    )
    parser.add_argument(
        "--date-field",
        default="ACQ_DATE",
        help="Name of the acquisition-date field (default: ACQ_DATE)",
    )
    parser.add_argument(
        "--date-encoding",
        choices=sorted(DATE_ENCODINGS),
        default="doy",
        help="How to encode the burn date in each cell (default: doy)",
    )
    parser.add_argument(
        "--src-crs",
        default=None,
        help="Source CRS to assume if the shapefile has none (e.g. 'EPSG:4326')",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Only process these years (default: every year present in the data)",
    )
    parser.add_argument(
        "--nodata",
        type=int,
        default=0,
        help="Value for cells with no detection (default: 0)",
    )
    parser.add_argument(
        "--output-template",
        default="viirs_snpp_{year}.tif",
        help="Output filename template (default: viirs_snpp_{year}.tif)",
    )
    parser.add_argument(
        "--compress",
        default="lzw",
        help="GeoTIFF compression (default: lzw); use 'none' to disable",
    )
    return parser.parse_args()


def encode_dates(dates, encoding):
    """Encode a series of datetimes to integer cell values per `encoding`."""
    dates = pd.to_datetime(dates)
    if encoding == "doy":
        return dates.dt.dayofyear.to_numpy()
    if encoding == "yyyymmdd":
        return (
            dates.dt.year * 10000 + dates.dt.month * 100 + dates.dt.day
        ).to_numpy()
    raise ValueError(f"Unknown date encoding: {encoding}")


def build_transform(extent, resolution):
    """Return (transform, width, height) for the output grid."""
    xmin, ymin, xmax, ymax = extent
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid extent {extent}: need xmin<xmax and ymin<ymax")
    width = int(round((xmax - xmin) / resolution))
    height = int(round((ymax - ymin) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Extent {extent} and resolution {resolution} yield an empty grid"
        )
    transform = from_origin(xmin, ymax, resolution, resolution)
    return transform, width, height


def rasterize_min_date(geometries, values, transform, out_shape, dtype, nodata):
    """Burn `values` onto the grid, keeping the minimum value per cell.

    rasterio burns shapes in order and later shapes overwrite earlier ones,
    so sorting descending makes the smallest (earliest) date land last and win.
    """
    order = np.argsort(values, kind="stable")[::-1]
    shapes = (
        (geom, int(val))
        for geom, val in zip(geometries[order], values[order])
    )
    return rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=nodata,
        dtype=dtype,
        merge_alg=MergeAlg.replace,
    )


def main():
    args = parse_args()

    dtype = DATE_ENCODINGS[args.date_encoding]
    transform, width, height = build_transform(args.extent, args.resolution)
    os.makedirs(args.output_dir, exist_ok=True)

    gdf = gpd.read_file(args.input)
    if args.date_field not in gdf.columns:
        raise KeyError(
            f"Date field '{args.date_field}' not found; columns: {list(gdf.columns)}"
        )

    # Establish source CRS, then reproject to the requested output projection.
    if gdf.crs is None:
        if args.src_crs is None:
            raise ValueError(
                "Shapefile has no CRS; pass --src-crs to declare the source projection"
            )
        gdf = gdf.set_crs(args.src_crs)
    gdf = gdf.to_crs(args.crs)

    # Drop rows we can't place or date.
    gdf = gdf[gdf.geometry.notna() & gdf[args.date_field].notna()]
    dates = pd.to_datetime(gdf[args.date_field], errors="coerce")
    gdf = gdf[dates.notna()]
    dates = dates[dates.notna()]

    values = encode_dates(dates, args.date_encoding)
    geometries = gdf.geometry.values
    year = dates.dt.year.to_numpy()

    years = sorted(np.unique(year)) if args.years is None else sorted(args.years)

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "height": height,
        "width": width,
        "crs": args.crs,
        "transform": transform,
        "nodata": args.nodata,
        "tiled": True,
    }
    if args.compress and args.compress.lower() != "none":
        profile["compress"] = args.compress

    for yr in years:
        mask = year == yr
        n = int(mask.sum())
        if n == 0:
            print(f"{yr}: no detections, skipping")
            continue
        burned = rasterize_min_date(
            geometries[mask],
            values[mask],
            transform,
            (height, width),
            dtype,
            args.nodata,
        )
        out_path = os.path.join(args.output_dir, args.output_template.format(year=yr))
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(burned, 1)
        n_cells = int((burned != args.nodata).sum())
        print(f"{yr}: {n} detections -> {n_cells} burned cells -> {out_path}")


if __name__ == "__main__":
    main()
