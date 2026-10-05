#!/usr/bin/env python
"""Rasterize VIIRS or MODIS hotspot points into annual GeoTIFFs holding each cell's earliest burn date.

Optional SCAN/TRACK/CONFIDENCE filters and per-satellite splitting ({satellite} in --output-template).
Usage: preprocess_fire_detections.py <input.shp> <output_dir> --crs C --resolution R --extent ..."""

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
    "doy": "uint16",
    "yyyymmdd": "int32",
}

# VIIRS categorical CONFIDENCE -> ordinal, so --confidence-min 0 drops low.
VIIRS_CONFIDENCE = {"l": 0, "n": 1, "h": 2}


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
        "--scan-limit",
        type=float,
        default=None,
        help="keep SCAN < this (drops large off-nadir pixels)",
    )
    parser.add_argument(
        "--track-limit",
        type=float,
        default=None,
        help="keep TRACK < this (drops large off-nadir pixels)",
    )
    parser.add_argument(
        "--confidence-min",
        type=float,
        default=None,
        help="keep CONFIDENCE > this; VIIRS l/n/h = 0/1/2",
    )
    parser.add_argument(
        "--satellite",
        nargs="+",
        default=None,
        help="keep only these SATELLITE values (case-insensitive)",
    )
    parser.add_argument(
        "--split-by-satellite",
        action="store_true",
        help="one raster per SATELLITE; template needs {satellite}",
    )
    parser.add_argument(
        "--src-crs",
        default=None,
        help="source CRS if the shapefile has none",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="only these years (default: all)",
    )
    parser.add_argument(
        "--nodata",
        type=int,
        default=0,
        help="value for cells with no detection",
    )
    parser.add_argument(
        "--output-template",
        default="viirs_snpp_{year}.tif",
        help="output filename template",
    )
    parser.add_argument(
        "--compress",
        default="lzw",
        help="GeoTIFF compression; 'none' to disable",
    )
    return parser.parse_args()


def encode_dates(dates, encoding):
    dates = pd.to_datetime(dates)
    if encoding == "doy":
        return dates.dt.dayofyear.to_numpy()
    if encoding == "yyyymmdd":
        return (
            dates.dt.year * 10000 + dates.dt.month * 100 + dates.dt.day
        ).to_numpy()
    raise ValueError(f"Unknown date encoding: {encoding}")


def apply_attribute_filters(gdf, scan_limit, track_limit, confidence_min):
    """Drop detections with SCAN >= scan_limit, TRACK >= track_limit or CONFIDENCE <= confidence_min."""
    filters = [
        ("SCAN", scan_limit, lambda col, lim: col < lim),
        ("TRACK", track_limit, lambda col, lim: col < lim),
        ("CONFIDENCE", confidence_min, lambda col, lim: col > lim),
    ]
    for field, limit, keep in filters:
        if limit is None:
            continue
        if field not in gdf.columns:
            raise KeyError(
                f"Filter field '{field}' not found; columns: {list(gdf.columns)}"
            )
        col = gdf[field]
        if field == "CONFIDENCE" and not pd.api.types.is_numeric_dtype(col):
            # VIIRS categorical confidence: low/nominal/high -> 0/1/2.
            col = col.astype(str).str.strip().str.lower().map(VIIRS_CONFIDENCE)
        col = pd.to_numeric(col, errors="coerce")
        if col.isna().all():
            raise ValueError(
                f"Field '{field}' has no numeric values; cannot apply a numeric limit"
            )
        n_before = len(gdf)
        gdf = gdf[keep(col, limit).to_numpy()]
        print(f"{field} filter ({limit}): kept {len(gdf)}/{n_before} detections")
    return gdf


def satellite_labels(gdf):
    if "SATELLITE" not in gdf.columns:
        raise KeyError(
            f"Field 'SATELLITE' not found; columns: {list(gdf.columns)}"
        )
    return gdf["SATELLITE"].astype(str).str.strip()


def apply_satellite_filter(gdf, satellites):
    if satellites is None:
        return gdf
    wanted = {s.strip().lower() for s in satellites}
    labels = satellite_labels(gdf).str.lower()
    missing = wanted - set(labels.unique())
    if missing:
        print(f"Warning: SATELLITE values not present in data: {sorted(missing)}")
    n_before = len(gdf)
    gdf = gdf[labels.isin(wanted).to_numpy()]
    print(f"SATELLITE filter ({sorted(wanted)}): kept {len(gdf)}/{n_before} detections")
    return gdf


def build_transform(extent, resolution):
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
    """Burn `values` onto the grid keeping the per-cell minimum (sorted descending so the earliest wins)."""
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

    if args.split_by_satellite and "{satellite}" not in args.output_template:
        raise ValueError(
            "--split-by-satellite needs {satellite} in --output-template, "
            f"got {args.output_template!r}"
        )
    dtype = DATE_ENCODINGS[args.date_encoding]
    transform, width, height = build_transform(args.extent, args.resolution)
    os.makedirs(args.output_dir, exist_ok=True)

    gdf = gpd.read_file(args.input)
    if args.date_field not in gdf.columns:
        raise KeyError(
            f"Date field '{args.date_field}' not found; columns: {list(gdf.columns)}"
        )

    if gdf.crs is None:
        if args.src_crs is None:
            raise ValueError(
                "Shapefile has no CRS; pass --src-crs to declare the source projection"
            )
        gdf = gdf.set_crs(args.src_crs)
    gdf = gdf.to_crs(args.crs)

    gdf = gdf[gdf.geometry.notna() & gdf[args.date_field].notna()]
    gdf = apply_attribute_filters(
        gdf, args.scan_limit, args.track_limit, args.confidence_min
    )
    gdf = apply_satellite_filter(gdf, args.satellite)
    dates = pd.to_datetime(gdf[args.date_field], errors="coerce")
    gdf = gdf[dates.notna()]
    dates = dates[dates.notna()]

    values = encode_dates(dates, args.date_encoding)
    geometries = gdf.geometry.values
    year = dates.dt.year.to_numpy()

    years = sorted(np.unique(year)) if args.years is None else sorted(args.years)

    # (name, row mask) per output group; name fills {satellite} in the template.
    if args.split_by_satellite:
        labels = satellite_labels(gdf).str.lower().to_numpy()
        groups = [(sat, labels == sat) for sat in sorted(np.unique(labels))]
    else:
        groups = [(None, np.ones(len(gdf), dtype=bool))]

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

    for sat, sat_mask in groups:
        prefix = f"{sat} " if sat is not None else ""
        for yr in years:
            mask = sat_mask & (year == yr)
            n = int(mask.sum())
            if n == 0:
                print(f"{prefix}{yr}: no detections, skipping")
                continue
            burned = rasterize_min_date(
                geometries[mask],
                values[mask],
                transform,
                (height, width),
                dtype,
                args.nodata,
            )
            out_name = args.output_template.format(year=yr, satellite=sat)
            out_path = os.path.join(args.output_dir, out_name)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(burned, 1)
            n_cells = int((burned != args.nodata).sum())
            print(
                f"{prefix}{yr}: {n} detections -> {n_cells} burned cells -> {out_path}"
            )


if __name__ == "__main__":
    main()
