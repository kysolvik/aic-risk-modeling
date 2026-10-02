"""Chip-year burn panel 2001-2025 from the targets-only export, for the long gamma fit.

Columns: burn_<product>, union counts, prev_<product> (year-1), lagged md_<index>_<slice>.
Products absent in a year are NaN, never 0.
Usage: build_target_panel.py --data_dir gs://.../targets_only --out out/target_panel/panel.parquet"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from aic_risk_modeling.preprocess.climate_indices import download_clim_indices

PRODUCTS = {"bd": "im_BurnDate", "mod14": "im_mod14", "terra": "im_terra",
            "aqua": "im_aqua", "snpp": "im_viirs_snpp", "vnp64": "im_BurnDate_viirs"}
UNIONS = {"bd_mod14": ("bd", "mod14"),
          "union4": ("bd", "snpp", "mod14", "vnp64")}
CLIM = ("soi", "tna", "oni", "mei", "amo")
FIRST_YEAR = 2001
LAST_YEAR = 2025


def _feature_spec(schema):
    spec = {}
    for k, t in schema.items():
        if k.startswith("im_"):
            spec[k] = tf.io.FixedLenFeature([128 * 128], tf.float32)
        elif t == "int64":
            spec[k] = tf.io.FixedLenFeature([], tf.int64)
        elif t == "float":
            spec[k] = tf.io.FixedLenFeature([], tf.float32)
        else:
            spec[k] = tf.io.FixedLenFeature([], tf.string)
    return spec


def read_counts(data_dir):
    with tf.io.gfile.GFile(os.path.join(data_dir, "schema.json")) as f:
        schema = json.load(f)["features"]
    bands = {}
    for k in schema:
        m = re.match(r"(im_.*)_(\d{4})$", k)
        if m:
            bands[(m.group(1), int(m.group(2)))] = k
    inv = {v: k for k, v in PRODUCTS.items()}
    files = sorted(tf.io.gfile.glob(os.path.join(data_dir, "full-*.tfrecord.gz")))
    spec = _feature_spec(schema)
    rows = []
    for rec in tf.data.TFRecordDataset(files, compression_type="GZIP"):
        ex = tf.io.parse_single_example(rec, spec)
        base = {"md_id": int(ex["md_id"]), "md_x": float(ex["md_x"]),
                "md_y": float(ex["md_y"])}
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            row = dict(base, year=year)
            masks = {}
            for prefix, prod in inv.items():
                key = bands.get((prefix, year))
                if key is None:
                    row[f"burn_{prod}"] = np.nan
                    continue
                masks[prod] = ex[key].numpy() > 0
                row[f"burn_{prod}"] = int(masks[prod].sum())
            for name, parts in UNIONS.items():
                if all(p in masks for p in parts):
                    row[f"burn_{name}"] = int(np.logical_or.reduce([masks[p] for p in parts]).sum())
                else:
                    row[f"burn_{name}"] = np.nan
            rows.append(row)
    d = pd.DataFrame(rows)
    n = d.groupby("year").md_id.nunique()
    if n.nunique() != 1 or set(d.md_id.unique()) != set(range(n.iloc[0])):
        raise ValueError(f"chip ids not contiguous 0..N-1 in every year: {n.to_dict()}")
    return d


def add_prev(d):
    d = d.sort_values(["md_id", "year"]).reset_index(drop=True)
    burn = [c for c in d.columns if c.startswith("burn_")]
    prev = d.groupby("md_id")[burn].shift(1)
    # shift(1) is year-1 only because every chip has every year (checked in read_counts)
    prev.columns = [c.replace("burn_", "prev_") for c in burn]
    return pd.concat([d, prev], axis=1)


def climate_table(first_year, last_year):
    """One row per label year with index slices over Y-1 / Y-2 (all drivers lag)."""
    rows = {y: {} for y in range(first_year, last_year + 1)}
    for name in CLIM:
        s = download_clim_indices(name, first_year - 2, last_year - 1)["metric"]
        for y in rows:
            y1 = s[s.index.year == y - 1]
            y2 = s[s.index.year == y - 2]
            rows[y][f"md_{name}_y1ond"] = float(y1[y1.index.month >= 10].mean())
            rows[y][f"md_{name}_y1"] = float(y1.mean())
            rows[y][f"md_{name}_y2"] = float(y2.mean())
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("year").reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    d = add_prev(read_counts(a.data_dir))
    d = d.merge(climate_table(FIRST_YEAR, LAST_YEAR), on="year", how="left", validate="m:1")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    d.to_parquet(a.out)
    print(f"wrote {a.out}: {len(d)} rows, {d.md_id.nunique()} chips, "
          f"years {d.year.min()}-{d.year.max()}")


if __name__ == "__main__":
    main()
