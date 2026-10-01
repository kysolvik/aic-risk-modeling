"""Full-basin label mosaics (label_<year>.tif) from the targets-only export.

cv_collect_results.py builds its references from `<label_dir>/label_<year>.tif`:
climatology = pixel-wise burn frequency over a fold's TRAIN years (2013..t-1),
persistence = the eval year's previous-year label. The predict runs only write
label mosaics for eval years, so the train-only years (2013-2017) were missing;
for v3 they came from throwaway predict runs (docker/run_mask_calcs.sh). The
targets-only export (`scripts/preprocessing/geebeam_targets_only.py`: one record
per chip, one band per product per year, same chips/md_x/md_y as fullgrid_v3)
gives every year directly, without the model or Cloud Run.

Target = the training label: gt0 on each product, combined with 'any' (the
v3p union4 config's output_features). Default products are union4.

Grid: cv_collect_results.Climatology requires every label mosaic to share one
transform EXACTLY, so chips are placed onto the grid of a reference mosaic from a
real predict run (--grid_from) rather than re-stitched. Re-running predict.py's
write_batch + mosaic.sh locally is NOT equivalent: the container computes chip
origins in float32 and its GDAL writes uint8, so a local stitch lands ~1 cm off
(-8842323.014 vs -8842323.0) and fails the exact-transform check. Chips sit at
whole-pixel offsets on that grid (max fractional offset 1e-3 px) and never
overlap (checked on all 2556), so placement is unambiguous. Chip centre md_x/md_y
-> top-left corner = centre - 64 px, rows north-down (predict's INVERT_YRES=0).

Verify with --check_dir against label mosaics from real v3p predict runs:
pixel-exact agreement on the eval years (2018-2023) is what makes the train-only
years (2013-2017) trustworthy.

Usage:
    .venv/bin/python scripts/analysis/build_label_mosaics.py \
        --data_dir gs://woodwell-aic-fire-risk/data/targets_only \
        --out_dir out/label_mosaics_v3p_union4 --years 2013-2023 \
        --grid_from out/label_mosaics_v3/label_2018.tif
    # verify against predict-run preds_mask.tif copied in as <dir>/label_<y>.tif
    .venv/bin/python scripts/analysis/build_label_mosaics.py --verify_only \
        --out_dir out/label_mosaics_v3p_union4 --check_dir /path/to/preds_masks
"""

import argparse
import json
import os
import re

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
UNION4 = ("im_BurnDate", "im_viirs_snpp", "im_mod14", "im_BurnDate_viirs")
CHIP = 128


def parse_years(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def read_labels(data_dir, years, products):
    """{year: (N, 128, 128) bool}, plus float32 md_x/md_y, from the targets-only TFRecords."""
    import tensorflow as tf
    with tf.io.gfile.GFile(os.path.join(data_dir, "schema.json")) as f:
        schema = json.load(f)["features"]
    bands = {}
    for k in schema:
        m = re.match(r"(im_.*)_(\d{4})$", k)
        if m:
            bands[(m.group(1), int(m.group(2)))] = k
    missing = [(p, y) for y in years for p in products if (p, y) not in bands]
    if missing:
        raise ValueError(f"targets-only export lacks {missing}")
    keys = sorted({bands[(p, y)] for y in years for p in products})
    spec = {k: tf.io.FixedLenFeature([CHIP * CHIP], tf.float32) for k in keys}
    # md_x/md_y parsed as float32, exactly as the predict pipeline sees them
    spec.update(md_id=tf.io.FixedLenFeature([], tf.int64),
                md_x=tf.io.FixedLenFeature([], tf.float32),
                md_y=tf.io.FixedLenFeature([], tf.float32))
    files = sorted(tf.io.gfile.glob(os.path.join(data_dir, "full-*.tfrecord.gz")))
    ids, xs, ys, labels = [], [], [], {y: [] for y in years}
    for rec in tf.data.TFRecordDataset(files, compression_type="GZIP"):
        ex = tf.io.parse_single_example(rec, spec)
        ids.append(int(ex["md_id"]))
        xs.append(ex["md_x"].numpy())
        ys.append(ex["md_y"].numpy())
        for y in years:
            hit = np.zeros(CHIP * CHIP, bool)
            for p in products:
                hit |= ex[bands[(p, y)]].numpy() > 0
            labels[y].append(hit.reshape(CHIP, CHIP))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate md_id in targets-only export")
    print(f"[labels] {len(ids)} chips from {len(files)} shards")
    return (np.array(xs, np.float32), np.array(ys, np.float32),
            {y: np.stack(v) for y, v in labels.items()})


def write_mosaic(year, xs, ys, lab, out_dir, grid_from):
    """Place (N,128,128) chips onto grid_from's grid and write label_<year>.tif."""
    import rasterio as rio
    with rio.open(grid_from) as ref:
        profile, T = ref.profile, ref.transform
        H, W = ref.shape
    res = T.a
    half = CHIP // 2
    col = (xs.astype(np.float64) - half * res - T.c) / res
    row = (T.f - (ys.astype(np.float64) + half * res)) / abs(T.e)
    frac = max(np.abs(col - np.round(col)).max(), np.abs(row - np.round(row)).max())
    if frac > 0.01:
        raise ValueError(f"chips are off {grid_from}'s pixel grid by up to {frac:.3f} px")
    col, row = np.round(col).astype(int), np.round(row).astype(int)
    if col.min() < 0 or row.min() < 0 or col.max() + CHIP > W or row.max() + CHIP > H:
        raise ValueError(f"chips fall outside {grid_from} ({H}x{W})")
    grid = np.zeros((H, W), np.uint8)
    cover = np.zeros((H, W), np.uint8)
    for r, c, m in zip(row, col, lab):
        grid[r:r + CHIP, c:c + CHIP] = m
        cover[r:r + CHIP, c:c + CHIP] += 1
    if cover.max() > 1:
        raise ValueError(f"{year}: chips overlap on the reference grid")
    profile.update(count=1, compress="lzw")
    dst = os.path.join(out_dir, f"label_{year}.tif")
    with rio.open(dst, "w", **profile) as out:
        out.write(grid.astype(profile["dtype"]), 1)
    print(f"[labels] {year}: {int(lab.sum())} burned px -> {dst}")


def verify(out_dir, check_dir):
    """Pixel-exact comparison of label_<y>.tif against check_dir's label_<y>.tif."""
    import rasterio as rio
    ok = True
    for name in sorted(os.listdir(check_dir)):
        m = re.match(r"label_(\d{4})\.tif$", name)
        if not m:
            continue
        mine = os.path.join(out_dir, name)
        if not os.path.exists(mine):
            print(f"[verify] {name}: not built, skip")
            continue
        with rio.open(mine) as a, rio.open(os.path.join(check_dir, name)) as b:
            if a.transform != b.transform or a.shape != b.shape or a.crs != b.crs:
                print(f"[verify] {name}: GRID MISMATCH {a.shape} {a.transform} vs {b.shape} {b.transform}")
                ok = False
                continue
            da, db = a.read(1) > 0, b.read(1) > 0
        n = int((da != db).sum())
        print(f"[verify] {name}: {'OK' if n == 0 else 'MISMATCH'} "
              f"({n} px differ; {int(da.sum())} vs {int(db.sum())} burned)")
        ok &= n == 0
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="gs://woodwell-aic-fire-risk/data/targets_only")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--years", default="2013-2023",
                   help="e.g. 2013-2023 or 2013-2017,2023 (folds need 2013-2023)")
    p.add_argument("--products", nargs="+", default=list(UNION4),
                   help="band prefixes OR-ed (gt0) into the label; default union4")
    p.add_argument("--grid_from", default=os.path.join(REPO, "out", "label_mosaics_v3", "label_2018.tif"),
                   help="mosaic from a real predict run on the same chip grid; output copies its grid")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check_dir", help="dir of label_<y>.tif from predict runs to compare against")
    p.add_argument("--verify_only", action="store_true")
    a = p.parse_args()

    if not a.verify_only:
        years = [y for y in parse_years(a.years)
                 if a.overwrite or not os.path.exists(os.path.join(a.out_dir, f"label_{y}.tif"))]
        if years:
            os.makedirs(a.out_dir, exist_ok=True)
            xs, ys, labels = read_labels(a.data_dir, years, a.products)
            for y in years:
                write_mosaic(y, xs, ys, labels[y], a.out_dir, a.grid_from)
        else:
            print("[labels] all requested years exist (use --overwrite to rebuild)")
    if a.check_dir:
        raise SystemExit(0 if verify(a.out_dir, a.check_dir) else 1)


if __name__ == "__main__":
    main()
