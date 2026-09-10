"""Export per-chip MapBiomas forest-fraction rasters aligned 1:1 with predictions.

The prediction chips (`out_<x>-<y>.tif` / `mask_<x>-<y>.tif`, written by
`scripts/predict/predict.py`) carry only a fire score and the burn label -- no
land cover. To split evaluation by forested vs non-forested land we need the
forest fraction on the *same* grid, one raster per chip.

`im_forest_-1` is the previous year's MapBiomas forest fraction in [0, 1] (class
< 10, mean-reduced to the 0.005 deg model grid; see
`scripts/preprocessing/geebeam_ali_inputs.py`). The PREVIOUS year is used on
purpose: the current-year map reclassifies pixels that just burned as
non-forest, which would define the stratum by the very outcome we score. It also
lives only in the input TFRecords, not in the prediction chips.

This script reads that band straight out of the input TFRecords and writes it
through the *identical* georeferencing path as the predictions -- it imports and
reuses `predict.write_batch` with the same center coords (`md_x`/`md_y`), profile
template and `--edge_crop`/`--invert_yres` flags. So each written
`forest_<x>-<y>.tif` registers pixel-for-pixel with the matching
`out_<x>-<y>.tif` and shares the exact `<x>-<y>` filename, and evaluation can join
the two by filename.

Run once per prediction year, pointing at the SAME data_dir the predictions came
from and with the SAME flags:

    .venv/bin/python scripts/analysis/export_forest_chips.py \
        --data_dir gs://aic-amazon/data/fullgrid_v2/allpreds_2023/ \
        --output_dir out/forest/2023/
    # repeat for allpreds_2024 -> out/forest/2024/, 2025 -> out/forest/2025/
"""

import argparse
import os
import sys

import numpy as np
import rasterio as rio

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "predict"))

import aic_risk_modeling as arm  # noqa: E402
from predict import write_batch, DEFAULT_PROFILE_TEMPLATE  # noqa: E402

TFRECORD_PATTERN = "*.tfrecord.gz"


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True,
                    help="input TFRecord dir (the same allpreds_<year>/ used for "
                         "prediction), local or gs://")
    ap.add_argument("--output_dir", required=True,
                    help="where to write forest_<x>-<y>.tif chips")
    ap.add_argument("--band", default="im_forest_-1",
                    help="forest-fraction band to export (default im_forest_-1, "
                         "the previous year's MapBiomas forest fraction)")
    ap.add_argument("--tfrecord_pattern", default=TFRECORD_PATTERN)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--edge_crop", type=int, default=0,
                    help="MUST match the --edge_crop used to write the predictions")
    # The prediction pipeline inverts y by default (docker entrypoint
    # INVERT_YRES=1), because assets/example.tif carries a +0.005 yres. Match it
    # so the forest chips register with the out_/mask_ chips; --no_invert_yres is
    # only for predictions written without it.
    ap.add_argument("--no_invert_yres", dest="invert_yres", action="store_false",
                    help="write with the raw (+yres) transform; use only if the "
                         "predictions were written WITHOUT --invert_yres")
    ap.set_defaults(invert_yres=True)
    ap.add_argument("--profile_template", default=DEFAULT_PROFILE_TEMPLATE,
                    help="GeoTIFF supplying the output CRS and pixel size")
    ap.add_argument("--max_chips", type=int, default=None,
                    help="stop after this many chips (smoke runs)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Raw (un-normalized) parse of every band; shuffle=False so we stream once.
    # We only touch the forest band and the raw center coords, but the merged
    # loader is what predict.py uses, so the chips line up by construction.
    ds = arm.train.build_merged_dataset(
        [args.data_dir], args.tfrecord_pattern, batch_size=args.batch_size,
        cache=False, axis="examples", shuffle=False)

    os.makedirs(args.output_dir, exist_ok=True)
    with rio.open(args.profile_template) as src:
        profile = src.profile
    profile.update(dtype=rio.float32, count=1, compress="lzw")
    base_transform = profile["transform"]

    n_chips = 0
    for batch in ds:
        if args.band not in batch:
            raise KeyError(
                f"band {args.band!r} not in the TFRecords; available im_ bands "
                f"include e.g. {[k for k in batch if k.startswith('im_forest')]}")
        forest = batch[args.band].numpy().astype(np.float32)   # (B, H, W)
        xs = batch["md_x"].numpy().reshape(-1)                 # raw center lon
        ys = batch["md_y"].numpy().reshape(-1)                 # raw center lat
        # write_batch indexes a mask per item even when write_mask=False; give it
        # a same-shaped placeholder it will never write.
        placeholder = np.zeros_like(forest)
        write_batch(forest, placeholder, xs, ys, base_transform, profile,
                    args.output_dir, args.edge_crop, args.invert_yres,
                    out_prefix="forest", write_mask=False)
        n_chips += forest.shape[0]
        if args.max_chips and n_chips >= args.max_chips:
            break

    print(f"[export_forest] wrote {n_chips} forest chips to {args.output_dir}",
          flush=True)


if __name__ == "__main__":
    main()
