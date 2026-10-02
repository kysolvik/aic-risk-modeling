"""Export per-chip forest-fraction rasters (forest_<x>-<y>.tif) aligned 1:1 with prediction chips.

Uses the previous year's MapBiomas fraction (the current year marks fresh burns non-forest).
Usage: export_forest_chips.py --data_dir gs://.../allpreds_2024/ --output_dir out/forest/2024/"""

import argparse
import os

import numpy as np

from aic_risk_modeling import train
from aic_risk_modeling.predict.core import TFRECORD_PATTERN, load_profile, write_batch

DEFAULT_PROFILE_TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "assets", "example_v3.tif")


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True,
                    help="TFRecord dir used for the predictions")
    ap.add_argument("--output_dir", required=True,
                    help="output dir for forest_<x>-<y>.tif")
    ap.add_argument("--band", default="im_forest_-1",
                    help="forest-fraction band")
    ap.add_argument("--tfrecord_pattern", default=TFRECORD_PATTERN)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--edge_crop", type=int, default=0,
                    help="must match the predictions' --edge_crop")
    ap.add_argument("--profile_template", default=DEFAULT_PROFILE_TEMPLATE,
                    help="GeoTIFF giving the output CRS and pixel size")
    ap.add_argument("--max_chips", type=int, default=None,
                    help="stop after this many chips")
    return ap.parse_args()


def main():
    args = parse_args()

    ds = train.build_merged_dataset(
        [args.data_dir], args.tfrecord_pattern, batch_size=args.batch_size,
        shuffle=False)

    os.makedirs(args.output_dir, exist_ok=True)
    profile = load_profile(args.profile_template)
    base_transform = profile["transform"]

    n_chips = 0
    for batch in ds:
        if args.band not in batch:
            raise KeyError(
                f"band {args.band!r} not in the TFRecords; available im_ bands "
                f"include e.g. {[k for k in batch if k.startswith('im_forest')]}")
        forest = batch[args.band].numpy().astype(np.float32)
        xs = batch["md_x"].numpy().reshape(-1)
        ys = batch["md_y"].numpy().reshape(-1)
        placeholder = np.zeros_like(forest)
        write_batch(forest, placeholder, xs, ys, base_transform, profile,
                    args.output_dir, args.edge_crop,
                    out_prefix="forest", write_mask=False)
        n_chips += forest.shape[0]
        if args.max_chips and n_chips >= args.max_chips:
            break

    print(f"[export_forest] wrote {n_chips} forest chips to {args.output_dir}",
          flush=True)


if __name__ == "__main__":
    main()
