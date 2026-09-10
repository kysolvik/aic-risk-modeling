"""Build a per-pixel burn-frequency climatology from label mosaics.

climatology(pixel) = mean over --years of (label_<year> > 0), a float32 raster on
the label-mosaic grid. This is the free "climatology" baseline used by
`compare_forest_split.py` / `pyramid_compare.py`: the long-run fire frequency at
each pixel. Default years 2013-2022 exclude every evaluation year (2023/2024/2025)
so there is no leakage.

The label mosaics are the full-basin `label_<year>.tif` (a copy of
`gs://aic-amazon/preds/mtsvit_v44_<year>/preds_mask.tif`, pixel-identical to the
per-chip `mask_` rasters).

    .venv/bin/python scripts/analysis/build_climatology.py \
        --label_dir out/label_mosaics --years 2013-2022 \
        --out out/label_mosaics/climatology_2013_2022.tif
"""

import argparse
import os

import numpy as np
import rasterio as rio


def _parse_years(spec):
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label_dir", required=True,
                    help="dir of label_<year>.tif full-basin mosaics")
    ap.add_argument("--years", default="2013-2022",
                    help="'2013-2022' range or a comma list of years")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    years = _parse_years(args.years)
    ref_profile = ref_transform = ref_shape = None
    total = None
    n = 0
    for y in years:
        path = os.path.join(args.label_dir, f"label_{y}.tif")
        with rio.open(path) as ds:
            burned = (ds.read(1) > 0).astype(np.float32)
            if total is None:
                ref_profile = ds.profile
                ref_transform, ref_shape = ds.transform, burned.shape
                total = np.zeros(ref_shape, dtype=np.float32)
            else:
                if burned.shape != ref_shape or ds.transform != ref_transform:
                    raise ValueError(
                        f"{path} grid {burned.shape}/{ds.transform} != reference "
                        f"{ref_shape}/{ref_transform}; mosaics must share a grid")
        total += burned
        n += 1
    if n == 0:
        raise SystemExit("no label mosaics read")
    clim = total / n

    ref_profile.update(dtype=rio.float32, count=1, compress="lzw", nodata=None)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with rio.open(args.out, "w", **ref_profile) as dst:
        dst.write(clim, 1)
    print(f"[climatology] {n} years {years[0]}-{years[-1]} -> {args.out} "
          f"(mean burn freq {clim.mean():.5f}, max {clim.max():.3f})")


if __name__ == "__main__":
    main()
