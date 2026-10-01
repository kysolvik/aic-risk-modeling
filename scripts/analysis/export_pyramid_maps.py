"""Export the Fig 3 (pyramid) eval fields as GeoTIFFs, one per pyramid level.

For each eval year of one CV fold and each block size b, writes the three fields
the pyramid figure scores, pooled exactly as `pyramid_compare` pools them, as a
raster with b x 463 m pixels, masked to the scored chip footprint (else nodata -1):

    b<b>/factored_<year>.tif    Factored (yeargain) prediction, MEAN-pooled
                                (raw model output: pos_weight-inflated,
                                rank-equivalent to deflated)
    b<b>/burnfreq_<clim>.tif    fold climatology (pixel burn frequency over the
                                train years), MEAN-pooled; same for every eval year
    b<b>/label_<year>.tif       union4 ground truth, ANY-burn (max-pooled 0/1)

`--score_pool max` max-pools the two score fields instead (written as
factored_<year>_max.tif / burnfreq_<clim>_max.tif), matching
`make_pyramid_figure.py --score_pool max`.

Chips are 128 px and aligned to the mosaic's 128-px grid, so pooling the mosaic
in b x b blocks (b | 128) never crosses a chip edge -- identical to the eval's
within-chip pooling (checked at runtime).

Inputs are the existing per-year `preds_out.tif` mosaics (verified identical to
the per-chip out_*.tif rasters) and `out/label_mosaics_v3p_union4/`.

    .venv/bin/python scripts/analysis/export_pyramid_maps.py
"""

import argparse
import glob
import os

import numpy as np
import rasterio as rio
from rasterio.windows import from_bounds

NODATA = -1.0


def chip_footprint(chips_dir, shape, transform):
    fp = np.zeros(shape, bool)
    for p in glob.glob(os.path.join(chips_dir, "out_*.tif")):
        with rio.open(p) as s:
            w = from_bounds(*s.bounds, transform=transform).round_offsets().round_lengths()
        fp[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width] = True
    return fp


def pool(arr, b, how):
    h, w = arr.shape
    blocks = arr.reshape(h // b, b, w // b, b)
    return blocks.max(axis=(1, 3)) if how == "max" else blocks.mean(axis=(1, 3), dtype=np.float64)


def write(path, arr, footprint, profile, b):
    out = np.where(footprint, arr.astype(np.float32), NODATA)
    prof = dict(profile, dtype="float32", count=1, nodata=NODATA, compress="deflate",
                height=out.shape[0], width=out.shape[1],
                transform=profile["transform"] * rio.Affine.scale(b))
    if min(out.shape) >= 256:
        prof.update(tiled=True, blockxsize=256, blockysize=256)
    else:
        prof.update(tiled=False)
        prof.pop("blockxsize", None), prof.pop("blockysize", None)
    with rio.open(path, "w", **prof) as d:
        d.write(out, 1)
    print(f"[pyramid_maps] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds_root", default="out/cv/preds")
    ap.add_argument("--arch", default="factored_v3p_union4_monthlyattn_wide_yeargain")
    ap.add_argument("--fold", default="fwdpair_2022")
    ap.add_argument("--years", default="2022,2023")
    ap.add_argument("--clim", default="2013_2021", help="fold train years, as in climatology_<clim>.tif")
    ap.add_argument("--blocks", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--score_pool", choices=["mean", "max"], default="mean")
    ap.add_argument("--label_dir", default="out/label_mosaics_v3p_union4")
    ap.add_argument("--out_dir", default=None, help="default out/pyramid_maps/<fold>")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join("out/pyramid_maps", args.fold)
    blocks = [int(b) for b in args.blocks.split(",")]
    sp = args.score_pool
    suffix = "" if sp == "mean" else f"_{sp}"
    for b in blocks:
        if 128 % b:
            raise ValueError(f"block {b} does not divide the 128-px chip")
        os.makedirs(os.path.join(out_dir, f"b{b}"), exist_ok=True)

    with rio.open(os.path.join(args.label_dir, f"climatology_{args.clim}.tif")) as s:
        clim, profile = s.read(1), s.profile
    union_fp = np.zeros(clim.shape, bool)
    for y in [int(v) for v in args.years.split(",")]:
        ydir = os.path.join(args.preds_root, args.arch, args.fold, str(y))
        with rio.open(os.path.join(ydir, "preds_out.tif")) as s:
            pred = s.read(1)
            if s.transform != profile["transform"] or pred.shape != clim.shape:
                raise ValueError(f"{ydir}/preds_out.tif is not on the label grid")
        with rio.open(os.path.join(args.label_dir, f"label_{y}.tif")) as s:
            label = s.read(1) > 0
        fp = chip_footprint(os.path.join(ydir, "chips"), clim.shape, profile["transform"])
        union_fp |= fp
        print(f"[pyramid_maps] {y}: {fp.sum() // 128**2} chips")
        for b in blocks:
            bfp = pool(fp, b, "max")
            if not np.array_equal(bfp, pool(fp, b, "min")):
                raise ValueError(f"block {b} straddles a chip edge")
            lab_b = pool(label, b, "max")
            print(f"  b{b}: {bfp.sum()} blocks, prevalence {lab_b[bfp].mean():.4f}")
            d = os.path.join(out_dir, f"b{b}")
            write(os.path.join(d, f"factored_{y}{suffix}.tif"), pool(pred, b, sp), bfp, profile, b)
            if not suffix:
                write(os.path.join(d, f"label_{y}.tif"), lab_b, bfp, profile, b)
    for b in blocks:
        write(os.path.join(out_dir, f"b{b}", f"burnfreq_{args.clim}{suffix}.tif"),
              pool(clim, b, sp), pool(union_fp, b, "max"), profile, b)


if __name__ == "__main__":
    main()
