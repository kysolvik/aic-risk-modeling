#!/usr/bin/env bash
# Mosaic the per-chip rasters written by predict.py / attribute.py into single
# GeoTIFFs.
#
#   mosaic.sh <chip_dir> <out_dir> [name] [prefixes]
#
# prefixes is a space-separated list of per-chip filename prefixes to stitch,
# defaulting to the predict pair "out mask"; attribute runs pass "shap" or
# "attr" (their rasters are named {prefix}_{x}-{y}.tif).
set -euo pipefail

in_dir=$1
out_dir=$2
name=${3:-preds}
prefixes=${4:-out mask}

mkdir -p "$out_dir"
for pfx in $prefixes; do
    list="$out_dir/.${pfx}.list"
    find "$in_dir" -maxdepth 1 -name "${pfx}_*.tif" | sort > "$list"
    if [ ! -s "$list" ]; then
        echo "[mosaic] no ${pfx}_*.tif in $in_dir, skipping"
        rm -f "$list"
        continue
    fi
    echo "[mosaic] $(wc -l < "$list") ${pfx} chips -> ${name}_${pfx}.tif"
    gdalbuildvrt -input_file_list "$list" "$out_dir/${name}_${pfx}.vrt"
    gdal_translate -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER \
        "$out_dir/${name}_${pfx}.vrt" "$out_dir/${name}_${pfx}.tif"
    # gdalbuildvrt drops per-band descriptions, so a Shapley/attr mosaic would
    # open as "Band 1..10". Copy them off one chip (rasterio ships in the image);
    # a no-op for the single-band, unnamed out/mask rasters.
    python - "$(head -n1 "$list")" "$out_dir/${name}_${pfx}.tif" <<'PY'
import sys
import rasterio
src, dst = sys.argv[1], sys.argv[2]
with rasterio.open(src) as s:
    descs = list(s.descriptions)
if any(descs):
    with rasterio.open(dst, 'r+') as d:
        for i, desc in enumerate(descs):
            if desc:
                d.set_band_description(i + 1, desc)
PY
    rm -f "$out_dir/${name}_${pfx}.vrt" "$list"
done
