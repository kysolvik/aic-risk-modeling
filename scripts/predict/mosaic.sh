#!/usr/bin/env bash
# Mosaic per-chip rasters from predict.py / attribute.py into one GeoTIFF per prefix.
# Usage: mosaic.sh <chip_dir> <out_dir> [name=preds] [prefixes="out mask"]
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
    # BIGTIFF=YES: IF_SAFER can pick classic TIFF under LZW and fail mid-write on a full basin.
    gdal_translate -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES \
        "$out_dir/${name}_${pfx}.vrt" "$out_dir/${name}_${pfx}.tif"
    # gdalbuildvrt drops band descriptions; copy them from one chip.
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
