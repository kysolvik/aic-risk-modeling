#!/usr/bin/env bash
# Mosaic the per-chip rasters written by predict.py into single GeoTIFFs.
#
#   mosaic.sh <chip_dir> <out_dir> [name]
#
set -euo pipefail

in_dir=$1
out_dir=$2
name=${3:-preds}

mkdir -p "$out_dir"
for pfx in out mask; do
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
    rm -f "$out_dir/${name}_${pfx}.vrt" "$list"
done
