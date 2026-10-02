# Rasterize MODIS active-fire hotspots per satellite onto the 927 m MODIS sinusoidal grid.
# Usage: run_modis_prep.sh <input shp> <output dir>
echo $1
echo $2
python preprocess_fire_detections.py \
    $1 \
    $2 \
    --crs "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs" \
    --resolution 926.625433055556 \
    --output-template "modis_{satellite}_{year}.tif" \
    --split-by-satellite \
    --scan-limit 1.5 \
    --confidence-min 29 \
    --extent -8886337.903002782 -2335096.091300001 -4701697.447323891 1167548.0456500005 # rounded to nearest modis pixel

