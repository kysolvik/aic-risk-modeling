# Rasterize VIIRS NOAA-20 hotspots onto the 463 m MODIS sinusoidal grid (SNPP: viirs_snpp_{year}.tif).
# Usage: run_viirs_prep.sh <input shp> <output dir>
echo $1
echo $2
python preprocess_fire_detections.py \
    $1 \
    $2 \
    --crs "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs" \
    --resolution 463.312716527778 \
    --output-template "viirs_noaa20_{year}.tif" \
    --extent -8886337.903002782 -2335096.091300001 -4701697.447323891 1167548.0456500005 # rounded to nearest modis pixel

