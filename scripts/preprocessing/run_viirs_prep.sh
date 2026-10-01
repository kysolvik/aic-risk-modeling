# Set to MODIS tile grid
echo $1 # input shp
echo $2 # output dir
python preprocess_fire_detections.py \
    $1 \
    $2 \
    --crs "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs" \
    --resolution 463.312716527778 \
    --output-template "viirs_noaa20_{year}.tif" \
    --extent -8886337.903002782 -2335096.091300001 -4701697.447323891 1167548.0456500005 # rounded to nearest modis pixel
#    --output-template "viirs_snpp_{year}.tif" \ # for snpp

