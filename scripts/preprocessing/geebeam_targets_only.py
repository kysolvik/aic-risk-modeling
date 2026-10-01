"""Execute GEE tile extraction in Beam + Dataflow"""

import argparse
import itertools
import logging

from aic_risk_modeling.preprocess import download_clim_indices
import ee
import google
import numpy as np

import geebeam

# Get default project id from environment (or specify PROJECT_ID manually)
DF_PROJECT_ID = google.auth.default()[1]
EE_PROJECT_ID = 'tropics-woodwell'

MONTH_START = 1
MONTH_END = 12
DAY_END = '31' # Set to num days in MONTH_END
MONTH_NAMES=(np.arange(MONTH_END) - MONTH_END).astype(str)

ee.Initialize(project=EE_PROJECT_ID)

# MODIS MCD64 fire memory
def prep_mcd64_year(y):
    mcd64 = (ee.ImageCollection('MODIS/061/MCD64A1')
             .select('BurnDate')
             .filter(ee.Filter.calendarRange(y, y, 'year'))
             .max()
             .unmask()
             )
    band_names = mcd64.bandNames().getInfo()
    band_names_new = [f'{b}_{y}' for b in band_names]
    mcd64 = mcd64.rename(band_names_new)
    return mcd64

mcd64_list = [prep_mcd64_year(y) for y in range(2000, 2026)]

# VIIRS VNP64 fire
def prep_vnp64_year(y):
    vnp64 = (ee.ImageCollection('NASA/VIIRS/002/VNP64A1')
             .select('Burn_Date')
             .filter(ee.Filter.calendarRange(y, y, 'year'))
             .max()
             .unmask()
             ).rename('BurnDate_viirs')
    band_names = vnp64.bandNames().getInfo()
    band_names_new = [f'{b}_{y}' for b in band_names]
    vnp64 = vnp64.rename(band_names_new)
    return vnp64

vnp64_list = [prep_vnp64_year(y) for y in range(2013, 2026)]

# VIIRS SNPP
def prep_viirs_year(y):
    viirs_snpp = (ee.ImageCollection('projects/mmacedo-reservoirid/assets/viirs_snpp_archive_msgrid')
                  .filter(ee.Filter.calendarRange(y, y, 'year'))
                  ).max().unmask().rename(f'viirs_snpp_{y}')
    return viirs_snpp

viirs_snpp_memory = [prep_viirs_year(y) for y in range(2013, 2026)]

# VIIRS NOAA20 fire target
def prep_viirs_noaa20_year(y):
    viirs_noaa20 = (ee.ImageCollection('projects/mmacedo-reservoirid/assets/viirs_noaa20_archive_msgrid')
                 .filter(ee.Filter.calendarRange(y, y, 'year'))
                 ).max().unmask().rename(f'viirs_noaa20_{y}')
    return viirs_noaa20
viirs_noaa20_memory = [prep_viirs_noaa20_year(y) for y in range(2018, 2026)]

# MOD14 active fire
def prep_mod14_year(y):
    mod14 = (ee.ImageCollection('projects/mmacedo-reservoirid/assets/mod14_archive_msgrid')
             .filter(ee.Filter.calendarRange(y, y, 'year'))
             ).max().unmask().rename(f'mod14_{y}')
    return mod14

mod14_memory = [prep_mod14_year(y) for y in range(2002, 2026)]
    
# Aqua active fire
def prep_aqua_year(y):
    aqua = (ee.ImageCollection('projects/mmacedo-reservoirid/assets/mod14_aqua_archive_msgrid')
             .filter(ee.Filter.calendarRange(y, y, 'year'))
             ).max().unmask().rename(f'aqua_{y}')
    return aqua

aqua_memory = [prep_aqua_year(y) for y in range(2002, 2026)]

# Terra active fire
def prep_terra_year(y):
    terra = (ee.ImageCollection('projects/mmacedo-reservoirid/assets/mod14_terra_archive_msgrid')
             .filter(ee.Filter.calendarRange(y, y, 'year'))
             ).max().unmask().rename(f'terra_{y}')
    return terra

terra_memory = [prep_terra_year(y) for y in range(2000, 2026)]

# Note that with split processing each will be processed separately
im_list = mcd64_list + viirs_snpp_memory + viirs_noaa20_memory + mod14_memory + aqua_memory + terra_memory + vnp64_list

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # Execute
    geebeam.grid_and_run_pipeline(
        image_list = im_list,
        project=EE_PROJECT_ID,
        dataflow_project=DF_PROJECT_ID,
        crs="SR-ORG:6974",
        align_transform=[463.312716528, 0.0, -20015109.354, 0.0, -463.312716528, 10007554.677],
        patch_size=128, # Pixel dimensions in each direction
        stride=128,
        tile_coverage='intersect',
        validation_ratio=0.0, # Fraction to select as validation data
        # output_type='tiff',
        # output_path='./local_test/new_targets/',
        # sampling_region='../data/municipios/santarem_PA_BR.shp',
        output_type='tfrecord',
        output_path='gs://woodwell-aic-fire-risk/data/targets_only/',
        sampling_region='../data/Limites_RAISG_2025/Lim_Raisg.shp'
    )