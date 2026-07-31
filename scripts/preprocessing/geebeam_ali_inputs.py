
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
PROJECT_ID = google.auth.default()[1]

parser = argparse.ArgumentParser()
parser.add_argument('--target_year', type=int, required=False, default=2024)
parser.add_argument('--random_seed', type=int, required=False, default=54)
# Beam args are leftover after parsing known args
args, other_args = parser.parse_known_args()
print(args.target_year)

RANDOM_SEED = args.random_seed
TARGET_YEAR = args.target_year

MONTH_START = 1
MONTH_END = 12
DAY_END = '31' # Set to num days in MONTH_END
MONTH_NAMES=(np.arange(MONTH_END) - MONTH_END).astype(str)

ee.Initialize(project=PROJECT_ID)

# Water deficit
def addCWD(era5LandImage):
    """Calculate CWD from single image"""
    era5LandImageCWD = (era5LandImage
        .addBands(
            (era5LandImage.select('total_precipitation_sum')
                  .subtract(era5LandImage.select('total_evaporation_sum').multiply(-1))
             ).rename('cwd')
        )
    )
    return era5LandImageCWD

# Helper function for renaming bands
def append_to_bandnames(im, append_string):
    """
    Example:
    mcd64_all = mcd64_all.map(
        lambda im: append_to_bandnames(
            im, ee.Number.parse(im.get('system:index')).subtract(ee.Number(TOTAL_MONTHS))
            )
        )
    """
    return im.rename(im.bandNames().map(lambda bn: ee.String(bn).cat(append_string)))

# VIIRS fire collection
def prep_viirs_nrt_year(y):
    # Target VIIRS Hot Spots
    viirs_path = f'projects/ksolvik-misc/assets/amazon_fire_dashboard/rasters/amazon_nrt_fire_{y}_raster'
    viirs_target = ee.Image(viirs_path)
    fireSize_band = 'b1'
    type_band = 'b2'
    conf_band = 'b3'

    return viirs_target.select(
        [fireSize_band, type_band, conf_band],
        ['fireSize', 'fire_type', 'confidence']
    ).toInt()

viirs_target = prep_viirs_nrt_year(TARGET_YEAR)

# MODIS MCD64 fire memory
def prep_mcd64_year(y):
    mcd64 = (ee.ImageCollection('MODIS/061/MCD64A1')
             .select('BurnDate')
             .filter(ee.Filter.calendarRange(y, y, 'year'))
             .max()
             .unmask()
             .gt(0)
             )
    band_names = mcd64.bandNames().getInfo()
    band_names_new = [f'{b}_{y-TARGET_YEAR}' for b in band_names]
    mcd64 = mcd64.rename(band_names_new)
    return mcd64

mcd64_list = [prep_mcd64_year(y) for y in range(TARGET_YEAR-6, TARGET_YEAR+1)]

# VIIRS fire memory
def prep_viirs_year(y):
    viirs_snpp = (ee.ImageCollection('projects/ksolvik-misc/assets/viirs_snpp_archive')
                  .filter(ee.Filter.calendarRange(y, y, 'year'))
                  ).max().unmask().gt(0).rename(f'viirs_snpp_{y-TARGET_YEAR}')
    return viirs_snpp

viirs_memory = [prep_viirs_year(y) for y in range(TARGET_YEAR-6, TARGET_YEAR+1)]

# MB Land-use/land-cover
mb_amz_lulc_im = (
    ee.Image('projects/mapbiomas-public/assets/amazon/lulc/collection6/mapbiomas_collection60_integration_v1')
    .select([f'classification_{y}' for y in range(TARGET_YEAR-6, TARGET_YEAR)])
)
mb_amz_lulc_bandnames = mb_amz_lulc_im.bandNames().getInfo()
def replace_y_with_offset(bn):
    bn_base, y = bn.split('_')
    bn_new = f'{bn_base}_{int(y)-TARGET_YEAR}'
    return bn_new
mb_amz_lulc_bandnames_new = [replace_y_with_offset(bn) for bn in mb_amz_lulc_bandnames]
mb_amz_lulc_im = mb_amz_lulc_im.rename(mb_amz_lulc_bandnames_new)

mb_amz_forest = (mb_amz_lulc_im
                 .lt(10)
                 .reduceResolution('mean', maxPixels=400)
).rename([bn.replace('classification', 'forest') for bn in mb_amz_lulc_bandnames_new])

mb_amz_pasture = (mb_amz_lulc_im
                 .eq(15)
                 .reduceResolution('mean', maxPixels=400)
).rename([bn.replace('classification', 'pasture') for bn in mb_amz_lulc_bandnames_new])

mb_amz_ag = (mb_amz_lulc_im
                 .eq(18)
                 .reduceResolution('mean', maxPixels=400)
).rename([bn.replace('classification', 'ag') for bn in mb_amz_lulc_bandnames_new])


# Deforestation
gfw_col = 'projects/glad/S2alert'
gfw_alert = (
    ee.Image(gfw_col+'/alert')
    .rename('alert').unmask(0)
    )
gfw_alert_date = (
    ee.Image(gfw_col+'/alertDate')
    .rename('alertdate').unmask(0)
    )
gfc_im = (ee.Image('UMD/hansen/global_forest_change_2025_v1_13')
          .select(['treecover2000', 'loss', 'lossyear'])
          .reduceResolution('mean', maxPixels=400)
)

# MODIS MOD13 NDVI and EVI - Monthly and annual
def prep_mod13_year(y):
    mod13 = (
        ee.ImageCollection('MODIS/061/MOD13A1')
        .select(['NDVI', 'EVI'])
        .filter(ee.Filter.calendarRange(y,
                                        y,
                                        'year'))
        .mean()
    )

    band_names = mod13.bandNames().getInfo()
    band_names_new = [f'{b}_{y-TARGET_YEAR}' for b in band_names]
    mod13 = mod13.rename(band_names_new)
    return mod13

def prep_modis13_monthly(y_start, y_end, bands):
    modmyd13 = (
        ee.ImageCollection('MODIS/061/MOD13A1').merge(
            ee.ImageCollection('MODIS/061/MYD13A1'))
        .select(bands))

    def monthly_mean(y, m):
        modmyd13_mm = (modmyd13
                .filter(ee.Filter.calendarRange(y, y, 'year'))
                .filter(ee.Filter.calendarRange(m, m, 'month'))
                .mean()
                ).set('month', m).set('year', y)
        return modmyd13_mm

    months = ee.List.sequence(MONTH_START,MONTH_END)
    years = ee.List.sequence(y_start, y_end)
    modmyd13_all = ee.ImageCollection.fromImages(
        years.map(lambda y: months.map(lambda m: monthly_mean(y,m))).flatten()
        )
    new_names = ee.List([bn + '_monthly_' + time for time, bn in itertools.product(MONTH_NAMES, bands)])
    return modmyd13_all.toBands().rename(new_names)

mod13_annual = [prep_mod13_year(y) for y in range(TARGET_YEAR-6, TARGET_YEAR)]
mod13_monthly = prep_modis13_monthly(TARGET_YEAR-1, TARGET_YEAR-1, ['NDVI','EVI'])

# Climate
def prep_era5_monthly(y_start, y_end):
    bands_ag = ['Precipitation_Flux', 'Temperature_Air_2m_Mean_24h', 'Temperature_Air_2m_Max_24h',
             'Temperature_Air_2m_Min_24h', 'Vapour_Pressure_Deficit_at_Maximum_Temperature']
    bands_land = ['total_evaporation_sum', 'total_precipitation_sum', 'cwd']
    bands = bands_ag + bands_land
    era5Ag = (ee.ImageCollection('projects/climate-engine-pro/assets/ce-ag-era5-v2/daily')
              .filter(ee.Filter.calendarRange(y_start,
                                              y_end,
                                              'year'))
              .select(
                bands_ag
            )
    )
    era5Land = (ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
                .filter(ee.Filter.calendarRange(y_start,
                                                y_end,
                                                'year'))
                .map(addCWD)
                .select(
                    bands_land
                )
    )

    def get_month(y, m):
        era5_all = ((
            era5Ag
            .filter(ee.Filter.calendarRange(y, y, 'year'))
            .filter(ee.Filter.calendarRange(m, m, 'month'))
            .mean()
            ).set('month', m).set('year', y)
            .addBands(
                era5Land
                .filter(ee.Filter.calendarRange(y, y, 'year'))
                .filter(ee.Filter.calendarRange(m, m, 'month'))
                .first()
            )
        )
        return era5_all

    months = ee.List.sequence(MONTH_START, MONTH_END)
    years = ee.List.sequence(y_start, y_end)
    era5_monthly = ee.ImageCollection.fromImages(
        years.map(lambda y: months.map(lambda m: get_month(y,m))).flatten()
        )
    new_names = ee.List([bn + '_monthly_' + time for time, bn in itertools.product(MONTH_NAMES, bands)])
    return era5_monthly.toBands().rename(new_names)
era5_im = prep_era5_monthly(TARGET_YEAR-1, TARGET_YEAR-1)

# Chirps CWD
def prep_chirps_monthly(y_start, y_end):
    chirps_cwd = (ee.ImageCollection('projects/mmacedo-reservoirid/assets/chirps_amazon_cwd')
              .filter(ee.Filter.calendarRange(y_start,
                                              y_end,
                                              'year'))
    )
    def get_month(y, m):
        chirps_filtered = ((
            chirps_cwd
            .filter(ee.Filter.calendarRange(y, y, 'year'))
            .filter(ee.Filter.calendarRange(m, m, 'month'))
            .first()
            ).set('month', m).set('year', y)
        )
        return chirps_filtered

    months = ee.List.sequence(MONTH_START, MONTH_END)
    years = ee.List.sequence(y_start, y_end)
    chirps_monthly = ee.ImageCollection.fromImages(
        years.map(lambda y: months.map(lambda m: get_month(y,m))).flatten()
        )
    new_names = ee.List(['chirps_cwd_monthly_' + time for time in MONTH_NAMES])
    return chirps_monthly.toBands().rename(new_names)

def prep_chirps_year(y):
    """Max monthly CWD within year"""
    chirps_cwd = (
        ee.ImageCollection('projects/mmacedo-reservoirid/assets/chirps_amazon_cwd')
        .filter(ee.Filter.calendarRange(y,
                                        y,
                                        'year'))
        .max()
    )

    band_names = ['chirps_cwd']
    band_names_new = [f'{b}_{y-TARGET_YEAR}' for b in band_names]
    return chirps_cwd.rename(band_names_new)

chirps_annual = [prep_chirps_year(y) for y in range(TARGET_YEAR-6, TARGET_YEAR)]
chirps_monthly = prep_chirps_monthly(TARGET_YEAR-1, TARGET_YEAR-1)


# Embeddings
def prep_embeddings_year(y):
    embeddings = (
                ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                .filter(ee.Filter.calendarRange(y,y,
                                                'year'))
                .mosaic()
                # Special step for embeddings: after mosaic, they don't have proj info
                # Setting to roughly middle of SA
                .setDefaultProjection('EPSG:32721', scale=10)
                .reduceResolution('mean', maxPixels=16, bestEffort=True)
                )
    band_names = embeddings.bandNames().getInfo()
    band_names_new = [f'{b}_{y-TARGET_YEAR}' for b in band_names]
    embeddings = embeddings.rename(band_names_new)
    return embeddings

embeddings_im = prep_embeddings_year(TARGET_YEAR-1)

# Accessibility to cities
atc_full =  ee.Image('projects/malariaatlasproject/assets/accessibility/accessibility_to_cities/2015_v1_0')
atc_im = atc_full.select('accessibility')

# World Population
landscanCol = ee.ImageCollection("projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL")
population = (
    landscanCol
    .filterDate(f'{TARGET_YEAR-1}-{MONTH_START}-01', f'{TARGET_YEAR-1}-{MONTH_END}-{DAY_END}')
    .select('b1')
    .mosaic()
    .unmask(0)
    .rename('Population_Density'))

# Night Lights
nightLightsCol = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
nightLights = (
    nightLightsCol
    .filterDate(f'{TARGET_YEAR-1}-{MONTH_START}-01', f'{TARGET_YEAR-1}-{MONTH_END}-{DAY_END}')
    .select('avg_rad')
    .mean()
    .unmask(0)
    .rename('Nighttime_Lights'))

# Topography
terrain = ee.Terrain.products(ee.Image('USGS/SRTMGL1_003'))
elevation = terrain.select('elevation').rename('Elevation')
slope = terrain.select('slope').rename('Slope')


# Protected areas
gov_types = ee.List(['Federal or national ministry or agency',
             'Sub-national ministry or agency',
             'Not Reported',
             'Collaborative governance',
             'Local communities',
             'Individual landowners',
             'Indigenous Peoples',
             'Joint governance',
             'Government-delegated management',
             'Transboundary governance',
             'Non-profit organisations',
             'For-profit organisations'])
gov_types_remap = ee.List([ee.Number(int(x)) for x in np.arange(12)+1])

wdpa_polys = ee.FeatureCollection('WCMC/WDPA/current/polygons').remap(
   gov_types, gov_types_remap, 'GOV_TYPE'
)
wdpa_im = ee.Image().int().paint(wdpa_polys, 'gov_type_numerical').rename(['gov_type'])

# Note that with split processing each will be processed separately
im_list = mcd64_list + mod13_annual + chirps_annual + viirs_memory + [
           viirs_target,
           mb_amz_pasture,
           mb_amz_forest,
           mb_amz_ag,
           gfw_alert,
           gfw_alert_date,
           mod13_monthly,
           atc_im,
           era5_im,
           embeddings_im,
           gfc_im,
           wdpa_im,
           elevation,
           slope,
           nightLights,
           population,
           chirps_monthly
]

# Get some climate indices as dict
print('Starting clim indices')
md_dict = {}
for ci in ['amo', 'tna','mei','soi','oni']:
    print(ci)
    md_dict[ci] = download_clim_indices(
        ci, year_start=TARGET_YEAR-6, year_end=TARGET_YEAR-1).values[:,0]
# Add target year as metadata
md_dict['year'] = TARGET_YEAR
print('Ending clim indices')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # Execute
    geebeam.grid_and_run_pipeline(
        image_list = im_list,
        project=PROJECT_ID,
        crs="EPSG:4326",
        align_transform=[0.005, 0.0, -85, 0.0, -0.005, 10.0],
        patch_size=128, # Pixel dimensions in each direction
        stride=128,
        tile_coverage='intersect',
        output_type='tfrecord',
        validation_ratio=0.0, # Fraction to select as validation data
        output_path=f'gs://woodwell-aic-fire-risk/data/fullgrid/allpreds_{TARGET_YEAR}',
        sampling_region='../data/Limites_RAISG_2025/Lim_Raisg.shp',
        extra_metadata=md_dict
    )
