"""Operational 463 m export: inputs limited to what is published by the issue date.

The forecast for TARGET_YEAR is issued Oct 31 of TARGET_YEAR-1. Every input is cut
at its own source's cutoff from aic_risk_modeling.preprocess.cutoff (set from
scripts/preprocessing/audit_source_latency.py), and the SAME cuts are applied when
exporting training years, so the model trains on exactly the information it gets at
issue time. Labels (`*_0` bands) are the full calendar TARGET_YEAR.

Differences from the 463 m full-year export (geebeam_ali_inputs.py):
- Grid: the v3 grid (MODIS sinusoidal 463.312716528 m, 128-px chips).
- Monthly features: 12-month window ending at each source's last available month
  (names -12 ... -1 relative to that end). Annual features: calendar years, the
  last input year cut at the source's cutoff. Annual-only products use the newest
  year available at issue (cutoff.ANNUAL_LAG) under the -1 name.
- MapBiomas (Amazonia col6, newest year Y-4), Hansen, SRTM, accessibility and WDPA are NOT
  exported here: geebeam_static_463m.py exports them once (same chips and md_ids) and
  merge_static.sh adds them per year.
- Climate indices: CPC SOI, Nino 3.4, Nino 4 (replace PSL SOI/ONI), plus AMO/MEI/TNA,
  each the 120 months ending at its own last available month.
- Predict-only label placeholders are zeros (no dependency on unpublished labels).
- The legacy fire-atlas target bands (fireSize/fire_type/confidence) are dropped.

    python geebeam_cutoff_463m.py --target_year 2025 [beam/dataflow args]
    python geebeam_cutoff_463m.py --target_year 2027 --predict_only [beam/dataflow args]
"""

import argparse
import datetime
import itertools
import logging

from aic_risk_modeling.preprocess import cutoff, download_clim_indices
import ee
import google
import numpy as np

import geebeam

# Get default project id from environment (or specify PROJECT_ID manually)
DF_PROJECT_ID = google.auth.default()[1]
EE_PROJECT_ID = 'tropics-woodwell'

parser = argparse.ArgumentParser()
parser.add_argument('--target_year', type=int, required=True)
parser.add_argument('--random_seed', type=int, required=False, default=54)
parser.add_argument('--predict_only', action='store_true')
# Beam args are leftover after parsing known args
args, other_args = parser.parse_known_args()

RANDOM_SEED = args.random_seed
TARGET_YEAR = args.target_year
PREDICT_ONLY = args.predict_only
LAST_YEAR = TARGET_YEAR - 1

# Grid: v3 (and geebeam_static_463m.py)
PIXEL_M = 463.312716528
PATCH_SIZE = 128
ALIGN_TRANSFORM = [PIXEL_M, 0.0, -20015109.354, 0.0, -PIXEL_M, 10007554.677]

N_MONTHS = 12
MONTH_NAMES = (np.arange(N_MONTHS) - N_MONTHS).astype(str)
N_ANNUAL = 10            # annual lags -10 ... -1
N_INDEX_MONTHS = 120
MODIS_COMPOSITE_DAYS = 16

ee.Initialize(project=EE_PROJECT_ID)


def year_filter(y_start, y_end=None):
    return ee.Filter.calendarRange(y_start, y_start if y_end is None else y_end, 'year')


def month_filter(m_start, m_end=None):
    return ee.Filter.calendarRange(m_start, m_start if m_end is None else m_end, 'month')


def placeholder(name):
    """Predict-only stand-in for a label band; never score a predict-only export."""
    return ee.Image.constant(0).rename(name)


def stack_months(images, bands):
    """Monthly images (oldest first) -> one image with bands <band>_monthly_<-12..-1>."""
    names = ee.List([bn + '_monthly_' + t for t, bn in itertools.product(MONTH_NAMES, bands)])
    return ee.ImageCollection.fromImages(images).toBands().rename(names)


def print_cutoffs():
    print(f'Target {TARGET_YEAR}, issue {cutoff.issue_date(TARGET_YEAR)}'
          f'{" (predict only)" if PREDICT_ONLY else ""}')
    for source in cutoff.LAST_MONTH:
        print(f'  {source:<14} through {cutoff.cutoff_date(source, TARGET_YEAR)}')
    for source in cutoff.ANNUAL_LAG:
        print(f'  {source:<20} year {cutoff.annual_year(source, TARGET_YEAR)}')


# ----------------------------------------------------------------------------- fire
# source -> (asset, band to select or None, output band prefix)
FIRE_SOURCES = {
    'mcd64a1': ('MODIS/061/MCD64A1', 'BurnDate', 'BurnDate'),
    'vnp64a1': ('NASA/VIIRS/002/VNP64A1', 'Burn_Date', 'BurnDate_viirs'),
    'viirs_snpp': ('projects/mmacedo-reservoirid/assets/viirs_snpp_archive_msgrid', None, 'viirs_snpp'),
    'viirs_noaa20': ('projects/mmacedo-reservoirid/assets/viirs_noaa20_archive_msgrid', None, 'viirs_noaa20'),
    'mod14': ('projects/mmacedo-reservoirid/assets/mod14_archive_msgrid', None, 'mod14'),
    'mod14_aqua': ('projects/mmacedo-reservoirid/assets/mod14_aqua_archive_msgrid', None, 'aqua'),
    'mod14_terra': ('projects/mmacedo-reservoirid/assets/mod14_terra_archive_msgrid', None, 'terra'),
}
# Input lags exported per source (as in the full export); every source also gets a `_0` label
FIRE_INPUT_LAGS = {'mcd64a1': 10, 'vnp64a1': 1, 'viirs_snpp': 1, 'viirs_noaa20': 0,
                   'mod14': 10, 'mod14_aqua': 10, 'mod14_terra': 10}
FIRE_START_YEAR = {'viirs_noaa20': 2018}  # NOAA20 archive start; no label band before it


def burn_year(source, y):
    """Annual burn-date (DOY) image of `source` for year `y`.

    Last input year: cells burned after the source's cutoff DOY are zeroed per image,
    before any max, so a later burn can't carry a post-cutoff date into the inputs.
    (The FIRMS archives hold the EARLIEST DOY per cell, so this cut is exact there.)
    """
    asset, band, _ = FIRE_SOURCES[source]
    col = ee.ImageCollection(asset)
    if band is not None:
        col = col.select(band)
    col = col.filter(year_filter(y))
    if y == LAST_YEAR:
        doy = cutoff.cutoff_doy(source, TARGET_YEAR)
        col = col.map(lambda im: im.where(im.gt(doy), 0))
    return col.max().unmask()


fire_bands = []
for source, (_, _, prefix) in FIRE_SOURCES.items():
    for k in range(FIRE_INPUT_LAGS[source], 0, -1):
        fire_bands.append(burn_year(source, TARGET_YEAR - k).rename(f'{prefix}_{-k}'))
    if PREDICT_ONLY:
        fire_bands.append(placeholder(f'{prefix}_0'))
    elif TARGET_YEAR >= FIRE_START_YEAR.get(source, 0):
        fire_bands.append(burn_year(source, TARGET_YEAR).rename(f'{prefix}_0'))


# ----------------------------------------------------------------------------- MOD13 NDVI/EVI
MOD13_BANDS = ['NDVI', 'EVI']


def complete_composites(col):
    """Composites whose whole 16-day window ends on or before the mod13 cutoff."""
    last_day = ee.Date(cutoff.cutoff_date('mod13', TARGET_YEAR).isoformat())
    # start + 15 days <= cutoff  <=>  start < cutoff - 14 days
    return col.filterDate('2000-01-01', last_day.advance(-(MODIS_COMPOSITE_DAYS - 2), 'day'))


def prep_mod13_year(y):
    """Annual mean of Terra MOD13A1 composites; the last input year only up to the cutoff."""
    col = ee.ImageCollection('MODIS/061/MOD13A1').select(MOD13_BANDS).filter(year_filter(y))
    if y == LAST_YEAR:
        col = complete_composites(col)
    return col.mean().rename([f'{b}_{y - TARGET_YEAR}' for b in MOD13_BANDS])


def prep_mod13_monthly():
    col = complete_composites(
        ee.ImageCollection('MODIS/061/MOD13A1')
        .merge(ee.ImageCollection('MODIS/061/MYD13A1'))
        .select(MOD13_BANDS))
    months = [col.filter(year_filter(y)).filter(month_filter(m)).mean()
              for y, m in cutoff.month_window('mod13', TARGET_YEAR, N_MONTHS)]
    return stack_months(months, MOD13_BANDS)


mod13_annual = [prep_mod13_year(y) for y in range(TARGET_YEAR - N_ANNUAL, TARGET_YEAR)]
mod13_monthly = prep_mod13_monthly()


# ----------------------------------------------------------------------------- weather
AGERA5_BANDS = ['Precipitation_Flux', 'Temperature_Air_2m_Mean_24h', 'Temperature_Air_2m_Max_24h',
                'Temperature_Air_2m_Min_24h', 'Vapour_Pressure_Deficit_at_Maximum_Temperature']
ERA5_LAND_BANDS = ['total_evaporation_sum', 'total_precipitation_sum', 'cwd']


def add_cwd(era5_land_image):
    """Water deficit = precipitation + evaporation (ERA5 evaporation is negative)."""
    return era5_land_image.addBands(
        era5_land_image.select('total_precipitation_sum')
        .subtract(era5_land_image.select('total_evaporation_sum').multiply(-1))
        .rename('cwd'))


def prep_agera5_monthly():
    window = cutoff.month_window('agera5', TARGET_YEAR, N_MONTHS)
    col = (ee.ImageCollection('projects/climate-engine-pro/assets/ce-ag-era5-v2/daily')
           .filter(year_filter(window[0][0], window[-1][0]))
           .select(AGERA5_BANDS))
    return stack_months([col.filter(year_filter(y)).filter(month_filter(m)).mean()
                         for y, m in window], AGERA5_BANDS)


def prep_era5_land_monthly():
    window = cutoff.month_window('era5_land', TARGET_YEAR, N_MONTHS)
    col = (ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
           .filter(year_filter(window[0][0], window[-1][0]))
           .map(add_cwd)
           .select(ERA5_LAND_BANDS))
    return stack_months([col.filter(year_filter(y)).filter(month_filter(m)).first()
                         for y, m in window], ERA5_LAND_BANDS)


agera5_im = prep_agera5_monthly()
era5_land_im = prep_era5_land_monthly()


# ----------------------------------------------------------------------------- CHIRPS CWD
# Monthly assets maintained by hand (Macedo); the audit confirms they reach the cutoff
CHIRPS_AMZ = 'projects/mmacedo-reservoirid/assets/chirps_amazon_cwd'
CHIRPS_CRD = 'projects/mmacedo-reservoirid/assets/chirps_cerrado_cwd'


def prep_chirps_monthly():
    """(merged, amazon_only) monthly CWD over the chirps_cwd window."""
    window = cutoff.month_window('chirps_cwd', TARGET_YEAR, N_MONTHS)
    years = year_filter(window[0][0], window[-1][0])
    amz_col = ee.ImageCollection(CHIRPS_AMZ).filter(years)
    crd_col = ee.ImageCollection(CHIRPS_CRD).filter(years)
    merged_images, amz_images = [], []
    for y, m in window:
        amz = amz_col.filter(year_filter(y)).filter(month_filter(m)).first()
        crd = crd_col.filter(year_filter(y)).filter(month_filter(m)).first()
        merged_images.append(ee.ImageCollection([amz, crd]).mosaic().unmask())
        amz_images.append(ee.Image(amz).unmask())
    merged = ee.ImageCollection(merged_images).toBands().rename(
        ee.List(['chirps_cwd_monthly_' + t for t in MONTH_NAMES]))
    amz_only = ee.ImageCollection(amz_images).toBands().rename(
        ee.List(['chirps_cwd_amz_monthly_' + t for t in MONTH_NAMES]))
    return merged, amz_only


def prep_chirps_year(y):
    """Max monthly deficit (min CWD) within year; the last input year up to the cutoff."""
    last_month = cutoff.last_month('chirps_cwd') if y == LAST_YEAR else 12
    amz = ee.ImageCollection(CHIRPS_AMZ).filter(year_filter(y)).filter(month_filter(1, last_month)).min()
    crd = ee.ImageCollection(CHIRPS_CRD).filter(year_filter(y)).filter(month_filter(1, last_month)).min()
    merged = ee.ImageCollection([amz, crd]).mosaic().unmask().rename(f'chirps_cwd_{y - TARGET_YEAR}')
    return merged, amz.unmask().rename(f'chirps_cwd_amz_{y - TARGET_YEAR}')


chirps_annual_pairs = [prep_chirps_year(y) for y in range(TARGET_YEAR - N_ANNUAL, TARGET_YEAR)]
chirps_annual = [pair[0] for pair in chirps_annual_pairs]
chirps_annual_amz = [pair[1] for pair in chirps_annual_pairs]
chirps_monthly, chirps_monthly_amz = prep_chirps_monthly()


# ----------------------------------------------------------------------------- other annual
# Nightlights: slots -2, -1 = the two newest usable years
NIGHTLIGHTS_START_YEAR = 2013
NIGHTLIGHTS_YEAR = cutoff.annual_year('nightlights', TARGET_YEAR)


def prep_nightlights(y, slot):
    col = (ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V21')
           .merge(ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V22')))
    lights = (col.filter(year_filter(max(y, NIGHTLIGHTS_START_YEAR))).first()
              .select(['median_masked', 'maximum', 'cf_cvg']).unmask(0))
    return lights.rename([f'{b}_{-slot}' for b in ['median_masked', 'maximum', 'cf_cvg']])


nightlights_list = [prep_nightlights(NIGHTLIGHTS_YEAR - (slot - 1), slot) for slot in (2, 1)]

# LandScan population: newest usable year
population = (ee.ImageCollection('projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL')
              .filter(year_filter(cutoff.annual_year('landscan', TARGET_YEAR)))
              .select('b1').mosaic().unmask(0).rename('Population_Density'))


# ----------------------------------------------------------------------------- assemble
im_list = (fire_bands + mod13_annual + chirps_annual + chirps_annual_amz
           + nightlights_list + [
               mod13_monthly,
               agera5_im,
               era5_land_im,
               chirps_monthly,
               chirps_monthly_amz,
               population,
           ])

# ----------------------------------------------------------------------------- climate indices
CLIMATE_INDICES = ['soi_cpc', 'nino34', 'nino4', 'amo', 'mei', 'tna']

md_dict = {}
for ci in CLIMATE_INDICES:
    # The 120 months ending at the index's own last available month (callers index
    # positionally, so the length is fixed)
    window = cutoff.month_window(ci, TARGET_YEAR, N_INDEX_MONTHS)
    (y0, _), (y1, m1) = window[0], window[-1]
    vals = download_clim_indices(ci, year_start=y0, year_end=y1, last_month=m1).values[:, 0]
    md_dict[ci] = vals[-N_INDEX_MONTHS:]
    assert len(md_dict[ci]) == N_INDEX_MONTHS, (ci, len(md_dict[ci]))
md_dict['year'] = TARGET_YEAR


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    print_cutoffs()
    geebeam.grid_and_run_pipeline(
        image_list=im_list,
        project=EE_PROJECT_ID,
        dataflow_project=DF_PROJECT_ID,
        crs="SR-ORG:6974",
        align_transform=ALIGN_TRANSFORM,
        patch_size=PATCH_SIZE,  # Pixel dimensions in each direction
        stride=PATCH_SIZE,
        tile_coverage='intersect',
        validation_ratio=0.0,  # Fraction to select as validation data
        output_type='tfrecord',
        output_path=f'gs://woodwell-aic-fire-risk/data/fullgrid_v5/dynamic/allpreds_{TARGET_YEAR}',
        sampling_region='../data/Limites_RAISG_2025/Lim_Raisg.shp',
#         output_type='tiff',
#         output_path='./local_test/dynamic/',
#         sampling_region='../data/municipios/santarem_PA_BR.shp',
        extra_metadata=md_dict
    )
