"""Target-year-independent 30 m layers at 463 m (run once).

Reducing 30 m data dominates the per-chip export cost and repeats for every target
year, so it is exported once here, on the same grid as geebeam_cutoff_463m.py (same
chips and md_ids). merge_static.sh then adds each year's slice to that year's export.

Bands (calendar-year names; build_static_year.py maps them to year slots):
- <class>_<year>: MapBiomas Amazonia col6 fraction per LULC_CLASSES feature
- treecover2000, lossfrac_<k>: fraction with Hansen lossyear == k
- Elevation, Slope (SRTM means), accessibility
- gov_type: WDPA protected-area governance type

    python geebeam_static_463m.py [beam/dataflow args]
"""

import logging

import ee
import google

import geebeam

DF_PROJECT_ID = google.auth.default()[1]
EE_PROJECT_ID = 'tropics-woodwell'

PIXEL_M = 463.312716528
PATCH_SIZE = 128
ALIGN_TRANSFORM = [PIXEL_M, 0.0, -20015109.354, 0.0, -PIXEL_M, 10007554.677]
# Upper bound on 30 m pixels per 463 m pixel (~17 x 17 incl. partial overlaps)
MAX_FINE_PIXELS = 1024

MAPBIOMAS_AMAZONIA = ('projects/mapbiomas-public/assets/amazon/lulc/collection6/'
                      'mapbiomas_collection60_integration_v1')
# Slot -10 of the 2013 target (Y-13) through the col6 end year
MAPBIOMAS_YEARS = range(2000, 2024)
# col6 legend codes -> features: exactly the full export's definitions (forest = codes < 10)
LULC_CLASSES = {
    'forest': [3, 4, 5, 6, 9],
    'pasture': [15],
    'ag': [18],
    'urban': [24],
    'mining': [30],
    'water': [33],
}
HANSEN = 'UMD/hansen/global_forest_change_2025_v1_13'
HANSEN_MAX_LOSSYEAR = 25  # v1.13 holds loss through 2025

ee.Initialize(project=EE_PROJECT_ID)


def mean_463(img):
    return img.reduceResolution(ee.Reducer.mean(), maxPixels=MAX_FINE_PIXELS)


def mapbiomas_fractions(year):
    classification = ee.Image(MAPBIOMAS_AMAZONIA).select(f'classification_{year}')
    return mean_463(ee.Image.cat([
        classification.remap(codes, [1] * len(codes), 0).rename(f'{name}_{year}')
        for name, codes in LULC_CLASSES.items()]))


def hansen_image():
    gfc = ee.Image(HANSEN)
    # Masked where there was no loss; unmask so the means are fractions of all pixels
    lossyear = gfc.select('lossyear').unmask(0)
    return mean_463(ee.Image.cat(
        [gfc.select('treecover2000')]
        + [lossyear.eq(k).rename(f'lossfrac_{k:02d}') for k in range(1, HANSEN_MAX_LOSSYEAR + 1)]))


def terrain_access_image():
    terrain = ee.Terrain.products(ee.Image('USGS/SRTMGL1_003'))
    return ee.Image.cat([
        mean_463(terrain.select('elevation')).rename('Elevation').unmask(0),
        mean_463(terrain.select('slope')).rename('Slope'),
        ee.Image('projects/malariaatlasproject/assets/accessibility/accessibility_to_cities/2015_v1_0')
        .select('accessibility').unmask(5000),
    ])


# WDPA governance type (1-12, masked = 0 outside protected areas); the current
# snapshot, as in earlier exports
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
gov_types_remap = ee.List(list(range(1, 13)))
wdpa_polys = ee.FeatureCollection('WCMC/WDPA/current/polygons').remap(
    gov_types, gov_types_remap, 'GOV_TYPE')
wdpa_im = ee.Image().int().paint(wdpa_polys, 'GOV_TYPE').rename(['gov_type'])


im_list = [mapbiomas_fractions(y) for y in MAPBIOMAS_YEARS] + [hansen_image(), terrain_access_image(), wdpa_im]


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    geebeam.grid_and_run_pipeline(
        image_list=im_list,
        project=EE_PROJECT_ID,
        dataflow_project=DF_PROJECT_ID,
        crs="SR-ORG:6974",
        align_transform=ALIGN_TRANSFORM,
        patch_size=PATCH_SIZE,
        stride=PATCH_SIZE,
        tile_coverage='intersect',
        validation_ratio=0.0,
        output_type='tfrecord',
        output_path='gs://woodwell-aic-fire-risk/data/fullgrid_v5/static',
        sampling_region='../data/Limites_RAISG_2025/Lim_Raisg.shp',
#         output_type='tiff',
#         output_path='./local_test/static/',
#         sampling_region='../data/municipios/santarem_PA_BR.shp',
    )
