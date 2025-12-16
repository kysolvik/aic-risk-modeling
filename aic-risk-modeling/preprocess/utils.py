"""
Download computed patch of Earth Engine data, plus helper functions
"""

import ee
import io
import numpy as np
import geopandas as gpd
import apache_beam as beam
import random
import tensorflow as tf
import os

from google.api_core import retry

def sample_random_points(roi: gpd.GeoDataFrame, n_sample: int, rng: np.random.Generator)->np.array:
    """Get random points within region of interest."""
    sample_df = roi.sample_points(n_sample, rng=rng).geometry.explode().get_coordinates()
    sample_df.index = np.arange(sample_df.shape[0])
    return sample_df.values

def array_to_example(structured_array):
    """Convert structured numpy array to tf.Example proto."""
    feature = {}
    for f in structured_array.dtype.names:
        feature[f] = tf.train.Feature(
            float_list = tf.train.FloatList(
                value = structured_array[f].flatten()))
    return tf.train.Example(
        features = tf.train.Features(feature = feature))

def serialize_example(structured_array):
    """Convert structured numpy array to serliazed tf.Example proto"""
    return array_to_example(structured_array).SerializeToString()

def split_dataset(element, n_partitions, validation_ratio) -> int:
    """Split dataset into training or validation data"""
    weights = [1 - validation_ratio, validation_ratio]
    return random.choices([0, 1], weights)[0]


class EEComputePatch(beam.DoFn):
    """DoFn() for computing EE patch
    
    config (dict): Dictionary containing configuration settings 
        in the following key:value pairs:
            project_id (str): Google Cloud project id
            patch_size (int): Patch size, in pixels, of output chips
            scale (float): Final scale, in m
            target_year (int): Year of prediction
            target_key (str): Name of target data, corresponds to key in self.prep_dict
            inputs_keys (list): Names of input data, correspond to keys in self.prep_dict
            proj (str): Projection, e.g. "EPSG:4326"
    """
    def __init__(self, config):
        self.config = config
        self.prep_dict = {
            'embeddings': self._prep_embeddings,
            'mcd64': self._prep_mcd64,
            'mb_fire': self._prep_mb_burned_area

        }

    def setup(self):
        print(f"Initializing Earth Engine for project: {self.config['project_id']}")
        ee.Initialize(project=self.config['project_id'], opt_url='https://earthengine-highvolume.googleapis.com')

        # Set some params
        self.proj = ee.Projection(self.config['proj']).atScale(self.config['scale'])
        self.proj_dict = self.proj.getInfo()
        self.scale_x = self.proj_dict['transform'][0]
        self.scale_y = -self.proj_dict['transform'][4]

        # Setup Earth Engine image object with all target bands
        inputs_list = [
            self.prep_dict[k](self.config['target_year']-1)
            for k in self.config['inputs_keys']
        ]
        outputs_list = [self.prep_dict[self.config['target_key']](self.config['target_year'])]
        full_list = inputs_list + outputs_list
        # Get original band names, with system indices prepended (toBands() adds)
        band_names = [
            bn 
            for image in full_list
            for bn in image.bandNames().getInfo()
        ]

        # Final prepped image
        self.prepped_image = ee.ImageCollection(inputs_list + outputs_list).toBands().rename(band_names)

    @retry.Retry()
    def process(self, coords):
        """Compute a patch of pixel, with upper-left corner defined by the coords."""

        # Make a request object.
        request = {
            'expression':self.prepped_image,
            'fileFormat': 'NPY',
            'grid': {
                'dimensions': {
                    'width': self.config['patch_size'],
                    'height':self.config['patch_size']
                },
                'affineTransform': {
                    'scaleX': self.scale_x,
                    'shearX': 0,
                    'translateX': coords[0],
                    'shearY': 0,
                    'scaleY': self.scale_y,
                    'translateY': coords[1]
                },
                'crsCode': self.config['proj'],
            },
        }

        yield np.load(io.BytesIO(ee.data.computePixels(request)))
    
    def _prep_embeddings(self, year):
        return (
            ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .mosaic()
            .setDefaultProjection(self.proj)
            .reduceResolution('mean', maxPixels=500)
            )

    def _prep_mcd64(self, year):
        return (
            ee.ImageCollection('MODIS/061/MCD64A1')
            .select('BurnDate')
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .min()
            )

    def _prep_mb_burned_area(self, year):
        return (
            ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_annual_burned_v1')
            .select(['burned_area_{}'.format(year)])
            .reduceResolution('mean', maxPixels=500)
            )

    def _prep_default(self, year):
        """Example prep method"""
        return (
            ee.ImageCollection()
            .mean()
            .reduceResolution('mean', maxPixels=500)
            )
