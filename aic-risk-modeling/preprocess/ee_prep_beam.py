"""
Beam pipeline for ingesting training data chips
"""

import argparse
import logging
import json
import os

import geopandas as gpd
import numpy as np
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow as tf

from .utils import EEComputePatch, sample_random_points, split_dataset, serialize_example

SEED = 54
RNG = np.random.default_rng(SEED)
ROI = gpd.read_file('./data/Limites_RAISG_2025/Lim_Raisg.shp')

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-json",
        required=True,
        help="JSON containing configuration dictionary.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory to save TFRecord files (local or GCS).",
    )

    # Beam args are leftover after parsing known args
    args, beam_args = parser.parse_known_args()
    
    return args, beam_args

def run():
    logging.getLogger().setLevel(logging.INFO)

    args, beam_args = parse_args()

    with open(args.config_json, 'r') as file:
        config_dict = json.loads(file.read())

    # Build beam pipeline
    sample_points  = sample_random_points(ROI, config_dict['n_sample'], RNG)

    beam_options = PipelineOptions(beam_args,
                                   project=config_dict['project_id'],
                                   region='us-east1',
                                   temp_location='gs://aic-fire-amazon/tmp/',
                                   save_main_session=True,
                                   use_public_ips=False,
                                   network='default',
                                   subnetwork='regions/us-east1/subnetworks/default',
                                   )

    with beam.Pipeline(options=beam_options) as pipeline:
        training_data, validation_data = (
            pipeline
            | 'Create points' >> beam.Create(sample_points)
            | 'Get patch' >> beam.ParDo(EEComputePatch(config_dict))
            | 'Serialize' >> beam.Map(serialize_example)
            | 'Split dataset' >> beam.Partition(split_dataset, 2, 0.2)
        )

        training_data | 'Write training data' >> beam.io.WriteToTFRecord(
            os.path.join(args.output_path, 'training'), file_name_suffix='.tfrecord.gz'
        )
        validation_data | 'Write validation data' >> beam.io.WriteToTFRecord(
            os.path.join(args.output_path, 'validation'), file_name_suffix='.tfrecord.gz'
        )



if __name__ == '__main__':
    run()