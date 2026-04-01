"""Entry point for training risk model on Vertex AI.

This module provides functionality to train a segmentation model for predicting burned areas.
It handles data loading from GCS, model building based on specified architecture, and training
with checkpointing and early stopping.
"""

import os
import json
import inspect

import numpy as np
import tensorflow as tf
import keras
from google.cloud import storage
from urllib.parse import urlparse

from aic_risk_modeling.train import data_loader, models, losses, data_norm

SEED = 54
RNG = np.random.default_rng(SEED)


def upload_csv_to_gcs(local_path, gcs_uri):
    """
    Upload a CSV file to Google Cloud Storage.

    Args:
        local_path (str): Path to local CSV file.
        gcs_uri (str): Full GCS URI (e.g. "gs://my-bucket/path/to/file.csv")
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with 'gs://'")

    # Parse bucket and blob path
    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    blob_path = parsed.path.lstrip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(local_path)

    print(f"Uploaded {local_path} to {gcs_uri}")


def load_config(
        config_path
    ):
    if config_path.startswith('gs://'):
        with tf.io.gfile.GFile(config_path, 'r') as f:
            config = json.load(f)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    return config


def _gcs_join(base: str, name: str) -> str:
    return base.rstrip("/") + "/" + name

def build_decoder(decoder_type, branch_models):
    function_name = f"decoder_{decoder_type}"
    try:
        # Attempt to get the function dynamically
        model_fn = getattr(models, function_name)
        model = model_fn(branch_models)
        print(f"Successfully initialized {decoder_type} model.")

    except AttributeError:
        # 1. Get all members of the 'model' module
        # 2. Filter for things that are functions AND start with 'get_'
        available_funcs = [
            name for name, obj in inspect.getmembers(model, inspect.isfunction)
            if name.startswith("decoder_")
        ]

        # 3. Clean up the names for the error message (e.g., 'get_unet' -> 'unet')
        valid_options = [n.replace("decoder_", "") for n in available_funcs]

        raise ValueError(
            f"Invalid model type '{decoder_type}'. \n"
            f"Expected one of: {valid_options}\n"
            f"Note: The script looks for functions named 'get_<type>' in model.py"
        )

    return model


def build_model(model_type, input_shape, input_name):
    function_name = f"get_{model_type}"
    try:
        # Attempt to get the function dynamically
        model_fn = getattr(models, function_name)
        model = model_fn(input_shape=input_shape, input_name=input_name)
        print(f"Successfully initialized {model_type} model.")

    except AttributeError:
        # 1. Get all members of the 'model' module
        # 2. Filter for things that are functions AND start with 'get_'
        available_funcs = [
            name for name, obj in inspect.getmembers(model, inspect.isfunction)
            if name.startswith("get_")
        ]

        # 3. Clean up the names for the error message (e.g., 'get_unet' -> 'unet')
        valid_options = [n.replace("get_", "") for n in available_funcs]

        raise ValueError(
            f"Invalid model type '{model_type}'. \n"
            f"Expected one of: {valid_options}\n"
            f"Note: The script looks for functions named 'get_<type>' in model.py"
        )

    return model

def build_all_models(inputs_config):
    all_models = []
    for input_key, input_dict in inputs_config.items():
        model_type = input_dict['model_type']
        if len(input_dict['timesteps']) > 0:
            time_dim = [len(input_dict['timesteps'])]
        else:
            time_dim = []
        feature_dim = [len(input_dict['feature_names'])]
        input_shape = time_dim + input_dict['shape'] + feature_dim
        print(input_shape)
        input_name = input_key
        all_models.append(build_model(model_type, input_shape, input_name))

    return all_models

def run(config):
    # Some options that have defaults
    steps_per_epoch=config.get('steps_per_epoch', 5000)
    weight_decay=config.get('weight_decay', None)

    # Get loss function
    loss_function = losses.get_loss(config['loss_function'])

    # Get datasets
    training_ds = data_loader.build_merged_dataset(
        data_dirs=config['data_dirs'],
        tfrecord_pattern=config['tfrecord_pattern'],
        shuffle=True,
        axis=config['merge_axis'],
        batch_size=config['batch_size'],
    )
    validation_ds = data_loader.build_merged_dataset(
        data_dirs=config['val_data_dirs'],
        tfrecord_pattern=config['tfrecord_pattern'],
        shuffle=False,
        axis=config['merge_axis'],
        batch_size=config['batch_size'],
    )

    # Normalize (uses first dir, hope that's representative-ish!)
    normalize_list = data_norm.get_normalize_list(config)
    norm_func = data_norm.create_normalizer(_gcs_join(config['data_dirs'][0], 'stats.pbtxt'), normalize_list)
    training_ds = training_ds.map(norm_func)
    validation_ds = validation_ds.map(norm_func)

    # Select bands
    # Select bands
    training_ds = data_loader.select_bands_transform(
        training_ds,
        input_feature_config=config['input_features'],
        output_feature_config=config['output_features'],

    )
    validation_ds = data_loader.select_bands_transform(
        validation_ds,
        input_feature_config=config['input_features'],
        output_feature_config=config['output_features'],
    )

    # Get model
    all_models = build_all_models(config['input_features'])

    # Build decoder (note: can build an identity decoder, if desired)
    model = build_decoder(config['decoder'], all_models)

    # Learning rate scheduler
    decay_steps = (config['epochs']-1)*steps_per_epoch
    warmup_steps = 1*steps_per_epoch
    initial_learning_rate = 0.0
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=config['learning_rate'],
        warmup_steps=warmup_steps
    )
    # Compile and run
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule,
                                         weight_decay=weight_decay),
        loss=loss_function,
        metrics=[
            keras.metrics.BinaryIoU(target_class_ids=[1]),
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc")
            ]
        )
    checkpoint_filepath = './checkpoint.model.keras'

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_pr_auc',
        mode='max',
        save_best_only=True)

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_pr_auc',
        mode='max',
        patience=8)

    csv_logger_callback = keras.callbacks.CSVLogger(
        './training.csv'
    )
    model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=config['epochs'],
        callbacks=[model_checkpoint_callback, early_stopping_callback, csv_logger_callback]
    )

    # Load best checkpoint
    model.load_weights(checkpoint_filepath)
    model.save(config['model_output_path'])

    # Copy logged data to gs
    output_root, _ = os.path.splitext(config['model_output_path'])
    csv_output_path = output_root + '.csv'
    upload_csv_to_gcs('./training.csv', csv_output_path)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    # Note: should add a schema verifier
    config = load_config(args.config_path)

    run(config)

    print("Training complete, model saved to:", config['model_output_path'])
