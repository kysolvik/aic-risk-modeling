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

from aic_risk_modeling.train import data_loader, models, losses

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

def run(
        data_dirs,
        tfrecord_pattern,
        val_data_dirs,
        val_tfrecord_pattern,
        merge_axis,
        patch_size,
        input_bands,
        output_band,
        batch_size,
        model_type,
        include_coords,
        epochs,
        steps_per_epoch,
        learning_rate,
        loss_function,
        weight_decay,
        model_output_path,
        transforms,
        stack_time_series,
        stack_inputs,
        years
):
    # Get datasets
    training_ds = data_loader.build_merged_dataset(
        data_dirs=data_dirs,
        tfrecord_pattern=tfrecord_pattern,
        shuffle=True,
        axis=merge_axis,
        patch_size=patch_size,
        batch_size=batch_size,
    )

    validation_ds = data_loader.build_merged_dataset(
        data_dirs=val_data_dirs,
        tfrecord_pattern=val_tfrecord_pattern,
        axis=merge_axis,
        shuffle=False,
        patch_size=patch_size,
        batch_size=batch_size,
    )

    # Select bands
    training_ds = data_loader.select_bands_transform(
        training_ds,
        input_bands=input_bands,
        output_bands=[output_band],
        transforms=transforms,
        stack_time_series=stack_time_series,
        stack_inputs=stack_inputs,
        years=years
    )
    validation_ds = data_loader.select_bands_transform(
        validation_ds,
        input_bands=input_bands,
        output_bands=[output_band],
        transforms=transforms,
        stack_time_series=stack_time_series,
        stack_inputs=stack_inputs,
        years=years
    )

    # Get model
    model = build_model(model_type.lower(), input_bands, include_coords,
                        patch_size, years=years)

    # Learning rate scheduler
    decay_steps = (epochs-1)*steps_per_epoch
    warmup_steps = 1*steps_per_epoch
    initial_learning_rate = 0.0
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=learning_rate,
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
        epochs=epochs,
        callbacks=[model_checkpoint_callback, early_stopping_callback, csv_logger_callback]
    )
    
    # Load best checkpoint
    model.load_weights(checkpoint_filepath)
    model.save(model_output_path)

    # Copy logged data to gs
    output_root, _ = os.path.splitext(model_output_path)
    csv_output_path = output_root + '.csv'
    upload_csv_to_gcs('./training.csv', csv_output_path)

    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path)

    # Get loss function
    loss_function = losses.get_loss(config['loss_function'])

    # Note: config.get() options are the optional ones
    run(
        model_type=config['model_type'],
        data_dirs=config['data_dirs'],
        tfrecord_pattern=config.get('tfrecord_pattern', 'train*tfrecord.gz'),
        val_data_dirs=config.get('val_data_dirs', config['data_dirs']),
        val_tfrecord_pattern=config.get('val_tfrecord_pattern', 'val*tfrecord.gz'),
        merge_axis=config.get('merge_axis', 'examples'),
        patch_size=config['patch_size'],
        input_bands=config['input_bands'],
        include_coords=config['include_coords'],
        output_band=config['output_band'],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        model_output_path=config['model_output_path'],
        transforms=config['transforms'],
        learning_rate=config['learning_rate'],
        steps_per_epoch=config.get('steps_per_epoch', 5000),
        loss_function=loss_function,
        weight_decay=config.get('weight_decay', None),
        stack_time_series=config['stack_time_series'],
        stack_inputs=config['stack_inputs'],
        years=config['years']
    )

    print("Training complete, model saved to:", config['model_output_path'])
