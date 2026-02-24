"""Entry point for training risk model on Vertex AI.

This module provides functionality to train a segmentation model for predicting burned areas.
It handles data loading from GCS, model building based on specified architecture, and training
with checkpointing and early stopping.

Command-line Arguments:
    --model_type (str, required): Type of model architecture to use (e.g., 'unet').
    --gcs_data_dir (str, required): GCS path to directory containing training/validation data.
    --tfrecord_pattern (str, optional): Pattern for TFRecord files. Default: '*.tfrecord'.
    --patch_size (int, optional): Spatial dimensions of input patches. Default: 128.
    --output_band (str, optional): Name of target output band. Default: 'BurnDate'.
    --batch_size (int, optional): Batch size for training. Default: 4.
    --epochs (int, required): Number of training epochs.
    --model_output_path (str, optional): Path to save trained model, can be cloud storage.
"""


import numpy as np
import tensorflow as tf
import keras
import json
import inspect

from aic_risk_modeling.train import data_loader, models

SEED = 54
RNG = np.random.default_rng(SEED)

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


def clean_class_weight(cw_dict):
    return {int(k): v for k, v in cw_dict.items()}


def build_model(model_type, input_bands, include_coords, patch_size, years):
    function_name = f"get_{model_type}"
    # NOTE: ONLY ALLOWS FOR IMAGE TIME SERIES INPUTS
    image_shape = [len(years), patch_size, patch_size, len(input_bands)]
    try:
        # Attempt to get the function dynamically
        model_fn = getattr(models, function_name)
        model = model_fn(image_shape, include_metadata=include_coords, metadata_shape=(2,))
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
        learning_rate,
        loss_function,
        class_weight,
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

    # Compile and run
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
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
        patience=5)

    model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[model_checkpoint_callback, early_stopping_callback]
    )

    model.save(model_output_path)

    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path)

    # Clean class weight
    class_weight = config.get('class_weight')
    if class_weight is not None:
        class_weight = clean_class_weight(class_weight)

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
        loss_function=config['loss_function'],
        class_weight=class_weight,
        stack_time_series=config['stack_time_series'],
        stack_inputs=config['stack_inputs'],
        years=config['years']
    )

    print("Training complete, model saved to:", config['model_output_path'])
