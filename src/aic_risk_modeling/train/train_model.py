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


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import json
import inspect

from aic_risk_modeling.train import data_loader, models

SEED = 54
RNG = np.random.default_rng(SEED)


def build_datasets(
        gcs_data_dir,
        tfrecord_pattern,
        patch_size,
        output_band,
        batch_size=4,
        ):
    training_pattern = os.path.join(gcs_data_dir, 'training-{}'.format(tfrecord_pattern))
    validation_pattern = os.path.join(gcs_data_dir, 'validation-{}'.format(tfrecord_pattern))

    schema = data_loader.load_schema_from_gcs(gcs_data_dir)
    feature_spec = data_loader.build_features_dict(schema, patch_size=patch_size)

    training_ds = data_loader.dataset_from_gcs(training_pattern, feature_spec,
                                input_bands=[k for k in feature_spec.keys() if k not in ['lat','lon','id', output_band]],
                                output_bands=[output_band],
                                batch_size=batch_size,
                                shuffle_buffer=256,
                                cache=False)
    validation_ds = data_loader.dataset_from_gcs(validation_pattern, feature_spec,
                                input_bands=[k for k in feature_spec.keys() if k not in ['lat','lon','id', output_band]],
                                output_bands=[output_band],
                                batch_size=batch_size,
                                shuffle=False,
                                cache=False)

    return training_ds, validation_ds, feature_spec

def build_model(model_type, feature_spec, patch_size, output_band):
    function_name = f"get_{model_type}"  # becomes "get_unet"
    input_bands = [k for k in feature_spec.keys() if k not in ['lat','lon','id', output_band]]
    input_shape = [patch_size, patch_size, len(input_bands)]
    try:
        # Attempt to get the function dynamically
        model_fn = getattr(models, function_name)
        model = model_fn(input_shape)
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

    # Attach input layer
    inputs_dict = {
        name: tf.keras.Input(shape=(None, None, 1), name=name)
        for name in input_bands
    }
    concat = tf.keras.layers.Concatenate()(list(inputs_dict.values()))
    new_model = tf.keras.Model(inputs=inputs_dict, outputs=model(concat))
    return new_model

def run(
        gcs_data_dir,
        tfrecord_pattern,
        patch_size,
        output_band,
        batch_size,
        model_type,
        epochs,
        model_output_path,

):
    # Get datasets
    training_ds, validation_ds, feature_spec = build_datasets(
        gcs_data_dir=gcs_data_dir,
        tfrecord_pattern=tfrecord_pattern,
        patch_size=patch_size,
        output_band=output_band,
        batch_size=batch_size,
    )

    # Get model
    model = build_model(model_type.lower(), feature_spec, patch_size, output_band)

    # Compile and run
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0025),
        loss="Dice",
        metrics=[
            tf.keras.metrics.BinaryIoU(target_class_ids=[1]),
            ]
        )
    checkpoint_filepath = './checkpoint.model.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5)

    model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=[model_checkpoint_callback, early_stopping_callback]
    )

    model.save(model_output_path)

    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--gcs_data_dir', type=str, required=True)
    parser.add_argument('--tfrecord_pattern', type=str, default='*.tfrecord')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--output_band', type=str, default='BurnDate')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--model_output_path', type=str)
    args = parser.parse_args()

    run(
        model_type=args.model_type,
        gcs_data_dir=args.gcs_data_dir,
        tfrecord_pattern=args.tfrecord_pattern,
        patch_size=args.patch_size,
        output_band=args.output_band,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_output_path=args.model_output_path
    )

    print("Training complete, model saved to:", args.model_output_path)
