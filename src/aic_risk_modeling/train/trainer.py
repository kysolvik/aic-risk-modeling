"""Entry point for training risk model on Vertex AI.

This module provides functionality to train a segmentation model for predicting burned areas.
Models and the training loop are PyTorch; data loading runs via tf.data TFRecord
pipeline in `data_loader`.
Training includes checkpointing and early stopping on configurable validation
metrics ('checkpoint_metric' / 'early_stopping_metric'); 'loss' is minimized,
every other metric is maximized.
"""

import csv
import inspect
import json
import math
import os
import tempfile

import numpy as np
import tensorflow as tf
import torch
from google.cloud import storage
from urllib.parse import urlparse

from aic_risk_modeling.train import data_loader, models, losses, data_norm
from aic_risk_modeling.train.metrics import (
    SegmentationMetrics, MulticlassSegmentationMetrics)

# TensorFlow is only used for data loading; keep it off the GPU.
tf.config.set_visible_devices([], "GPU")

SEED = 54
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)


def upload_file_to_gcs(local_path, gcs_uri):
    """
    Upload a local file to Google Cloud Storage.

    Args:
        local_path (str): Path to local file.
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

def build_decoder(decoder_type, branch_models, decoder_config=None,
                  num_classes=1):
    function_name = f"decoder_{decoder_type}"
    try:
        # Attempt to get the function dynamically
        model_fn = getattr(models, function_name)
        model = model_fn(branch_models, num_classes=num_classes,
                         **(decoder_config or {}))
        print(f"Successfully initialized {decoder_type} model.")

    except AttributeError:
        available_funcs = [
            name for name, obj in inspect.getmembers(models, inspect.isfunction)
            if name.startswith("decoder_")
        ]
        valid_options = [n.replace("decoder_", "") for n in available_funcs]

        raise ValueError(
            f"Invalid decoder type '{decoder_type}'. \n"
            f"Expected one of: {valid_options}\n"
            f"Note: The script looks for functions named 'decoder_<type>' in models.py"
        )

    return model


def build_model(model_type, input_shape, input_name, **model_kwargs):
    function_name = f"get_{model_type}"
    try:
        # Attempt to get the function dynamically
        model_fn = getattr(models, function_name)
        model = model_fn(input_shape=input_shape, input_name=input_name,
                         **model_kwargs)
        print(f"Successfully initialized {model_type} model.")

    except AttributeError:
        available_funcs = [
            name for name, obj in inspect.getmembers(models, inspect.isfunction)
            if name.startswith("get_")
        ]
        valid_options = [n.replace("get_", "") for n in available_funcs]

        raise ValueError(
            f"Invalid model type '{model_type}'. \n"
            f"Expected one of: {valid_options}\n"
            f"Note: The script looks for functions named 'get_<type>' in models.py"
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
        # Optional per-branch factory kwargs (e.g. a projection branch's
        # out_channels) from the config's input_features group.
        model_kwargs = input_dict.get('model_kwargs') or {}
        all_models.append(
            build_model(model_type, input_shape, input_name, **model_kwargs))

    return all_models


def save_model(model, config, output_path):
    """Save model weights (with the config needed to rebuild it); supports gs:// paths."""
    payload = {'config': config, 'model_state_dict': model.state_dict()}
    if output_path.startswith('gs://'):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(output_path))
            torch.save(payload, local_path)
            upload_file_to_gcs(local_path, output_path)
    else:
        torch.save(payload, output_path)


def load_model(model_path, map_location='cpu'):
    """Rebuild a model from a checkpoint saved by `save_model`/`run`."""
    if model_path.startswith('gs://'):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(model_path))
            tf.io.gfile.copy(model_path, local_path)
            checkpoint = torch.load(local_path, map_location=map_location)
    else:
        checkpoint = torch.load(model_path, map_location=map_location)

    config = checkpoint['config']
    model = build_decoder(config['decoder'],
                          build_all_models(config['input_features']),
                          config.get('decoder_config'),
                          num_classes=config.get('num_classes') or 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def _torch_batches(dataset, device):
    """Yield (inputs, labels, sample_weight) from a tf.data dataset as torch tensors.

    sample_weight is None when the dataset yields plain (inputs, labels)
    2-tuples (i.e. no per-pixel weighting configured)."""
    for batch in dataset.as_numpy_iterator():
        if len(batch) == 3:
            inputs, labels, weights = batch
            weights = torch.as_tensor(weights).float().to(device)
        else:
            inputs, labels = batch
            weights = None
        inputs = {k: torch.as_tensor(v).to(device) for k, v in inputs.items()}
        labels = torch.as_tensor(labels).float().to(device)
        yield inputs, labels, weights


def _cosine_warmup_schedule(optimizer, warmup_steps, decay_steps):
    """Per-step linear warmup from 0, then cosine decay to 0
    (keras CosineDecay with warmup_target equivalent)."""
    def factor(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = min(step - warmup_steps, decay_steps) / max(decay_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _run_epoch(model, dataset, loss_function, device, metrics,
               optimizer=None, scaler=None, scheduler=None, log_every=500):
    """One pass over `dataset`; trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    metrics.reset()
    total_loss = 0.0
    num_batches = 0
    amp_enabled = device.type == 'cuda'
    amp_dtype = torch.float16 if amp_enabled else torch.bfloat16

    with torch.set_grad_enabled(training):
        for inputs, labels, weights in _torch_batches(dataset, device):
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                preds = model(inputs)
            # preds are float32 (the fusion head opts out of autocast)
            loss = loss_function(labels, preds, weights)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            metrics.update(labels, preds)
            total_loss += loss.item()
            num_batches += 1
            if training and num_batches % log_every == 0:
                print(f"  step {num_batches}: loss={total_loss / num_batches:.4f}",
                      flush=True)

    results = metrics.compute()
    results['loss'] = total_loss / max(num_batches, 1)
    return results


class _BestTracker:
    """Tracks whether a monitored validation metric has improved.

    `metric` names a key of the per-epoch validation results; 'loss' is
    minimized, every other metric is maximized. Ties are not improvements.
    """

    def __init__(self, metric):
        self.metric = metric
        self._sign = -1.0 if metric == 'loss' else 1.0
        self._best = float('-inf')

    def improved(self, results):
        value = self._sign * results[self.metric]
        if value > self._best:
            self._best = value
            return True
        return False


def run(config):
    # Some options that have defaults
    steps_per_epoch = config.get('steps_per_epoch', 5000)
    weight_decay = config.get('weight_decay', 0.01)
    patience = config.get('early_stopping_patience', 4)
    # num_classes == 1 is binary segmentation; >1 is multi-class (the output
    # feature is left as raw integer class labels, e.g. viirs_type 0-4).
    num_classes = config.get('num_classes') or 1
    # Positive-class weight for the binary weighted losses; also the default
    # weight for fire pixels of types not explicitly listed in 'sample_weight'.
    pos_weight = config.get('pos_weight', 9.0)
    # Aggregate burn-area term options, used by 'weighted_bce_area'.
    area_weight = config.get('area_loss_weight', 1.0)
    area_block_size = config.get('area_block_size')

    # Get loss function
    loss_function = losses.get_loss(config['loss_function'],
                                    num_classes=num_classes,
                                    class_weights=config.get('class_weights'),
                                    pos_weight=pos_weight,
                                    area_weight=area_weight,
                                    area_block_size=area_block_size)

    # Get datasets
    training_ds = data_loader.build_merged_dataset(
        data_dirs=config['data_dirs'],
        tfrecord_pattern=config['tfrecord_pattern'],
        shuffle=True,
        rename_dict=config.get('rename_dict', None),
        axis=config['merge_axis'],
        batch_size=config['batch_size'],
        seed=SEED,
    )
    validation_ds = data_loader.build_merged_dataset(
        data_dirs=config['val_data_dirs'],
        tfrecord_pattern=config['tfrecord_pattern'],
        shuffle=False,
        rename_dict=config.get('rename_dict', None),
        axis=config['merge_axis'],
        batch_size=config['batch_size'],
        seed=SEED,
    )

    # Normalize. Prefer an explicit stats file (e.g. pooled stats.json from
    # data_stats); fall back to the first dir's stats.pbtxt.
    stats_path = config.get(
        'stats_path', _gcs_join(config['data_dirs'][0], 'stats.pbtxt'))
    normalize_list = data_norm.get_normalize_list(config)
    norm_func = data_norm.create_normalizer(stats_path, normalize_list)
    training_ds = training_ds.map(norm_func, num_parallel_calls=tf.data.AUTOTUNE)
    validation_ds = validation_ds.map(norm_func, num_parallel_calls=tf.data.AUTOTUNE)

    # Select bands. An optional 'sample_weight' config block adds a per-pixel
    # loss weight map (e.g. up-weighting fire types 3/4) to train and val.
    sample_weight_config = config.get('sample_weight')
    training_ds = data_loader.select_bands_transform(
        training_ds,
        input_feature_config=config['input_features'],
        output_feature_config=config['output_features'],
        sample_weight_config=sample_weight_config,
        pos_weight=pos_weight,
    )
    validation_ds = data_loader.select_bands_transform(
        validation_ds,
        input_feature_config=config['input_features'],
        output_feature_config=config['output_features'],
        sample_weight_config=sample_weight_config,
        pos_weight=pos_weight,
    )

    # Get branch models
    all_models = build_all_models(config['input_features'])

    # Build decoder (note: can build an identity decoder, if desired)
    model = build_decoder(config['decoder'], all_models,
                          config.get('decoder_config'),
                          num_classes=num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Training on device: {device}")

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['learning_rate'],
                                  weight_decay=weight_decay)
    decay_steps = (config['epochs'] - 1) * steps_per_epoch
    warmup_steps = 1 * steps_per_epoch
    scheduler = _cosine_warmup_schedule(optimizer, warmup_steps, decay_steps)
    # Mixed precision (mirrors the old keras mixed_float16 policy)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    # Multi-class tracks a confusion matrix and by default checkpoints on
    # foreground IoU; binary keeps the streaming ROC/PR-AUC metrics and by
    # default checkpoints on PR AUC. Any validation results key (e.g. 'loss')
    # can be configured instead.
    if num_classes > 1:
        train_metrics = MulticlassSegmentationMetrics(num_classes)
        val_metrics = MulticlassSegmentationMetrics(num_classes)
        checkpoint_metric = config.get('checkpoint_metric', 'fire_iou')
    else:
        # The area_ratio metric deflates predictions by pos_weight, but only
        # for losses that actually train toward the inflated optimum.
        metric_pos_weight = (
            pos_weight
            if config['loss_function'] in losses.POS_WEIGHT_LOSSES else 1.0)
        train_metrics = SegmentationMetrics(pos_weight=metric_pos_weight)
        val_metrics = SegmentationMetrics(pos_weight=metric_pos_weight)
        checkpoint_metric = config.get('checkpoint_metric', 'pr_auc')

    # Early stopping watches its own (configurable) metric, defaulting to the
    # checkpoint metric, so e.g. checkpointing on val loss while stopping on
    # PR AUC stagnation (or vice versa) is possible.
    early_stopping_metric = config.get('early_stopping_metric',
                                       checkpoint_metric)
    checkpoint_tracker = _BestTracker(checkpoint_metric)
    early_stop_tracker = _BestTracker(early_stopping_metric)

    checkpoint_filepath = './checkpoint.model.pt'
    csv_path = './training.csv'
    history = []
    epochs_since_improvement = 0

    for epoch in range(config['epochs']):
        train_results = _run_epoch(
            model, training_ds, loss_function, device, train_metrics,
            optimizer=optimizer, scaler=scaler, scheduler=scheduler)
        val_results = _run_epoch(
            model, validation_ds, loss_function, device, val_metrics)

        row = {'epoch': epoch, 'learning_rate': scheduler.get_last_lr()[0]}
        row.update(train_results)
        row.update({f'val_{k}': v for k, v in val_results.items()})
        history.append(row)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        print(f"Epoch {epoch + 1}/{config['epochs']}: " +
              " - ".join(f"{k}={v:.6f}" for k, v in row.items() if k != 'epoch'),
              flush=True)

        # Checkpoint on best val checkpoint_metric, stop on early_stopping_metric
        if checkpoint_tracker.improved(val_results):
            torch.save({'config': config, 'model_state_dict': model.state_dict()},
                       checkpoint_filepath)
        if early_stop_tracker.improved(val_results):
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"Early stopping: no val_{early_stopping_metric} "
                      f"improvement in {patience} epochs.")
                break

    # Load best checkpoint
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    save_model(model, config, config['model_output_path'])

    # Copy logged data to gs
    output_root, _ = os.path.splitext(config['model_output_path'])
    csv_output_path = output_root + '.csv'
    if csv_output_path.startswith("gs://"):
        upload_file_to_gcs(csv_path, csv_output_path)

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
