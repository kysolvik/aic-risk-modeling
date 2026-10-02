"""PyTorch training loop with tf.data input; checkpoints and early-stops on configurable val metrics.

checkpoint_metric 'last' keeps the final epoch and is the only mode that allows no val set."""

import csv
import glob
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
from aic_risk_modeling.train.metrics import SegmentationMetrics

# TensorFlow is only used for data loading; keep it off the GPU.
tf.config.set_visible_devices([], "GPU")

SEED = 54
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)


def set_seed(seed):
    """Re-seed torch/numpy/tf from the config's seed; returns it for the data pipeline."""
    global RNG
    RNG = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    return seed


def upload_file_to_gcs(local_path, gcs_uri):
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with 'gs://'")

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

def build_decoder(decoder_type, branch_models, decoder_config=None):
    function_name = f"decoder_{decoder_type}"
    try:
        model_fn = getattr(models, function_name)
        model = model_fn(branch_models, **(decoder_config or {}))
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
        n_timesteps = len(input_dict['timesteps'])
        n_features = len(input_dict['feature_names'])
        if n_timesteps > 0 and input_dict.get('stack_timesteps'):
            # Time stays its own axis; otherwise it is folded into channels.
            input_shape = [n_timesteps] + input_dict['shape'] + [n_features]
        elif n_timesteps > 0:
            input_shape = input_dict['shape'] + [n_features * n_timesteps]
        else:
            input_shape = input_dict['shape'] + [n_features]
        print(input_shape)
        input_name = input_key
        model_kwargs = input_dict.get('model_kwargs') or {}
        all_models.append(
            build_model(model_type, input_shape, input_name, **model_kwargs))

    return all_models


def save_model(model, config, output_path):
    """Save weights plus the config needed to rebuild the model (local or gs://)."""
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
                          config.get('decoder_config'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def _cache_dataset_to_disk(dataset, cache_dir):
    """Replay a deterministic (validation) dataset from local disk after the first pass.

    Deletes stale cache files first: Dataset.cache() silently serves whatever is on disk."""
    os.makedirs(cache_dir, exist_ok=True)
    prefix = os.path.join(cache_dir, 'val')
    for stale in glob.glob(prefix + '*'):
        os.remove(stale)
    return dataset.cache(prefix).prefetch(tf.data.AUTOTUNE)


def _torch_batches(dataset, device):
    """Yield (inputs, labels) batches as torch tensors on `device`."""
    for inputs, labels in dataset.as_numpy_iterator():
        inputs = {k: torch.as_tensor(v).to(device) for k, v in inputs.items()}
        yield inputs, torch.as_tensor(labels).float().to(device)


def _cosine_warmup_schedule(optimizer, warmup_steps, decay_steps):
    """Per-step linear warmup from 0, then cosine decay to 0."""
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
        for inputs, labels in _torch_batches(dataset, device):
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                preds = model(inputs)
            loss = loss_function(labels, preds)

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
    """Whether a val metric improved ('loss' minimized, others maximized; 'last' always improves)."""

    def __init__(self, metric):
        self.metric = metric
        self._sign = -1.0 if metric == 'loss' else 1.0
        self._best = float('-inf')

    def improved(self, results):
        if self.metric == 'last':
            return True
        value = self._sign * results[self.metric]
        if value > self._best:
            self._best = value
            return True
        return False


def _monitoring(config):
    """(has_val, checkpoint_metric, early_stopping_metric); without val data only 'last' is allowed."""
    has_val = bool(config.get('val_data_dirs'))
    checkpoint_metric = config.get('checkpoint_metric', 'pr_auc')
    early_stopping_metric = config.get('early_stopping_metric',
                                       checkpoint_metric)
    if not has_val:
        bad = {m for m in (checkpoint_metric, early_stopping_metric) if m != 'last'}
        if bad:
            raise ValueError(
                f"no val_data_dirs: checkpoint/early-stopping metric {sorted(bad)} "
                "needs a validation set (use checkpoint_metric 'last')")
    return has_val, checkpoint_metric, early_stopping_metric


def run(config):
    seed = set_seed(config.get('seed', SEED))
    steps_per_epoch = config.get('steps_per_epoch', 5000)
    weight_decay = config.get('weight_decay', 0.01)
    patience = config.get('early_stopping_patience', 4)
    pos_weight = config.get('pos_weight', 9.0)
    # Fail before any data loads if the config can't be monitored.
    has_val, checkpoint_metric, early_stopping_metric = _monitoring(config)

    loss_function = losses.get_loss(config['loss_function'], pos_weight=pos_weight)

    training_ds = data_loader.build_merged_dataset(
        data_dirs=config['data_dirs'],
        tfrecord_pattern=config['tfrecord_pattern'],
        shuffle=True,
        batch_size=config['batch_size'],
        seed=seed,
    )
    validation_ds = data_loader.build_merged_dataset(
        data_dirs=config['val_data_dirs'],
        tfrecord_pattern=config['tfrecord_pattern'],
        shuffle=False,
        batch_size=config['batch_size'],
        seed=seed,
    ) if has_val else None

    # Explicit stats file if given, else the first data dir's stats.pbtxt.
    stats_path = config.get(
        'stats_path', _gcs_join(config['data_dirs'][0], 'stats.pbtxt'))
    normalize_list = data_norm.get_normalize_list(config)
    robust_features = data_norm.get_robust_normalize_list(config)
    norm_func = data_norm.create_normalizer(
        stats_path, normalize_list, robust_features=robust_features)
    training_ds = training_ds.map(norm_func, num_parallel_calls=tf.data.AUTOTUNE)
    if has_val:
        validation_ds = validation_ds.map(norm_func, num_parallel_calls=tf.data.AUTOTUNE)

    training_ds = data_loader.select_bands_transform(
        training_ds,
        input_feature_config=config['input_features'],
        output_feature_config=config['output_features'],
    )
    if has_val:
        validation_ds = data_loader.select_bands_transform(
            validation_ds,
            input_feature_config=config['input_features'],
            output_feature_config=config['output_features'],
        )
    # Must be the last validation op so the cache holds fully processed batches.
    if has_val and config.get('val_cache_dir'):
        validation_ds = _cache_dataset_to_disk(
            validation_ds, config['val_cache_dir'])

    all_models = build_all_models(config['input_features'])

    model = build_decoder(config['decoder'], all_models,
                          config.get('decoder_config'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Training on device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['learning_rate'],
                                  weight_decay=weight_decay)
    decay_steps = (config['epochs'] - 1) * steps_per_epoch
    warmup_steps = 1 * steps_per_epoch
    scheduler = _cosine_warmup_schedule(optimizer, warmup_steps, decay_steps)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    train_metrics = SegmentationMetrics(pos_weight=pos_weight)
    val_metrics = SegmentationMetrics(pos_weight=pos_weight)

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
            model, validation_ds, loss_function, device, val_metrics) if has_val else {}

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

    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    save_model(model, config, config['model_output_path'])

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

    config = load_config(args.config_path)

    run(config)

    print("Training complete, model saved to:", config['model_output_path'])
