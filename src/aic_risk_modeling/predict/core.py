"""Shared prediction pipeline: config -> normalized dataset -> model -> per-chip GeoTIFFs."""

import numpy as np
import rasterio as rio
import torch
from rasterio.transform import Affine

from aic_risk_modeling import train

TFRECORD_PATTERN = '*.tfrecord.gz'
CENTERED = True

# Passthrough group that carries the raw coordinates through to the model inputs.
MD_SIDECAR_GROUP = {
    'feature_names': ['md_x_raw', 'md_y_raw'],
    'transforms': {},
    'timesteps': [],
    'shape': [1],
    'stack_timesteps': False,
    'normalize': False,
    'model_type': 'none',
}


def add_common_args(parser, default_profile_template):
    """Flags shared by predict.py and attribute.py."""
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--edge_crop', type=int, default=0)
    parser.add_argument(
        '--stats_path', default=None,
        help="normalization stats; default config['stats_path']")
    parser.add_argument(
        '--profile_template', default=default_profile_template,
        help='GeoTIFF giving the output CRS and pixel size')
    parser.add_argument('--tfrecord_pattern', default=TFRECORD_PATTERN)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_chips', type=int, default=None)
    parser.add_argument(
        '--seed', type=int, default=None,
        help='seed the chip order (matters for --max_chips)')


def resolve_stats_path(explicit, config, data_dir):
    """--stats_path > config['stats_path'] > <data_dir>/stats.pbtxt.

    Use the training stats: per-data_dir stats re-center each year and erase the year offset."""
    if explicit:
        return explicit
    if config.get('stats_path'):
        return config['stats_path']
    return data_dir.rstrip('/') + '/stats.pbtxt'


def add_md_sidecar(config):
    """Add the md_sidecar passthrough group so a training config works for prediction."""
    config['input_features']['md_sidecar'] = dict(MD_SIDECAR_GROUP)
    return config


def set_raw_x_y(features):
    """Copy raw md_x/md_y before normalization so they reach the outputs for georeferencing."""
    features['md_x_raw'] = features['md_x']
    features['md_y_raw'] = features['md_y']
    return features


def build_dataset(config, data_dir, stats_path, tfrecord_pattern, batch_size, seed):
    """Unshuffled, normalized (inputs, labels) batches; `config` must carry md_sidecar."""
    ds = train.build_merged_dataset([data_dir], tfrecord_pattern, batch_size=batch_size,
                                    shuffle=False, seed=seed)
    ds = ds.map(set_raw_x_y)
    norm_func = train.create_normalizer(
        stats_path, train.get_normalize_list(config),
        robust_features=train.get_robust_normalize_list(config))
    ds = ds.map(norm_func)
    return train.select_bands_transform(
        ds, input_feature_config=config['input_features'],
        output_feature_config=config['output_features'])


def load_for_inference(checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return train.trainer.load_model(checkpoint).to(device), device


def load_profile(template):
    """Single-band float32 LZW profile taken from the template GeoTIFF."""
    with rio.open(template) as src:
        profile = src.profile
    profile.update(dtype=rio.float32, count=1, compress='lzw')
    return profile


def sidecar_xy(inputs):
    """Raw centre coords from the md_sidecar group, stacked [batch, 1, 2]."""
    md = inputs['md_sidecar']
    return md[:, 0, 0].cpu().numpy(), md[:, 0, 1].cpu().numpy()


def write_batch(outs, masks, xs, ys, base_transform, profile, output_dir,
                edge_crop, band_names=None, out_prefix='out', write_mask=True):
    """One `<out_prefix>_<x>-<y>.tif` (+ `mask_<x>-<y>.tif`) per chip in the batch."""
    for i in range(len(outs)):
        out = outs[i]
        mask = masks[i]
        x = xs[i]
        y = ys[i]
        # Predictions are (H, W); attribution bands are (H, W, n_bands).
        if out.ndim == 2:
            out = out[:, :, np.newaxis]
        transform = Affine(base_transform[0], base_transform[1], x,
                           base_transform[3], base_transform[4], y)
        if CENTERED:
            transform = transform*rio.Affine.translation(int(-out.shape[0]/2), int(-out.shape[1]/2))
        if edge_crop > 0:
            out = out[edge_crop:-edge_crop, edge_crop:-edge_crop]
            mask = mask[edge_crop:-edge_crop, edge_crop:-edge_crop]
            transform = transform*rio.Affine.translation(edge_crop, edge_crop)
        n_bands = out.shape[2]
        profile.update(dtype=rio.int8,
                       count=1,
                       height=out.shape[0],
                       width=out.shape[1],
                       transform=transform)
        if write_mask:
            with rio.open(
                    f'{output_dir}/mask_{x}-{y}.tif', 'w', **profile) as dst_dataset:
                dst_dataset.write(mask.astype(rio.int8), 1)

        profile.update(dtype=rio.float32, count=n_bands)
        with rio.open(
                f'{output_dir}/{out_prefix}_{x}-{y}.tif', 'w', **profile) as dst_dataset:
            for b in range(n_bands):
                dst_dataset.write(out[:, :, b].astype(rio.float32), b + 1)
                if band_names:
                    dst_dataset.set_band_description(b + 1, band_names[b])
