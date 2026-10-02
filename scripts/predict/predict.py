
import os

import numpy as np
import tensorflow as tf
import torch
import aic_risk_modeling as arm
import rasterio as rio
from rasterio.transform import Affine
from tqdm import tqdm
import argparse

TFRECORD_PATTERN = '*.tfrecord.gz'
CENTERED = True# True if x, y are for center for each tile

# GeoTIFF supplying the output CRS and pixel size. Resolved off __file__ rather
# than the cwd so it works from any working directory, in a container or out.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
DEFAULT_PROFILE_TEMPLATE = os.path.join(_REPO_ROOT, 'assets', 'example_v3.tif')

# Passthrough feature group injected into the config at runtime so the raw
# (un-normalized) coordinates ride along in the model inputs dict.
MD_SIDECAR_GROUP = {
    'feature_names': ['md_x_raw', 'md_y_raw'],
    'transforms': {},
    'timesteps': [],
    'shape': [1],
    'stack_timesteps': False,
    'normalize': False,
    'model_type': 'none',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--edge_crop',
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        '--stats_path',
        type=str,
        default=None,
        help='normalization stats (.json or .pbtxt), local or gs://; default is '
             "config['stats_path'], else <data_dir>/stats.pbtxt (legacy)"
    )
    parser.add_argument(
        '--profile_template',
        type=str,
        default=DEFAULT_PROFILE_TEMPLATE,
        help='GeoTIFF supplying the CRS and pixel size for the output chips'
    )
    parser.add_argument(
        '--tfrecord_pattern',
        type=str,
        default=TFRECORD_PATTERN
    )
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument(
        '--max_chips', type=int, default=None,
        help='stop after this many chips (smoke runs)')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='seed the tfrecord listing/interleave order; without it chip order '
             'varies run to run, which matters only for --max_chips and A/B runs')
    return parser.parse_args()


def resolve_stats_path(explicit, config, data_dir):
    """Pick the normalization stats file.

    Precedence: --stats_path > config['stats_path'] > <data_dir>/stats.pbtxt.

    Prediction must normalize with the same statistics the model trained on.
    The per-data_dir fallback re-centers every prediction year independently,
    which erases the year-to-year offset the model learned; it is kept only so
    older invocations that relied on it keep working. Note the GCS allpreds_*
    directories carry no stats.pbtxt at all, so the fallback cannot serve them.
    """
    if explicit:
        return explicit
    if config.get('stats_path'):
        return config['stats_path']
    # rstrip: 'gs://b/d//stats.pbtxt' is a different object from 'gs://b/d/stats.pbtxt'
    return data_dir.rstrip('/') + '/stats.pbtxt'


def add_md_sidecar(config):
    """Inject the md_sidecar passthrough group so a training config can be reused
    for prediction without maintaining a separate `_pred.json`."""
    config['input_features']['md_sidecar'] = dict(MD_SIDECAR_GROUP)
    return config


def set_raw_x_y(features):
    """Stash the raw coords before normalization so they survive into the model
    inputs (via the md_sidecar group) for georeferencing the output tiles.

    Copies rather than pops so the normalized md_x/md_y remain available to the
    md_single input group.
    """
    features['md_x_raw'] = features['md_x']
    features['md_y_raw'] = features['md_y']
    return features


# dtype to uint8, and specify LZW compression.
def write_batch(outs, masks, xs, ys, base_transform, profile, output_dir,
                edge_crop, band_names=None, out_prefix='out', write_mask=True):
    for i in range(len(outs)):
        out =outs[i]
        mask =masks[i]
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
            out =out[edge_crop:-edge_crop, edge_crop:-edge_crop]
            mask =mask[edge_crop:-edge_crop, edge_crop:-edge_crop]
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

def main():
    args = parse_args()
    # load_config handles gs:// via tf.io.gfile; plain open() does not.
    config = arm.train.trainer.load_config(args.config_path)
    stats_path = resolve_stats_path(args.stats_path, config, args.data_dir)
    print(f'[predict] normalizing with stats: {stats_path}', flush=True)
    # Add the md_sidecar passthrough group at runtime (carries md_x_raw/md_y_raw).
    config = add_md_sidecar(config)

    ds = arm.train.build_merged_dataset([args.data_dir],
                                        args.tfrecord_pattern,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        seed=args.seed
                                        )
    # Stash raw lat/lon before normalization so they survive as md_x_raw/md_y_raw
    ds = ds.map(set_raw_x_y)
    # Normalization
    normalize_list = arm.train.get_normalize_list(config)
    robust_features = arm.train.get_robust_normalize_list(config)
    norm_func = arm.train.create_normalizer(
        stats_path, normalize_list, robust_features=robust_features)
    ds = ds.map(norm_func)

    # Select bands (the md_sidecar group stacks md_x_raw/md_y_raw into inputs)
    ds = arm.train.select_bands_transform(
        ds,
        input_feature_config=config['input_features'],
        output_feature_config=config['output_features']
    )
    # Load checkpoint
    model = arm.train.trainer.load_model(args.checkpoint)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    amp_enabled = device.type == 'cuda'
    amp_dtype = torch.float16 if amp_enabled else torch.bfloat16

    # Set the output profile up before the loop: rasters are written per batch
    # rather than accumulated, so a full grid stays flat in memory.
    os.makedirs(args.output_dir, exist_ok=True)
    with rio.open(args.profile_template) as src:
        profile = src.profile
    profile.update(
        dtype=rio.float32,
        count=1,
        compress='lzw')
    base_transform = profile['transform']

    n_chips = 0
    with torch.no_grad():
        for inputs, labels in tqdm(arm.train.trainer._torch_batches(ds, device),
                                   desc='Predicting', unit='batch'):
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                preds = model(inputs)

            # md_sidecar is stacked [batch, 1, 2] -> (md_x_raw, md_y_raw)
            md_sidecar = inputs['md_sidecar']
            md_x_raw = md_sidecar[:, 0, 0].cpu().numpy()
            md_y_raw = md_sidecar[:, 0, 1].cpu().numpy()

            write_batch(preds.float().cpu().numpy(), labels.cpu().numpy(),
                        md_x_raw, md_y_raw, base_transform, profile,
                        args.output_dir, args.edge_crop)

            n_chips += int(labels.shape[0])
            if args.max_chips and n_chips >= args.max_chips:
                break

    print(f'[predict] wrote {n_chips} chips to {args.output_dir}', flush=True)

if __name__ == '__main__':
    main()
