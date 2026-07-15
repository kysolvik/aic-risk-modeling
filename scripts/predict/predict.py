
import numpy as np
import tensorflow as tf
import torch
import aic_risk_modeling as arm
import rasterio as rio
from rasterio.transform import Affine
from tqdm import tqdm
import json
import argparse

TFRECORD_PATTERN = '*.tfrecord.gz'
CENTERED = True# True if x, y are for center for each tile

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
    return parser.parse_args()


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
def write_batch(outs, masks, xs, ys, base_transform, profile, output_dir, edge_crop):
    for i in range(len(outs)):
        out =outs[i]
        mask =masks[i]
        x = xs[i]
        y = ys[i]
        # Binary preds are (H, W); multiclass are (H, W, num_classes). Give both
        # an explicit band axis so we write one raster band per class.
        if out.ndim == 2:
            out = out[:, :, np.newaxis]
        transform = Affine(base_transform[0], base_transform[1], x,
                                       base_transform[3], base_transform[4], y)
        if CENTERED:
            transform = transform*rio.Affine.translation(int(-out.shape[0]/2), int(-out.shape[1]/2))
        if edge_crop > 0:
            out =out[edge_crop:-edge_crop, edge_crop:-edge_crop]
            mask =mask[edge_crop:-edge_crop, edge_crop:-edge_crop]
            transform = transform*rio.Affine.translation(8, 8)
        n_bands = out.shape[2]
        profile.update(dtype=rio.int8,
                       count=1,
                       height=out.shape[0],
                       width=out.shape[1],
                       transform=transform)
        with rio.open(
            f'{output_dir}/mask_{x}-{y}.tif', 'w', **profile) as dst_dataset:
                dst_dataset.write(mask.astype(rio.int8), 1)

        # One band per class (softmax probabilities); single band for binary.
        profile.update(dtype=rio.float32, count=n_bands)
        with rio.open(
            f'{output_dir}/out_{x}-{y}.tif', 'w', **profile) as dst_dataset:
                for b in range(n_bands):
                    dst_dataset.write(out[:, :, b].astype(rio.float32), b + 1)

def main():
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    # Add the md_sidecar passthrough group at runtime (carries md_x_raw/md_y_raw).
    config = add_md_sidecar(config)

    # Merged dataset test, merging along features-axis
    ds = arm.train.build_merged_dataset([args.data_dir],
                                        TFRECORD_PATTERN,
                                        batch_size=4,
                                        cache=False,
                                        axis='examples',
                                        shuffle=False
                                        )
    # Stash raw lat/lon before normalization so they survive as md_x_raw/md_y_raw
    ds = ds.map(set_raw_x_y)
    # Normalization
    normalize_list = arm.train.get_normalize_list(config)
    norm_func = arm.train.create_normalizer(args.data_dir + '/stats.pbtxt', normalize_list)
    ds = ds.map(norm_func)

    # Select bands (the md_sidecar group stacks md_x_raw/md_y_raw into inputs)
    ds = arm.train.select_bands_transform(
        ds,
        input_feature_config=config['input_features'],
        output_feature_config=config['output_features']
    )
    # Load checkpoint
    model = arm.train.trainer.load_model(args.checkpoint)

    all_outs = []
    all_x = []
    all_y = []
    all_masks = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    amp_enabled = device.type == 'cuda'
    amp_dtype = torch.float16 if amp_enabled else torch.bfloat16
    with torch.no_grad():
        for inputs, labels, weights in tqdm(arm.train.trainer._torch_batches(ds, device),
                                   desc='Predicting', unit='batch'):
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                preds = model(inputs)
            # Multiclass preds are (B, H, W, C): prepend the argmax class as an
            # extra band. Binary preds are (B, H, W) and must not match here
            # (shape[-1] is W for them, so a bare `shape[-1] > 1` misfires and
            # corrupts the rasters with an argmax-over-width column).
            if preds.ndim == 4 and preds.shape[-1] > 1:
                p = torch.argmax(preds, dim=-1, keepdim=True)
                preds = torch.cat([p.float(), preds], dim=-1)

            # md_sidecar is stacked [batch, 1, 2] -> (md_x_raw, md_y_raw)
            md_sidecar = inputs['md_sidecar']
            md_x_raw = md_sidecar[:, 0, 0].cpu().numpy()
            md_y_raw = md_sidecar[:, 0, 1].cpu().numpy()
            all_x.append(md_x_raw)
            all_y.append(md_y_raw)
            all_masks.append(labels.cpu().numpy())
            all_outs.append(preds.float().cpu().numpy())

    src = rio.open('./out/example.tif')
    profile = src.profile
    profile.update(
        dtype=rio.float32,
        count=1,
        compress='lzw')
    base_transform = profile['transform']

    for i in range(len(all_outs)):
        write_batch(all_outs[i], all_masks[i], all_x[i], all_y[i],
                    base_transform, profile, args.output_dir, args.edge_crop)

if __name__ == '__main__':
    main()
