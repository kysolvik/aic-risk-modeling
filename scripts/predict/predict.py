"""Write per-chip prediction GeoTIFFs (out_<x>-<y>.tif + mask_<x>-<y>.tif) for one data dir."""

import argparse
import os

import torch
from tqdm import tqdm

from aic_risk_modeling.predict import core
from aic_risk_modeling.train import trainer

# Resolved off __file__ rather than the cwd so it works from any working directory,
# in a container or out.
DEFAULT_PROFILE_TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'assets', 'example_v3.tif')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    core.add_common_args(parser, DEFAULT_PROFILE_TEMPLATE)
    args = parser.parse_args()

    # load_config handles gs:// via tf.io.gfile; plain open() does not.
    config = trainer.load_config(args.config_path)
    stats_path = core.resolve_stats_path(args.stats_path, config, args.data_dir)
    print(f'[predict] normalizing with stats: {stats_path}', flush=True)
    config = core.add_md_sidecar(config)
    ds = core.build_dataset(config, args.data_dir, stats_path, args.tfrecord_pattern,
                            args.batch_size, args.seed)
    model, device = core.load_for_inference(args.checkpoint)
    amp_enabled = device.type == 'cuda'
    amp_dtype = torch.float16 if amp_enabled else torch.bfloat16

    # Rasters are written per batch rather than accumulated, so a full grid stays
    # flat in memory.
    os.makedirs(args.output_dir, exist_ok=True)
    profile = core.load_profile(args.profile_template)
    base_transform = profile['transform']

    n_chips = 0
    with torch.no_grad():
        for inputs, labels in tqdm(trainer._torch_batches(ds, device),
                                   desc='Predicting', unit='batch'):
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                preds = model(inputs)
            xs, ys = core.sidecar_xy(inputs)
            core.write_batch(preds.float().cpu().numpy(), labels.cpu().numpy(),
                             xs, ys, base_transform, profile,
                             args.output_dir, args.edge_crop)
            n_chips += int(labels.shape[0])
            if args.max_chips and n_chips >= args.max_chips:
                break

    print(f'[predict] wrote {n_chips} chips to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
