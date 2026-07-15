"""Write per-driver risk-attribution rasters for a trained binary model.

For every chip in a prediction data dir, runs the one-at-a-time occlusion
attribution from `aic_risk_modeling.eval.attribution` (see its docstring for
method, baselines, and caveats) and writes one multi-band
GeoTIFF per chip, `attr_{x}-{y}.tif`, next to the usual prediction naming.

Bands (descriptions are set, so QGIS shows them):
    1                risk -- deflated (calibrated-scale) burn probability
    2 .. N+1         delta_<driver> -- contribution of each driver, spec
                     order; positive = raises risk vs grid-average conditions
    N+2              residual_interactions -- (risk - band N+3) - sum(deltas)
    N+3              risk_all_drivers_baseline

NOTE: bands are DEFLATED probabilities, while predict.py's out_*.tif hold the raw
(inflated) model output.

Cost: N+2 model forwards per batch (9 with the 7 default drivers) vs 1 for
predict.py; measured ~20 s/chip on CPU at batch_size 4 (v11, 2026-07-15), so
a full-grid year (~2,800 chips) is ~15 CPU-hours. Use --drivers with fewer
groups, --batch_size, or --max_chips to manage runtime; sharding by tfrecord
across processes also works.

Example (CPU, needs GCS read access):
    python scripts/predict/attribute.py \
        --config_path configs/mtsvit_test_v11.json \
        --checkpoint out/mtsvit_test_v11.pt \
        --data_dir gs://aic-amazon/data/fullgrid/allpreds_2024 \
        --output_dir out/attr_v11_2024_smoke --edge_crop 8 --max_chips 8
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rasterio as rio
import torch
from tqdm import tqdm

import aic_risk_modeling as arm
from aic_risk_modeling.eval import attribution
from aic_risk_modeling.train import data_norm

from predict import (add_md_sidecar, set_raw_x_y, write_batch,
                     TFRECORD_PATTERN)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--edge_crop', type=int, default=0)
    parser.add_argument(
        '--drivers', type=str, default=None,
        help='driver-spec JSON (see configs/attribution_drivers_default.json);'
             ' default = built-in DEFAULT_DRIVERS')
    parser.add_argument(
        '--pos_weight', type=float, default=None,
        help='deflation weight; default = config pos_weight (9.0 if unset)')
    parser.add_argument(
        '--max_chips', type=int, default=None,
        help='stop after this many chips (smoke runs)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument(
        '--write_mask', action='store_true',
        help='also write mask_{x}-{y}.tif ground-truth rasters')
    return parser.parse_args()


def check_stats_coverage(stats_path, normalize_list):
    """Warn about normalized features missing from the stats file: the
    pipeline silently leaves those raw, which would make the 0.0 grid-average
    baseline (and training itself) wrong."""
    if stats_path.endswith('.json'):
        stats = data_norm.load_stats_json(stats_path)
    else:
        stats = data_norm.load_stats_from_text(stats_path)
    missing = [n for n in normalize_list
               if not data_norm.get_norm_stats(stats, n)]
    if missing:
        print(f'WARNING: {len(missing)} features have no stats entry and '
              f'stay un-normalized (baseline 0.0 invalid for them): '
              f'{missing[:10]}{"..." if len(missing) > 10 else ""}')


def main():
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    pos_weight = (args.pos_weight if args.pos_weight is not None
                  else config.get('pos_weight', 9.0))

    # Resolve drivers against the real input groups BEFORE injecting the
    # md_sidecar passthrough group, so it can never be named as a driver.
    spec_json = None
    if args.drivers:
        with open(args.drivers, 'r') as f:
            spec_json = json.load(f)
    driver_spec = attribution.resolve_driver_spec(
        spec_json, config['input_features'])
    baselines = attribution.resolve_baselines(driver_spec, config)
    config = add_md_sidecar(config)

    ds = arm.train.build_merged_dataset([args.data_dir],
                                        TFRECORD_PATTERN,
                                        batch_size=args.batch_size,
                                        cache=False,
                                        axis='examples',
                                        shuffle=False
                                        )
    ds = ds.map(set_raw_x_y)
    normalize_list = arm.train.get_normalize_list(config)
    stats_path = args.data_dir.rstrip('/') + '/stats.pbtxt'
    check_stats_coverage(stats_path, normalize_list)
    norm_func = arm.train.create_normalizer(stats_path, normalize_list)
    ds = ds.map(norm_func)
    ds = arm.train.select_bands_transform(
        ds,
        input_feature_config=config['input_features'],
        output_feature_config=config['output_features']
    )

    model = arm.train.trainer.load_model(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    src = rio.open('./out/example.tif')
    profile = src.profile
    profile.update(
        dtype=rio.float32,
        count=1,
        compress='lzw')
    base_transform = profile['transform']

    n_chips = 0
    names = []
    start = time.time()
    # No autocast even on GPU: deltas can be ~1e-3 and must stay float32.
    # Rasters are written per batch rather than accumulated (10 float32
    # bands per chip adds up over a full grid).
    for inputs, labels, _ in tqdm(arm.train.trainer._torch_batches(ds, device),
                                  desc='Attributing', unit='batch'):
        bands, names = attribution.attribution_bands(
            model, inputs, driver_spec, baselines, pos_weight)

        # md_sidecar is stacked [batch, 1, 2] -> (md_x_raw, md_y_raw)
        md_sidecar = inputs['md_sidecar']
        md_x_raw = md_sidecar[:, 0, 0].cpu().numpy()
        md_y_raw = md_sidecar[:, 0, 1].cpu().numpy()
        write_batch(bands.float().cpu().numpy(), labels.cpu().numpy(),
                    md_x_raw, md_y_raw, base_transform, profile,
                    args.output_dir, args.edge_crop,
                    band_names=names, out_prefix='attr',
                    write_mask=args.write_mask)

        n_chips += labels.shape[0]
        if args.max_chips and n_chips >= args.max_chips:
            break

    elapsed = time.time() - start
    print(f'Wrote attribution rasters for {n_chips} chips to '
          f'{args.output_dir} ({elapsed / max(n_chips, 1):.1f} s/chip; bands: '
          f'{", ".join(names)})')


if __name__ == '__main__':
    main()
