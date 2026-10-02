"""Write per-chip driver-attribution GeoTIFFs: attr_<x>-<y>.tif (OAT) or shap_<x>-<y>.tif (--shapley).

Bands: risk, one per driver, residual_interactions, risk_all_drivers_baseline (deflated probabilities).
Usage: attribute.py --config_path C --checkpoint M --data_dir D --output_dir O [--shapley]"""

import argparse
import os
import time

from tqdm import tqdm

from aic_risk_modeling.eval import attribution
from aic_risk_modeling.predict import core
from aic_risk_modeling.train import data_norm, trainer
from predict import DEFAULT_PROFILE_TEMPLATE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    core.add_common_args(parser, DEFAULT_PROFILE_TEMPLATE)
    parser.add_argument(
        '--drivers', type=str, default=None,
        help='driver-spec JSON; default built-in DEFAULT_DRIVERS')
    parser.add_argument(
        '--pos_weight', type=float, default=None,
        help='deflation weight; default config pos_weight')
    parser.add_argument(
        '--write_mask', action='store_true',
        help='also write mask_{x}-{y}.tif')
    parser.add_argument(
        '--shapley', action='store_true',
        help='Shapley attribution instead of OAT occlusion')
    parser.add_argument(
        '--shapley_samples', type=int, default=None,
        help='Monte-Carlo permutations instead of exact Shapley')
    parser.add_argument(
        '--shapley_seed', type=int, default=0,
        help='seed for --shapley_samples')
    return parser.parse_args()


def check_stats_coverage(stats_path, normalize_list):
    """Warn about normalized features missing from the stats file (they would stay raw)."""
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
    if args.shapley_samples is not None and not args.shapley:
        raise SystemExit('--shapley_samples requires --shapley')
    config = trainer.load_config(args.config_path)
    pos_weight = (args.pos_weight if args.pos_weight is not None
                  else config.get('pos_weight', 9.0))

    # Resolve drivers before adding md_sidecar so it can never be named as a driver.
    spec_json = None
    if args.drivers:
        spec_json = trainer.load_config(args.drivers)
    driver_spec = attribution.resolve_driver_spec(
        spec_json, config['input_features'])
    baselines = attribution.resolve_baselines(driver_spec, config)
    config = core.add_md_sidecar(config)

    stats_path = core.resolve_stats_path(args.stats_path, config, args.data_dir)
    print(f'[attribute] normalizing with stats: {stats_path}', flush=True)
    check_stats_coverage(stats_path, data_norm.get_normalize_list(config))
    ds = core.build_dataset(config, args.data_dir, stats_path, args.tfrecord_pattern,
                            args.batch_size, args.seed)
    model, device = core.load_for_inference(args.checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)
    profile = core.load_profile(args.profile_template)
    base_transform = profile['transform']

    n_chips = 0
    names = []
    start = time.time()
    # No autocast even on GPU: deltas can be ~1e-3 and must stay float32.
    out_prefix = 'shap' if args.shapley else 'attr'
    for inputs, labels in tqdm(trainer._torch_batches(ds, device),
                                  desc='Attributing', unit='batch'):
        if args.shapley:
            bands, names = attribution.shapley_bands(
                model, inputs, driver_spec, baselines, pos_weight,
                samples=args.shapley_samples, seed=args.shapley_seed)
        else:
            bands, names = attribution.attribution_bands(
                model, inputs, driver_spec, baselines, pos_weight)

        xs, ys = core.sidecar_xy(inputs)
        core.write_batch(bands.float().cpu().numpy(), labels.cpu().numpy(),
                         xs, ys, base_transform, profile,
                         args.output_dir, args.edge_crop,
                         band_names=names, out_prefix=out_prefix,
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
