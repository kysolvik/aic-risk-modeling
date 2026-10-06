"""One target year's static bands from the static export (geebeam_static_463m.py).

Per chip (md_id): MapBiomas years placed in their annual slots (<class>_-<slot>), and
Hansen loss / mean lossyear cut at the year's cutoff (sums of the per-year loss
fractions, exact since both are linear in the 30 m pixels). The other bands are copied.
Output names = the inline bands of the earlier exports. Run via merge_static.sh.

    python build_static_year.py --static_dir gs://.../fullgrid_v5/static --target_year 2025 \
        --output_dir ~/static_year/allpreds_2025
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from aic_risk_modeling.preprocess import cutoff

LULC_FEATURES = ['forest', 'pasture', 'ag', 'urban', 'mining', 'water']
N_ANNUAL = 10


def mapbiomas_year(target_year, slot):
    """Year in annual slot -`slot` (1 = newest usable year)."""
    y = cutoff.annual_year('mapbiomas_amazonia', target_year) - (slot - 1)
    if y > cutoff.MAPBIOMAS_AMAZONIA_LAST_YEAR:
        raise ValueError(f'MapBiomas Amazonia col6 ends {cutoff.MAPBIOMAS_AMAZONIA_LAST_YEAR}; '
                         f'target {target_year} needs {y}. Switch to a newer collection '
                         '(and re-export every year with it).')
    return y


def year_bands(static, target_year):
    """{year band name: array} from {static band name: array}."""
    out = {}
    for slot in range(N_ANNUAL, 0, -1):
        y = mapbiomas_year(target_year, slot)
        for name in LULC_FEATURES:
            out[f'{name}_{-slot}'] = static[f'{name}_{y}']

    # Hansen: loss known through the newest usable year only
    lossfrac = [static[f'lossfrac_{k:02d}']
                for k in range(1, cutoff.annual_year('hansen', target_year) - 2000 + 1)]
    out['loss'] = sum(lossfrac)
    out['lossyear'] = sum(k * f for k, f in enumerate(lossfrac, 1))

    for name in ['treecover2000', 'Elevation', 'Slope', 'accessibility', 'gov_type']:
        out[name] = static[name]
    return out


def convert_example(serialized, target_year):
    feats = tf.train.Example.FromString(serialized).features.feature
    static = {key[3:]: np.asarray(feat.float_list.value, dtype=np.float32)
              for key, feat in feats.items() if key.startswith('im_')}
    out = {'md_id': tf.train.Feature(int64_list=tf.train.Int64List(value=feats['md_id'].int64_list.value))}
    for name, arr in year_bands(static, target_year).items():
        out[f'im_{name}'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=arr.astype(np.float32)))
    return tf.train.Example(features=tf.train.Features(feature=out)).SerializeToString()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--static_dir', required=True)
    ap.add_argument('--target_year', type=int, required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--tfrecord_pattern', default='*.tfrecord.gz')
    args = ap.parse_args(argv)

    shards = sorted(tf.io.gfile.glob(os.path.join(args.static_dir, args.tfrecord_pattern)))
    if not shards:
        raise ValueError(f'no shards in {args.static_dir}')
    tf.io.gfile.makedirs(args.output_dir)
    n = 0
    for shard in shards:
        dst = os.path.join(args.output_dir, os.path.basename(shard))
        with tf.io.TFRecordWriter(dst, options='GZIP') as writer:
            for rec in tf.data.TFRecordDataset([shard], compression_type='GZIP'):
                writer.write(convert_example(rec.numpy(), args.target_year))
                n += 1
        print(f'  {dst} ({n} chips so far)')
    print(f'{n} chips -> {args.output_dir}')


if __name__ == '__main__':
    main()
