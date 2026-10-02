"""Pooled per-feature normalization stats (exact moments + subsampled median) as JSON.

Usage: python -m aic_risk_modeling.train.data_stats --data_dirs DIR [DIR ...] --output stats.json"""

import argparse
import datetime
import json

import numpy as np
import tensorflow as tf

from aic_risk_modeling.train import data_loader

DEFAULT_RESERVOIR_SIZE = 10_000


class FeatureAccumulator:
    """Streaming exact count/mean/M2/min/max plus a bottom-k uniform subsample for the median."""

    def __init__(self, reservoir_size=DEFAULT_RESERVOIR_SIZE, seed=54):
        self.reservoir_size = reservoir_size
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = np.inf
        self.max = -np.inf
        self.sample_keys = np.empty(0)
        self.sample_values = np.empty(0)

    def update(self, values):
        v = np.asarray(values, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return

        n_b = v.size
        mean_b = v.mean()
        m2_b = v.var() * n_b
        n = self.count + n_b
        delta = mean_b - self.mean
        self.mean += delta * n_b / n
        self.m2 += m2_b + delta * delta * self.count * n_b / n
        self.count = n

        self.min = min(self.min, float(v.min()))
        self.max = max(self.max, float(v.max()))

        keys = np.concatenate([self.sample_keys, self.rng.random(v.size)])
        vals = np.concatenate([self.sample_values, v])
        if keys.size > self.reservoir_size:
            keep = np.argpartition(keys, self.reservoir_size)[:self.reservoir_size]
            keys, vals = keys[keep], vals[keep]
        self.sample_keys, self.sample_values = keys, vals

    def result(self):
        if self.count == 0:
            return None
        return {
            'count': int(self.count),
            'mean': float(self.mean),
            'stddev': float(np.sqrt(self.m2 / self.count)),
            'min': float(self.min),
            'max': float(self.max),
            'median': float(np.median(self.sample_values)),
        }


def compute_stats(data_dirs, tfrecord_pattern="*.tfrecord.gz", batch_size=32,
                  reservoir_size=DEFAULT_RESERVOIR_SIZE, max_batches_per_dir=None):
    """Stream TFRecords from data_dirs -> {'features': {name: stats}, 'metadata': {...}}."""
    accumulators = {}
    n_examples = 0
    for data_dir in data_dirs:
        ds = data_loader.dataset_from_dir(
            data_dir, tfrecord_pattern=tfrecord_pattern, batch_size=batch_size)
        for i, batch in enumerate(ds.as_numpy_iterator()):
            if max_batches_per_dir is not None and i >= max_batches_per_dir:
                break
            for name, arr in batch.items():
                if not np.issubdtype(arr.dtype, np.number):
                    continue
                if name not in accumulators:
                    accumulators[name] = FeatureAccumulator(reservoir_size)
                accumulators[name].update(arr)
            n_examples += len(next(iter(batch.values())))
        print(f"Processed {data_dir} (cumulative examples: {n_examples})")

    features = {name: acc.result() for name, acc in sorted(accumulators.items())
                if acc.result() is not None}
    return {
        'features': features,
        'metadata': {
            'data_dirs': list(data_dirs),
            'tfrecord_pattern': tfrecord_pattern,
            'n_examples': n_examples,
            'reservoir_size': reservoir_size,
            'max_batches_per_dir': max_batches_per_dir,
            'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    }


def write_stats(stats, output_path):
    with tf.io.gfile.GFile(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote stats for {len(stats['features'])} features to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute pooled normalization stats for TFRecord dirs")
    parser.add_argument('--data_dirs', nargs='+', required=True,
                        help='TFRecord dirs (local or gs://); stats are pooled')
    parser.add_argument('--output', required=True,
                        help='Output JSON path (local or gs://)')
    parser.add_argument('--tfrecord_pattern', default='*.tfrecord.gz')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--reservoir_size', type=int,
                        default=DEFAULT_RESERVOIR_SIZE,
                        help='Subsample size per feature for the median')
    parser.add_argument('--max_batches_per_dir', type=int, default=None,
                        help='Cap batches per dir for quick approximate runs')
    args = parser.parse_args()

    stats = compute_stats(
        data_dirs=args.data_dirs,
        tfrecord_pattern=args.tfrecord_pattern,
        batch_size=args.batch_size,
        reservoir_size=args.reservoir_size,
        max_batches_per_dir=args.max_batches_per_dir,
    )
    write_stats(stats, args.output)


if __name__ == "__main__":
    main()
