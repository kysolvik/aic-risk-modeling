"""Pool per-year tfdv stats.pbtxt summaries into one data_stats-style JSON.

Each fullgrid export dir (allpreds_<year>/) ships a stats.pbtxt with, per
numeric feature: tot_num_values, mean, std_dev, min, max, median, and a
QUANTILES histogram. Streaming the raw TFRecords again just to pool
normalization constants is wasteful, so this reads only
those ~1.5 MiB summaries and pools them:

  * count / mean / stddev  -- exact (Chan et al. parallel variance merge)
  * min / max              -- exact (just keep the running values)
  * median / robust_scale  -- from the mixture of per-year QUANTILES histograms
                              (each bucket treated as uniform mass), solved by
                              bisection on the pooled piecewise-uniform CDF

The output matches aic_risk_modeling.train.data_stats.write_stats schema, plus a
per-feature `robust_scale` (IQR/1.349) so robust-normed bands keep the tfdv
quantile scale instead of falling back to std_dev. data_norm.get_norm_stats
consumes it directly as stats_path.

Usage:
    python scripts/preprocessing/pool_stats_pbtxt.py \
        --data_dirs gs://aic-amazon/data/fullgrid_v3/allpreds_2013/ ... \
        --output gs://aic-amazon/data/fullgrid_v3/stats_2013_2022.json
"""

import argparse
import datetime
import json

import numpy as np
import tensorflow as tf
from tensorflow_metadata.proto.v0 import statistics_pb2

from aic_risk_modeling.train import data_norm

ROBUST_DIVISOR = 1.349  # 2 * 0.6745; matches std_dev for normal data


def _quantile_buckets(num_stats):
    """Return [(low, high, mass), ...] from the QUANTILES histogram, or []."""
    for hist in num_stats.histograms:
        if hist.type != statistics_pb2.Histogram.QUANTILES or not hist.buckets:
            continue
        return [(b.low_value, b.high_value, b.sample_count) for b in hist.buckets]
    return []


class QuantilePool:
    """Mixture of piecewise-uniform buckets; supports pooled quantile queries."""

    def __init__(self):
        self.buckets = []  # (low, high, mass)

    def add(self, buckets):
        self.buckets.extend(buckets)

    @property
    def total_mass(self):
        return sum(m for _, _, m in self.buckets)

    def _cdf(self, v):
        # Fraction of total mass at or below v, each bucket assumed uniform.
        acc = 0.0
        for low, high, mass in self.buckets:
            if v >= high:
                acc += mass
            elif v > low and high > low:
                acc += mass * (v - low) / (high - low)
        return acc

    def quantile(self, p):
        if not self.buckets:
            return None
        target = p * self.total_mass
        lo = min(low for low, _, _ in self.buckets)
        hi = max(high for _, high, _ in self.buckets)
        if hi <= lo:
            return lo
        for _ in range(100):  # bisection to ~machine precision on [lo, hi]
            mid = 0.5 * (lo + hi)
            if self._cdf(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


class Pooled:
    """Exact streaming pool of count/mean/M2/min/max plus a QuantilePool."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = np.inf
        self.max = -np.inf
        self.q = QuantilePool()

    def add(self, n, mean, std_dev, vmin, vmax, qbuckets):
        if n <= 0:
            return
        m2_b = std_dev * std_dev * n
        total = self.count + n
        delta = mean - self.mean
        self.mean += delta * n / total
        self.m2 += m2_b + delta * delta * self.count * n / total
        self.count = total
        self.min = min(self.min, vmin)
        self.max = max(self.max, vmax)
        if qbuckets:
            self.q.add(qbuckets)

    def result(self):
        if self.count == 0:
            return None
        out = {
            'count': int(self.count),
            'mean': float(self.mean),
            'stddev': float(np.sqrt(self.m2 / self.count)),
            'min': float(self.min),
            'max': float(self.max),
        }
        median = self.q.quantile(0.5)
        out['median'] = float(median) if median is not None else float(self.mean)
        if self.q.buckets:
            iqr = self.q.quantile(0.75) - self.q.quantile(0.25)
            if iqr > 0:
                out['robust_scale'] = float(iqr / ROBUST_DIVISOR)
        return out


def pool_stats(data_dirs, stats_filename='stats.pbtxt'):
    pools = {}
    for d in data_dirs:
        path = d.rstrip('/') + '/' + stats_filename
        stats_list = data_norm.load_stats_from_text(path)
        n_feats = 0
        for dataset in stats_list.datasets:
            for feature in dataset.features:
                if not feature.HasField('num_stats'):
                    continue
                num = feature.num_stats
                name = feature.path.step[0]
                pools.setdefault(name, Pooled()).add(
                    n=num.common_stats.tot_num_values,
                    mean=num.mean,
                    std_dev=num.std_dev,
                    vmin=num.min,
                    vmax=num.max,
                    qbuckets=_quantile_buckets(num),
                )
                n_feats += 1
        print(f"Pooled {n_feats} numeric features from {path}")
    features = {name: p.result() for name, p in sorted(pools.items())
                if p.result() is not None}
    return {
        'features': features,
        'metadata': {
            'source': 'pool_stats_pbtxt',
            'data_dirs': list(data_dirs),
            'stats_filename': stats_filename,
            'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dirs', nargs='+', required=True,
                    help='Export dirs (local or gs://), each with a stats.pbtxt')
    ap.add_argument('--output', required=True, help='Output JSON (local or gs://)')
    ap.add_argument('--stats_filename', default='stats.pbtxt')
    args = ap.parse_args()

    stats = pool_stats(args.data_dirs, args.stats_filename)
    with tf.io.gfile.GFile(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote pooled stats for {len(stats['features'])} features to "
          f"{args.output}")


if __name__ == '__main__':
    main()
