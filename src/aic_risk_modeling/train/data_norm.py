import json

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2
import tensorflow as tf

def load_stats_from_text(path):
    """Load tfdv-generated DatasetFeatureStatisticsList from a text file."""
    stats_list = statistics_pb2.DatasetFeatureStatisticsList()

    with tf.io.gfile.GFile(path, 'r') as f:
        stats_text = f.read()

    text_format.Parse(stats_text, stats_list)

    return stats_list

def load_stats_json(path):
    """Load stats written by data_stats.write_stats (local or gs://)."""
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)

def _robust_scale_from_quantiles(num_stats):
    """Robust scale (IQR / 1.349) from a tfdv QUANTILES histogram.

    1.349 = 2 * 0.6745, so for normally distributed data this matches the
    standard deviation. Robust normalization uses this instead of std_dev so a
    nodata value baked into the raw values (e.g. AgERA5 temperature
    unmask(0), whose 0 K pixels inflate std_dev ~10x and squash the band)
    cannot corrupt the scale. As long as no data is below a25 or above q75,
     should be fairly robust. Returns None when no usable quantile
    histogram is present or the IQR is degenerate.
    """
    from tensorflow_metadata.proto.v0 import statistics_pb2
    for hist in num_stats.histograms:
        if hist.type != statistics_pb2.Histogram.QUANTILES or not hist.buckets:
            continue
        edges = [hist.buckets[0].low_value] + [b.high_value for b in hist.buckets]
        n = len(edges) - 1  # number of equal-count buckets (deciles => 10)
        if n < 1:
            continue

        def quantile(pct):
            pos = pct * n
            lo = int(pos)
            if lo >= n:
                return edges[n]
            return edges[lo] + (pos - lo) * (edges[lo + 1] - edges[lo])

        iqr = quantile(0.75) - quantile(0.25)
        if iqr > 0:
            return iqr / 1.349
    return None


def get_norm_stats(stats_list, target_feature):
    """Extract normalization statistics for a given feature.

    Accepts either a tfdv DatasetFeatureStatisticsList proto or the dict
    loaded from a data_stats JSON file.
    """
    if isinstance(stats_list, dict):
        return stats_list.get('features', stats_list).get(target_feature)
    for dataset in stats_list.datasets:
        for feature in dataset.features:
            feat_name = feature.path.step[0]
            if feat_name == target_feature:
                num_stats = feature.num_stats
                return {
                    'mean': num_stats.mean,
                    'stddev': num_stats.std_dev,
                    'min': num_stats.min,
                    'max': num_stats.max,
                    'median': num_stats.median,
                    'robust_scale': _robust_scale_from_quantiles(num_stats),
                }
    return None


def _normalize_single_features_dict(f, normalize_list):
    if 'normalize' in f.keys() and f['normalize']:
        for fn in f['feature_names']:
            if fn not in f['transforms'].keys():
                if len(f['timesteps']) > 0:
                    normalize_list.extend([
                        fn + '_' + str(ts) for ts in f['timesteps']
                    ])
                else:
                    normalize_list.append(fn)
    return normalize_list

def get_normalize_list(config):
    """Retrieve flat list of variable names to normalize.

    IMPORTANT: if transform is defined for var, skips normalizing
    """
    normalize_list = []

    # Input features
    for k, f in config['input_features'].items():
        normalize_list = _normalize_single_features_dict(f, normalize_list)

    # Output features
    f = config['output_features']
    normalize_list = _normalize_single_features_dict(f, normalize_list)

    return normalize_list


def _robust_normalize_single_features_dict(f, robust_list):
    for fn in f.get('robust_norm', []):
        if len(f['timesteps']) > 0:
            robust_list.extend([fn + '_' + str(ts) for ts in f['timesteps']])
        else:
            robust_list.append(fn)
    return robust_list

def get_robust_normalize_list(config):
    """Retrieve flat list of variable names that should use robust
    normalization: filters out min NA values (values equal to the feature's
    global min are replaced) plus median instead of mean centering.
    """
    robust_list = []

    for k, f in config['input_features'].items():
        robust_list = _robust_normalize_single_features_dict(f, robust_list)

    f = config['output_features']
    robust_list = _robust_normalize_single_features_dict(f, robust_list)

    return robust_list

def create_normalizer(stats_path, features_to_normalize, robust_features=None):
    """Create a normalization function based on provided statistics.

    `stats_path` may be a data_stats JSON file (*.json) or a tfdv stats.pbtxt.

    `robust_features` is an iterable of (already timestep-expanded) feature
    names that should use robust normalization instead of the default: median
    (rather than mean) centering, and values equal to the feature's global min
    replaced with that center before scaling. This is meant for features
    exported with a nodata value (e.g. -32768) baked into the raw values,
    which otherwise skews the mean/variance used for standardization. See
    `get_robust_normalize_list` for deriving this from a training config.
    """
    robust_features = set(robust_features or [])
    norm_constants = {}
    if stats_path.endswith('.json'):
        stats = load_stats_json(stats_path)
    else:
        stats = load_stats_from_text(stats_path)
    for name in features_to_normalize:
        s = get_norm_stats(stats, name)
        if s:
            norm_constants[name] = s

    @tf.autograph.experimental.do_not_convert
    def normalize_fn(features):
        for name, stats in norm_constants.items():
            if name in features:
                if name in ['md_x_topleft','md_x', 'md_y_topleft',
                            'md_y', 'md_id']:
                    features[f'{name}_raw'] = features[name]

                is_robust = name in robust_features
                if is_robust:
                    center_val = stats['median']
                    # Scale by a robust spread (IQR/1.349) when the stats source
                    # provides quantiles, so a nodata sentinel baked into the
                    # raw values cannot inflate the scale and squash the band.
                    # Falls back to std_dev for sources without quantiles
                    # (e.g. data_stats JSON), preserving prior behavior.
                    scale_val = stats.get('robust_scale') or stats['stddev']
                else:
                    center_val = stats['mean']
                    scale_val = stats['stddev']
                center = tf.constant(center_val, dtype=tf.float32)
                scale = tf.constant(scale_val, dtype=tf.float32)

                if is_robust:
                    out_tensor = tf.where(features[name] == stats['min'],
                                          center,
                                          features[name])
                else:
                    out_tensor = features[name]

                # Some exported bands carry NaN where the source asset has no
                # coverage (im_chirps_cwd_monthly is ~4-5% of chips). The stats
                # exclude NaN from accumulation, so center/scale stay finite, but
                # an unfilled NaN pixel propagates all the way to the loss.
                # Impute the center so those pixels standardize to 0.
                out_tensor = tf.cast(out_tensor, tf.float32)
                out_tensor = tf.where(tf.math.is_finite(out_tensor),
                                      out_tensor, center)

                if scale_val == 0:
                    features[name] = out_tensor - center
                else:
                    features[name] = (out_tensor - center) / (scale + 1e-7)

        return features

    return normalize_fn
