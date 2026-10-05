"""Feature normalization from tfdv stats.pbtxt or data_stats JSON, applied in tf.data."""
import json

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2
import tensorflow as tf
from . import transforms

def load_stats_from_text(path):
    stats_list = statistics_pb2.DatasetFeatureStatisticsList()

    with tf.io.gfile.GFile(path, 'r') as f:
        stats_text = f.read()

    text_format.Parse(stats_text, stats_list)

    return stats_list

def load_stats_json(path):
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)

def _robust_scale_from_quantiles(num_stats):
    """IQR / 1.349 from a tfdv QUANTILES histogram (= std for normal data), or None."""
    from tensorflow_metadata.proto.v0 import statistics_pb2
    for hist in num_stats.histograms:
        if hist.type != statistics_pb2.Histogram.QUANTILES or not hist.buckets:
            continue
        edges = [hist.buckets[0].low_value] + [b.high_value for b in hist.buckets]
        n = len(edges) - 1
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
    """Normalization stats for one feature from a stats proto or a data_stats dict."""
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
            if (fn not in f['transforms'].keys()
                    or f['transforms'][fn] in transforms.NORMALIZE_THROUGH_TRANSFORMS):
                if len(f['timesteps']) > 0:
                    normalize_list.extend([
                        fn + '_' + str(ts) for ts in f['timesteps']
                    ])
                else:
                    normalize_list.append(fn)
    return normalize_list

def get_normalize_list(config):
    """Timestep-expanded names to z-score; transformed features are skipped unless value-preserving."""
    normalize_list = []

    for k, f in config['input_features'].items():
        normalize_list = _normalize_single_features_dict(f, normalize_list)

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
    """Timestep-expanded names listed under `robust_norm` in the config."""
    robust_list = []

    for k, f in config['input_features'].items():
        robust_list = _robust_normalize_single_features_dict(f, robust_list)

    f = config['output_features']
    robust_list = _robust_normalize_single_features_dict(f, robust_list)

    return robust_list

def load_stats(stats_path):
    if stats_path.endswith('.json'):
        return load_stats_json(stats_path)
    return load_stats_from_text(stats_path)


def create_normalizer(stats_path, features_to_normalize, robust_features=None):
    """Build a tf.data map fn that standardizes `features_to_normalize` in place.

    Robust features are median-centred, scaled by IQR/1.349 when available, and have values
    equal to the global min (a nodata sentinel) replaced by the median."""
    robust_features = set(robust_features or [])
    norm_constants = {}
    stats = load_stats(stats_path)
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

                # Some bands carry NaN where the source has no coverage; impute the center.
                out_tensor = tf.cast(out_tensor, tf.float32)
                out_tensor = tf.where(tf.math.is_finite(out_tensor),
                                      out_tensor, center)

                if scale_val == 0:
                    features[name] = out_tensor - center
                else:
                    features[name] = (out_tensor - center) / (scale + 1e-7)

        return features

    return normalize_fn
