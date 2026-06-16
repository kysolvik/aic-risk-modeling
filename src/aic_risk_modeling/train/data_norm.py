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

def create_normalizer(stats_path, features_to_normalize,
                      use_median=False, ignore_min=False, ignore_max=False):
    """Create a normalization function based on provided statistics.

    `stats_path` may be a data_stats JSON file (*.json) or a tfdv stats.pbtxt.
    """
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
                if use_median:
                    center = tf.constant(stats['median'], dtype=tf.float32)
                else:
                    center = tf.constant(stats['mean'], dtype=tf.float32)
                std = tf.constant(stats['stddev'], dtype=tf.float32)

                if ignore_min:
                    out_tensor = tf.where(features[name] == stats['min'],
                                          center,
                                          features[name])
                if ignore_max:
                    out_tensor = tf.where(features[name] == stats['max'],
                                          center,
                                          features[name])

                if not ignore_min and not ignore_max:
                    out_tensor = features[name]


                if stats['stddev'] == 0:
                    features[name] = (tf.cast(out_tensor, tf.float32) - center)
                else:
                    features[name] = (tf.cast(out_tensor, tf.float32) - center) / (std + 1e-7)

        return features

    return normalize_fn
