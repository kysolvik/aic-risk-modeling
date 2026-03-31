from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2
import tensorflow as tf
import keras

def load_stats_from_text(path):
    """Load tfdv-generated DatasetFeatureStatisticsList from a text file."""
    stats_list = statistics_pb2.DatasetFeatureStatisticsList()
    
    with tf.io.gfile.GFile(path, 'r') as f:
        stats_text = f.read()
    
    text_format.Parse(stats_text, stats_list)
    
    return stats_list

def get_norm_stats(stats_list, target_feature):
    """Extract normalization statistics for a given feature."""
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

def get_normalize_list(config):
    """Retrieve flat list of variable names to normalize."""
    normalize_list = []
    for k, f in config['input_features'].items():
        if f['normalize']:
            if len(f['timesteps']) > 0:
                normalize_list.extend([
                    fn + '_' + str(ts) for fn in f['feature_names'] for ts in f['timesteps']
                ])
            else:
                normalize_list.extend(f['feature_names'])

    # Output features
    f = config['output_features']
    if 'normalize' in f.keys() and f['normalize']:
        if len(f['timesteps']) > 0:
            normalize_list.extend([
                fn + '_' + str(ts) for fn in f['feature_names'] for ts in f['timesteps']
            ])
        else:
            normalize_list.extend(f['feature_names'])
    return normalize_list

def create_normalizer(stats_txt_path, features_to_normalize,
                      use_median=False, ignore_min=False, ignore_max=False):
    """Create a normalization function based on provided statistics."""
    norm_constants = {}
    stats_proto = load_stats_from_text(stats_txt_path)
    for name in features_to_normalize:
        s = get_norm_stats(stats_proto, name)
        if s:
            norm_constants[name] = s

    @tf.autograph.experimental.do_not_convert
    def normalize_fn(features):
        for name, stats in norm_constants.items():
            if name in features:
                if use_median:
                    center = keras.ops.convert_to_tensor(stats['median'], dtype='float32')
                else:
                    center = keras.ops.convert_to_tensor(stats['mean'], dtype='float32')
                std = keras.ops.convert_to_tensor(stats['stddev'], dtype='float32')

                if ignore_min:
                    out_tensor = keras.ops.where(features[name] == stats['min'],
                                                 center,
                                                 features[name])
                if ignore_max:
                    out_tensor = keras.ops.where(features[name] == stats['max'],
                                                 center,
                                                 features[name])

                if not ignore_min and not ignore_max:
                    out_tensor = features[name]

                
                if stats['stddev'] == 0:
                    features[name] = (keras.ops.cast(out_tensor, 'float32') - center)
                else:
                    features[name] = (keras.ops.cast(out_tensor, 'float32') - center) / (std + 1e-7)
        
        return features

    return normalize_fn