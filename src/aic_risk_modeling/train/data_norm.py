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
                    'max': num_stats.max
                }
    return None

def create_normalizer(stats_txt_path, features_to_normalize):
    """Create a normalization function based on provided statistics."""
    norm_constants = {}
    stats_proto = load_stats_from_text(stats_txt_path)
    for name in features_to_normalize:
        s = get_norm_stats(stats_proto, name)
        if s:
            norm_constants[name] = s

    def normalize_fn(features):
        for name, stats in norm_constants.items():
            if name in features:
                mean = tf.constant(stats['mean'], dtype=tf.float32)
                std = tf.constant(stats['stddev'], dtype=tf.float32)
                
                features[name] = (tf.cast(features[name], tf.float32) - mean) / (std + 1e-7)
        
        return features

    return normalize_fn