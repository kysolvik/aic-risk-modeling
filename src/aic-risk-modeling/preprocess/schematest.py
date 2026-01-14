import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2


NON_IMG_FEATURES = ['lon','lat','id']
PATCH_SIZE = 128

def schema_to_feature_spec(schema, non_img_features, patch_size):
    feature_spec = {}
    for feature in schema.feature:
        if feature.type == schema_pb2.FeatureType.BYTES:
            feature_spec[feature.name] = tf.io.FixedLenFeature([], tf.string)
        elif feature.type == schema_pb2.FeatureType.INT:
            feature_spec[feature.name] = tf.io.FixedLenFeature([], tf.int64)
        elif feature.type == schema_pb2.FeatureType.FLOAT:
            if not feature.name in non_img_features:
                print(feature.name)
                feature_spec[feature.name] = tf.io.FixedLenFeature([patch_size, patch_size], tf.float32)
            else:
                feature_spec[feature.name] = tf.io.FixedLenFeature([], tf.float32)
    return feature_spec


schema = tfdv.load_schema_text('schema.pbtxt')
feature_spec = schema_to_feature_spec(schema, NON_IMG_FEATURES, PATCH_SIZE)

pattern = './*.tfrecord.gz'
dataset = tf.data.Dataset.list_files(pattern).interleave(
    lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'))
dataset = dataset.map(
    lambda x: tf.io.parse_single_example(x, feature_spec)
)


# Inspect the first element from the training dataset.
for inputs in dataset.take(1):
  print("inputs:")
  for name, values in inputs.items():
    print(f"  {name}: {values.dtype.name} {values.shape}")