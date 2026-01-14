"""Helpers to infer schema and build tf.data.Dataset directly from GCS output

Functions
- infer_schema_from_gcs(gcs_dir): load schema.pbtxt or infer from stats.tfrecord at GCS path
- schema_to_feature_spec(schema, non_img_features, patch_size): convert schema proto to TF parsing spec
- build_features_dict(schema, patch_size, non_img_features): convenience wrapper to produce a features dict
- dataset_from_gcs(tfrecord_pattern, feature_spec, input_bands, output_bands, batch_size, shuffle, compression): return batched dataset yielding (inputs_dict, labels)

Usage example
>>> from train.data import infer_schema_from_gcs, schema_to_feature_spec, dataset_from_gcs
>>> schema = infer_schema_from_gcs('gs://aic-fire-amazon/results/')
>>> feature_spec = schema_to_feature_spec(schema, non_img_features=['lon','lat','id'], patch_size=128)
>>> ds = dataset_from_gcs('gs://aic-fire-amazon/results/training-*.tfrecord.gz', feature_spec, input_bands=[...], output_bands=['BurnDate'], batch_size=8)
"""
import os
import logging
from typing import List, Dict, Tuple, Optional

import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2

logger = logging.getLogger(__name__)


def _gcs_join(base: str, name: str) -> str:
    return base.rstrip("/") + "/" + name


def infer_schema_from_gcs(gcs_dir: str) -> schema_pb2.Schema:
    """Load a schema from `schema.pbtxt` in GCS or infer from `stats.tfrecord`.

    Args:
        gcs_dir: GCS path where Dataflow results were written (e.g. gs://.../results)

    Returns:
        A tensorflow_metadata.schema_pb2.Schema

    Raises:
        FileNotFoundError: if neither `schema.pbtxt` nor `stats.tfrecord` are found
    """
    schema_path = _gcs_join(gcs_dir, "schema.pbtxt")
    stats_path = _gcs_join(gcs_dir, "stats.tfrecord")

    # Prefer existing schema
    if tf.io.gfile.exists(schema_path):
        logger.info("Loading schema from %s", schema_path)
        return tfdv.load_schema_text(schema_path)

    # Otherwise try to infer from stats
    if tf.io.gfile.exists(stats_path):
        logger.info("Inferring schema from %s", stats_path)
        stats = tfdv.load_statistics(stats_path)
        schema = tfdv.infer_schema(stats)
        return schema

    raise FileNotFoundError(
        f"Could not find schema.pbtxt or stats.tfrecord in {gcs_dir}")


def schema_to_feature_spec(
    schema: schema_pb2.Schema,
    non_img_features: Optional[List[str]] = None,
    patch_size: int = 128,
) -> Dict[str, tf.io.FixedLenFeature]:
    """Convert a schema proto to a TensorFlow feature_spec dictionary.

    Note on conversion rules:
    - BYTES -> tf.string scalar
    - INT -> tf.int64 scalar
    - FLOAT -> if feature name not in `non_img_features` assume image patch -> shape (patch_size, patch_size)
              else scalar (float)

    Args:
        schema: schema proto
        non_img_features: names to treat as non-image (scalar) floats; default ['lon','lat','id']
        patch_size: size each side of square patch

    Returns:
        Dict suitable for tf.io.parse_single_example
    """
    if non_img_features is None:
        non_img_features = ["lon", "lat", "id"]

    feature_spec = {}
    for feature in schema.feature:
        if feature.type == schema_pb2.FeatureType.BYTES:
            feature_spec[feature.name] = tf.io.FixedLenFeature([], tf.string)
        elif feature.type == schema_pb2.FeatureType.INT:
            feature_spec[feature.name] = tf.io.FixedLenFeature([], tf.int64)
        elif feature.type == schema_pb2.FeatureType.FLOAT:
            if feature.name not in non_img_features:
                feature_spec[feature.name] = tf.io.FixedLenFeature([patch_size, patch_size], tf.float32)
            else:
                feature_spec[feature.name] = tf.io.FixedLenFeature([], tf.float32)
        else:
            # Fallback to a scalar float
            feature_spec[feature.name] = tf.io.FixedLenFeature([], tf.float32)
    return feature_spec


def build_features_dict(
    schema: schema_pb2.Schema,
    patch_size: int = 128,
    non_img_features: Optional[List[str]] = None,
) -> Dict[str, tf.io.FixedLenFeature]:
    """Convenience wrapper—returns feature_spec (same shape as schema_to_feature_spec)"""
    return schema_to_feature_spec(schema, non_img_features=non_img_features, patch_size=patch_size)


def _to_tuple_transform(example: Dict, input_bands: List[str], output_bands: List[str]):
    inputs = {name: example[name] for name in input_bands}
    # If multiple outputs return dict/list, here return single output scalar/mask
    if len(output_bands) == 1:
        return inputs, example[output_bands[0]]
    return inputs, {name: example[name] for name in output_bands}


def dataset_from_gcs(
    tfrecord_pattern: str,
    feature_spec: Dict[str, tf.io.FixedLenFeature],
    input_bands: List[str],
    output_bands: List[str],
    batch_size: int = 8,
    shuffle: bool = True,
    cache: bool = True,
    compression: Optional[str] = "GZIP",
    shuffle_buffer: int = 512,
) -> tf.data.Dataset:
    """Builds a tf.data.Dataset from TFRecord files on GCS.

    Args:
        tfrecord_pattern: file glob (e.g., 'gs://.../training-*.tfrecord.gz')
        feature_spec: output of `schema_to_feature_spec`
        input_bands: list of feature names to use as inputs (image bands)
        output_bands: list of feature names to use as outputs (targets)
        batch_size: batch size
        shuffle: whether to shuffle
        cache: whether to cache dataset in memory
        compression: e.g., 'GZIP' or None

    Returns:
        A batched tf.data.Dataset yielding (inputs_dict, outputs_tensor)
    """
    files = tf.io.gfile.glob(tfrecord_pattern)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for pattern {tfrecord_pattern}")

    ds = tf.data.Dataset.list_files(tfrecord_pattern)
    ds = ds.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=compression),
                       cycle_length=tf.data.AUTOTUNE,
                       num_parallel_calls=tf.data.AUTOTUNE)

    parse_fn = lambda x: tf.io.parse_single_example(x, feature_spec)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.map(lambda ex: _to_tuple_transform(ex, input_bands, output_bands), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    # Quick example
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gcs_dir",
        required=True,
        help="GCS directory containing schema or stats (e.g. gs://.../results)"
        )
    parser.add_argument(
        "--tfrecord_pattern",
        required=False,
        help="TFRecord glob pattern (e.g. *.tfrecord.gz)",
        default="*.tfrecord.gz")
    parser.add_argument(
        "--patch_size",
        required=False,
        type=int,
        default=128)
    args = parser.parse_args()

    tfrecord_pattern_path = os.path.join(args.gcs_dir, args.tfrecord_pattern)

    schema = infer_schema_from_gcs(args.gcs_dir)
    feature_spec = build_features_dict(schema, patch_size=args.patch_size)

    print("Example features:")
    for i, f in enumerate(list(feature_spec.keys())[:20]):
        print(i + 1, f)

    ds = dataset_from_gcs(tfrecord_pattern_path, feature_spec, input_bands=[k for k in feature_spec.keys() if k not in ['lat','lon','id']], output_bands=['BurnDate'], batch_size=1)
    for inputs, labels in ds.take(1):
        print("Batch inputs keys:", list(inputs.keys()))
        print("Label shape:", labels.shape)
