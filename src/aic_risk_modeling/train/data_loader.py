"""Helpers to infer schema and build tf.data.Dataset directly from GCS output

Functions
- load_schema_from_gcs(gcs_dir): load schema.pbtxt from GCS path
- schema_to_feature_spec(schema, non_img_features, patch_size): convert schema proto to TF parsing spec
- build_features_dict(schema, patch_size, non_img_features): convenience wrapper to produce a features dict
- dataset_from_dir(tfrecord_pattern, feature_spec, batch_size, shuffle): return batched dataset with all features as dict
- select_inputs_outputs(dataset, input_bands, output_bands, transforms): extract inputs/outputs from feature dict
- merge_datasets(datasets, merge_fn): merge multiple datasets along feature axis
- apply_transforms(example, transforms): apply custom transforms to example fields

Typical workflow:
>>> # Load raw data from multiple sources
>>> ds1 = dataset_from_dir('gs://bucket/data1-*.tfrecord.gz', feature_spec)
>>> ds2 = dataset_from_dir('gs://bucket/data2-*.tfrecord.gz', feature_spec)
>>> # Merge datasets
>>> merged = merge_datasets([ds1, ds2])
>>> # Select inputs/outputs and apply transforms
>>> transforms = {'BurnDate': lambda x: x > 0}
>>> final_ds = select_inputs_outputs(merged,
>>>                                    input_bands=['A01', 'A02'],
>>>                                    output_bands=['BurnDate'],
>>>                                    transforms=transforms)
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Tuple, Optional, Callable

import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf import text_format

from aic_risk_modeling.train import transforms

logger = logging.getLogger(__name__)


def _gcs_join(base: str, name: str) -> str:
    return base.rstrip("/") + "/" + name


def load_schema_from_gcs(gcs_dir: str) -> schema_pb2.Schema:
    """Load a schema from `schema.pbtxt` in GCS or infer from `stats.tfrecord`.

    Args:
        gcs_dir: GCS path where Dataflow results were written (e.g. gs://.../results)

    Returns:
        A tensorflow_metadata.schema_pb2.Schema

    Raises:
        FileNotFoundError: if neither `schema.pbtxt` nor `stats.tfrecord` are found
    """
    schema_path = _gcs_join(gcs_dir, "schema.pbtxt")
    schema = schema_pb2.Schema()

    # Prefer existing schema
    if tf.io.gfile.exists(schema_path):
        logger.info("Loading schema from %s", schema_path)
        with tf.io.gfile.GFile(schema_path, "r") as f:
            return text_format.Parse(f.read(), schema)

    raise FileNotFoundError(
        f"Could not find schema.pbtxt in {gcs_dir}")


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


def apply_transforms(
    example: Dict,
    transforms: Optional[Dict[str, Callable]] = None
) -> Dict:
    """Apply custom transforms to specific fields in an example.

    Args:
        example: Dictionary of features
        transforms: Dict mapping feature names to transform functions.
                   If a feature has a transform, apply it; otherwise keep as-is.

    Returns:
        Dictionary with transforms applied to specified features
    """
    if transforms is None:
        return example

    result = example.copy()
    for feature_name, transform_fn in transforms.items():
        if feature_name in result:
            if callable(transform_fn):
                result[feature_name] = transform_fn(result[feature_name])
            elif isinstance(transform_fn, str):
                # Look up in registry
                try:
                    callable_fn = transforms.transform_registry[transform_fn]
                    result[feature_name] = callable_fn(result[feature_name])
                except KeyError:
                    raise ValueError(
                        f"Transform '{transform_fn}' for feature '{feature_name}' not found in registry\n."
                        "Existing transforms: " + str(transforms.transform_registry.keys()))
            else:
                raise ValueError(
                    f"Transform for feature '{feature_name}' must be a callable or a string key in the registry."
                    )



    return result


def _to_tuple_transform(
    example: Dict,
    input_bands: List[str],
    output_bands: List[str],
    transforms: Optional[Dict[str, Callable]] = None
):
    """Transform a parsed example into (inputs, outputs) tuple.

    Args:
        example: Dictionary of parsed features
        input_bands: List of feature names to use as inputs
        output_bands: List of feature names to use as outputs
        transforms: Optional dict of feature_name -> transform_fn for custom transforms

    Returns:
        Tuple of (inputs_dict, outputs_dict or outputs_tensor)
    """
    # Apply transforms first
    example = apply_transforms(example, transforms)

    # Extract inputs and outputs
    inputs = {name: example[name] for name in input_bands}

    # Return outputs based on number of output bands
    if len(output_bands) == 1:
        # Single output: return as tensor
        return inputs, example[output_bands[0]]
    else:
        # Multiple outputs: return as dict
        return inputs, {name: example[name] for name in output_bands}


def dataset_from_dir(
    dir: str,
    tfrecord_pattern: str = "*.tfrecord.gz",
    feature_spec: Optional[Dict[str, tf.io.FixedLenFeature] | None] = None,
    batch_size: int = 8,
    patch_size: int = 128,
    shuffle: bool = False,
    cache: Optional[str | bool] = False,
    compression: Optional[str] = "GZIP",
    shuffle_buffer: int = 512,
) -> tf.data.Dataset:
    """Builds a tf.data.Dataset from TFRecord files, returning all features as a dict.

    Use this to load raw data that will be merged with other datasets before selecting
    inputs/outputs. For input/output selection and transforms, use `select_inputs_outputs()`.

    Args:
        dir: Directory containing tfrecord.gz files
        tfrecord_pattern: file glob (e.g., 'training-*.tfrecord.gz')
        feature_spec: output of `schema_to_feature_spec`. Alternatively, if none will check
            for schema.pbtxt file in dir and attempt to load feature spec.
        batch_size: batch size
        shuffle: whether to shuffle
        cache: False (no caching), True (in memory caching), or str (cache to disk at path).
        compression: e.g., 'GZIP' or None
        shuffle_buffer: buffer size for shuffling

    Returns:
        A batched tf.data.Dataset yielding all features as a dict

    Example:
        >>> ds1 = dataset_from_dir('gs://.../training-*.tfrecord.gz', feature_spec, batch_size=8)
        >>> ds2 = dataset_from_dir('gs://.../other-*.tfrecord.gz', feature_spec, batch_size=8)
        >>> merged = merge_datasets([ds1, ds2])
        >>> final = select_inputs_outputs(merged, input_bands=['A01'], output_bands=['BurnDate'])
    """
    full_path_pattern = os.path.join(dir, tfrecord_pattern)
    files = tf.io.gfile.glob(full_path_pattern)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for pattern {full_path_pattern}")

    ds = tf.data.Dataset.list_files(full_path_pattern)
    ds = ds.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=compression),
                       cycle_length=tf.data.AUTOTUNE,
                       num_parallel_calls=tf.data.AUTOTUNE)

    # Get feature spec
    if feature_spec is None:
        schema = load_schema_from_gcs(dir)
        feature_spec = schema_to_feature_spec(schema)

    parse_fn = lambda x: tf.io.parse_single_example(x, feature_spec)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if isinstance(cache, str):
        ds = ds.cache(cache)
    elif cache is True:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def select_inputs_outputs(
    dataset: tf.data.Dataset,
    input_bands: List[str],
    output_bands: List[str],
    transforms: Optional[Dict[str, Callable]] = None,
) -> tf.data.Dataset:
    """Select input and output bands from a dataset of feature dicts, with optional transforms.

    Use this after merging datasets to split features into inputs/outputs.

    Args:
        dataset: A dataset yielding feature dicts (e.g., from dataset_from_dir or merge_datasets)
        input_bands: List of feature names to use as inputs
        output_bands: List of feature names to use as outputs
        transforms: Optional dict mapping feature names to transform functions.
                   E.g., {'BurnDate': lambda x: x > 0} converts BurnDate to binary.

    Returns:
        A dataset yielding (inputs_dict, outputs_dict/tensor) tuples

    Example:
        >>> merged = merge_datasets([ds1, ds2])
        >>> transforms = {'BurnDate': lambda x: x > 0}
        >>> final = select_inputs_outputs(
        ...     merged,
        ...     input_bands=['A01', 'A02'],
        ...     output_bands=['BurnDate'],
        ...     transforms=transforms
        ... )
    """
    def select_fn(example):
        return _to_tuple_transform(example, input_bands, output_bands, transforms)

    return dataset.map(select_fn, num_parallel_calls=tf.data.AUTOTUNE)


def _merged_zipped_ds(*zipped_ds):
    # Merge all input dicts
    merged_inputs = {}
    for ds in zipped_ds:
        merged_inputs.update(ds)
    return merged_inputs


def merge_datasets(
    datasets: List[tf.data.Dataset]
) -> tf.data.Dataset:
    """Merge multiple datasets by zipping them along the feature axis.

    Args:
        datasets: List of tf.data.Datasets to merge. Each should yield inputs_dict.

    Returns:
        A merged tf.data.Dataset

    Example:
        >>> ds1 = dataset_from_dir(...)  # yields (inputs1, outputs1)
        >>> ds2 = dataset_from_dir(...)  # yields (inputs2, outputs2)
        >>> merged = merge_datasets([ds1, ds2])  # yields {**inputs1, **inputs2}
    """
    if not datasets:
        raise ValueError("Must provide at least one dataset to merge")

    # Zip datasets and apply merge function
    zipped = tf.data.Dataset.zip(tuple(datasets))
    return zipped.map(_merged_zipped_ds, num_parallel_calls=tf.data.AUTOTUNE)


# Alias for backward compatibility
dataset_from_gcs = dataset_from_dir


if __name__ == "__main__":
    # Quick example demonstrating the load → merge → select workflow
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

    schema = load_schema_from_gcs(args.gcs_dir)
    feature_spec = schema_to_feature_spec(schema, patch_size=args.patch_size)

    print("Example features:")
    for i, f in enumerate(list(feature_spec.keys())[:20]):
        print(i + 1, f)

    # Example workflow: load raw data → merge → select inputs/outputs
    print("\n--- Loading raw dataset (all features) ---")
    ds_raw = dataset_from_dir(
        tfrecord_pattern_path,
        feature_spec,
        batch_size=2
    )

    for batch in ds_raw.take(1):
        print(f"Raw batch keys: {list(batch.keys())}")
        print(f"Sample feature shapes:")
        for key, val in list(batch.items())[:3]:
            print(f"  {key}: {val.shape}")

    # Select inputs/outputs with transforms
    print("\n--- After selecting inputs/outputs with transforms ---")
    transforms = {'BurnDate': lambda x: x > 0}
    ds_final = select_inputs_outputs(
        ds_raw,
        input_bands=[k for k in feature_spec.keys() if k not in ['lat', 'lon', 'id', 'BurnDate']],
        output_bands=['BurnDate'],
        transforms=transforms
    )

    for inputs, labels in ds_final.take(1):
        print(f"Inputs keys: {list(inputs.keys())}")
        print(f"Label shape: {labels.shape}")
        print(f"Label dtype: {labels.dtype}")
