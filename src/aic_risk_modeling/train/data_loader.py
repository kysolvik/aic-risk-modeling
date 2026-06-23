"""Helpers to infer schema and build tf.data.Dataset directly from GCS output

Functions
- load_schema_from_gcs(gcs_dir): load schema.pbtxt from GCS path
- schema_to_feature_spec(schema, non_img_features, patch_size): convert schema proto to TF parsing spec
- build_features_dict(schema, patch_size, non_img_features): convenience wrapper to produce a features dict
- dataset_from_dir(tfrecord_pattern, feature_spec, batch_size, shuffle): return batched dataset with all features as dict
- select_bands_transform(dataset, input_bands, output_bands, transforms): extract inputs/outputs from feature dict
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
>>> final_ds = select_bands_transform(merged,
>>>                                    input_bands=['A01', 'A02'],
>>>                                    output_bands=['BurnDate'],
>>>                                    transforms=transforms)
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict,  Optional, Callable

import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict

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
    patch_size: int = 128
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
    feature_spec = {}
    for feature in schema.feature:
        if feature.name.startswith('im_'):
            tf_size = [patch_size, patch_size]
        else:
            feature_size = int(MessageToDict(feature)['shape']['dim'][0]['size'])
            if feature_size > 0:
                tf_size = [feature_size]
            else:
                tf_size = []
        if feature.type == schema_pb2.FeatureType.BYTES:
            feature_spec[feature.name] = tf.io.FixedLenFeature(tf_size, tf.string)
        elif feature.type == schema_pb2.FeatureType.INT:
            feature_spec[feature.name] = tf.io.FixedLenFeature(tf_size, tf.int64)
        elif feature.type == schema_pb2.FeatureType.FLOAT:
                feature_spec[feature.name] = tf.io.FixedLenFeature(tf_size, tf.float32)
        else:
            # Fallback to a scalar float
            feature_spec[feature.name] = tf.io.FixedLenFeature([], tf.float32)
    return feature_spec


def build_features_dict(
    schema: schema_pb2.Schema,
    patch_size: int
) -> Dict[str, tf.io.FixedLenFeature]:
    """Convenience wrapper—returns feature_spec (same shape as schema_to_feature_spec)"""
    return schema_to_feature_spec(schema, patch_size)

def _apply_single_transform(result, feature_name, transform_fn):
    if callable(transform_fn):
        result[feature_name] = transform_fn(result[feature_name])
    elif isinstance(transform_fn, str):
        # Look up in registry
        try:
            callable_fn = transforms.transform_registry[transform_fn]
            return callable_fn(result[feature_name])
        except KeyError:
            raise ValueError(
                f"Transform '{transform_fn}' for feature '{feature_name}' not found in registry\n."
                "Existing transforms: " + str(transforms.transform_registry.keys()))
    else:
        raise ValueError(
            f"Transform for feature '{feature_name}' must be a callable or a string key in the registry."
            )

def apply_transforms(
    example: Dict,
    transform_dict: Optional[Dict[str, Callable]] = None,
    timesteps: Optional[List[int]] = None,
) -> Dict:
    """Apply custom transforms to specific fields in an example.

    Args:
        example: Dictionary of features
        transforms: Dict mapping feature names to transform functions.
                   If a feature has a transform, apply it; otherwise keep as-is.

    Returns:
        Dictionary with transforms applied to specified features
    """
    if transform_dict is None or len(transform_dict)==0:
        return example

    result = example.copy()
    done_list = []
    for feature_name, transform_fn in transform_dict.items():
        # First check for transforms with exact name match (no adding years)
        # This could include "BurnDate_2024", which would override a general
        # "BurnDate" transform
        if feature_name not in done_list and feature_name in result:
            result[feature_name] = _apply_single_transform(result, feature_name, transform_fn)
            done_list.append(feature_name)
        if timesteps is not None and len(timesteps) > 0:
            # Then check for transforms with years appended
            # If "BurnDate" transform is specified, it will be applied for all
            # years (e.g. BurnDate_2023, BurnDate_2022...), but NOT those which
            # had their own transform specified (e.g. BurnDate_2024, in the example above)
            for ts in timesteps:
                feature_name_wyear = f"{feature_name}_{ts}"
                if feature_name_wyear not in done_list and feature_name_wyear in result:
                    result[feature_name_wyear] = _apply_single_transform(
                        result, feature_name_wyear, transform_fn)
    return result


def dataset_from_dir(
    dir: str,
    tfrecord_pattern: str = "*.tfrecord.gz",
    feature_spec: Optional[Dict[str, tf.io.FixedLenFeature] | None] = None,
    batch_size: int = 8,
    shuffle: bool = False,
    rename_dict=None,
    cache: Optional[str | bool] = False,
    compression: Optional[str] = "GZIP",
    shuffle_buffer: int = 512,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Builds a tf.data.Dataset from TFRecord files, returning all features as a dict.

    Use this to load raw data that will be merged with other datasets before selecting
    inputs/outputs. For input/output selection and transforms, use `select_bands_transform()`.

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
        seed: optional RNG seed. When set, file listing and shuffling are
            reproducible and interleave is forced deterministic, so repeated
            runs see the identical data order. When None (default), order is
            random as before.

    Returns:
        A batched tf.data.Dataset yielding all features as a dict

    Example:
        >>> ds1 = dataset_from_dir('gs://.../training-*.tfrecord.gz', feature_spec, batch_size=8)
        >>> ds2 = dataset_from_dir('gs://.../other-*.tfrecord.gz', feature_spec, batch_size=8)
        >>> merged = merge_datasets([ds1, ds2])
        >>> final = select_bands_transform(merged, input_bands=['A01'], output_bands=['BurnDate'])
    """
    full_path_pattern = os.path.join(dir, tfrecord_pattern)
    files = tf.io.gfile.glob(full_path_pattern)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for pattern {full_path_pattern}")

    ds = tf.data.Dataset.list_files(full_path_pattern, seed=seed)

    @tf.autograph.experimental.do_not_convert
    def interleave_fn(x):
        return tf.data.TFRecordDataset(x, compression_type=compression)

    ds = ds.interleave(interleave_fn,
                       cycle_length=tf.data.AUTOTUNE,
                       num_parallel_calls=tf.data.AUTOTUNE,
                       deterministic=True if seed is not None else None)

    # Get feature spec
    if feature_spec is None:
        schema = load_schema_from_gcs(dir)
        feature_spec = schema_to_feature_spec(schema)

    @tf.autograph.experimental.do_not_convert
    def parse_fn(x):
        return tf.io.parse_single_example(x, feature_spec)

    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if rename_dict is not None:
        @tf.autograph.experimental.do_not_convert
        def _rename_features(example):
            return {rename_dict.get(k, k): v for k, v in example.items()}
        ds = ds.map(_rename_features, num_parallel_calls=tf.data.AUTOTUNE)
    if isinstance(cache, str):
        ds = ds.cache(cache)
    elif cache is True:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def _stack_time_series(features, input_keys, years):
    grouped_tensors = []
    for year in years:
        year_keys = [k for k in input_keys if k.endswith(f"_{year}")]
        year_tensor = _stack_vars(features, year_keys)
        grouped_tensors.append(year_tensor)

    timeseries_tensor = tf.stack(grouped_tensors, axis=1)
    return timeseries_tensor


def _stack_vars(features, input_keys, exclude_keys: Optional[List[str]] = None):
    if exclude_keys:
        filter_keys = [k for k in input_keys if k not in exclude_keys]
    else:
        filter_keys = input_keys

    stacked_tensor = tf.stack([features[k] for k in filter_keys], axis=-1)

    return stacked_tensor

def _prep_metadata(example):
    """Just coords for now"""
    return tf.stack([example['md_y'], example['md_x']], axis=-1)

def _reshape_tensors(
        example,
        shape
    ):
    return {key: tf.reshape(example[key], [-1] + shape) for key in example.keys()}

def _single_feature_group_prep(
        example,
        feature_config
):
    # Apply transforms
    example = apply_transforms(example,
                               feature_config['transforms'],
                               feature_config['timesteps']
                               )

    # Append timesteps to input names, if necessary
    if feature_config['timesteps'] is None or len(feature_config['timesteps']) == 0:
        inputs_w_time = feature_config['feature_names']
    else:
        inputs_w_time = [f"{k}_{ts}" for k in feature_config['feature_names'] for ts in feature_config['timesteps']]
    all_inputs = {name: example[name] for name in inputs_w_time}

    # Stack (if neither, returns dict)
    if feature_config['stack_timesteps']:
        # Get groups of years
        all_inputs = _stack_time_series(all_inputs, inputs_w_time, feature_config['timesteps'])
    else:
        all_inputs = _stack_vars(all_inputs, inputs_w_time)

    return all_inputs

def _to_tuple_transform(
    example: Dict,
    input_feature_config: dict,
    output_feature_config: dict
):
    """Transform a parsed example into (inputs, outputs) tuple.

    Returns:
        Tuple of (inputs_dict, outputs_dict or outputs_tensor)
    """

    # Input features first
    inputs = {}
    for feat_group in input_feature_config.keys():
        inputs[feat_group] = _single_feature_group_prep(
            example,
            input_feature_config[feat_group]
        )

    # Return outputs based on number of output bands
    if len(output_feature_config['feature_names']) == 1:
        # Single output: return as tensor
        outputs = _single_feature_group_prep(
            example,
            output_feature_config
        )[...,0]
    else:
        # Multiple outputs: return as dict
        outputs = _single_feature_group_prep(
            example,
            output_feature_config
        )
    return inputs, outputs

def select_bands_transform(
    dataset: tf.data.Dataset,
    input_feature_config: dict,
    output_feature_config: dict
) -> tf.data.Dataset:
    """Select input and output bands from a dataset of feature dicts, with optional transforms.

    Use this after merging datasets to split features into inputs/outputs.

    Args:
        dataset: A dataset yielding feature dicts (e.g., from dataset_from_dir or merge_datasets)
        input_feature_config:
        output_feature_config:
    Returns:
        A dataset yielding (inputs_dict, outputs_dict/tensor) tuples
    """
    def select_fn(example):
        inputs, labels = _to_tuple_transform(example, input_feature_config, output_feature_config)
        return inputs, labels

    return dataset.map(select_fn, num_parallel_calls=tf.data.AUTOTUNE)

def _merged_zipped_ds(*zipped_ds):
    # Merge all input dicts
    merged_inputs = {}
    for ds in zipped_ds:
        merged_inputs.update(ds)
    return merged_inputs

def _remove_unshared_features(datasets):
    """Remove features that aren't shared across all datasets.

    Args:
        datasets: List of tf.data.Datasets, each yielding feature dicts

    Returns:
        List of datasets with a map applied that filters to only shared feature keys
    """
    if not datasets:
        return datasets

    # Get feature keys from first batch of each dataset
    shared_keys = None
    for ds in datasets:
        # Take one batch to inspect keys
        batch_keys = set(ds.element_spec.keys())
        if shared_keys is None:
            shared_keys = batch_keys
        else:
            shared_keys = shared_keys.intersection(batch_keys)

    if shared_keys is None:
        raise ValueError("Could not determine feature keys from datasets")

    # Filter each dataset to only include shared keys
    filtered_datasets = []
    for ds in datasets:
        def filter_features(features):
            return {k: v for k, v in features.items() if k in shared_keys}
        filtered_ds = ds.map(filter_features, num_parallel_calls=tf.data.AUTOTUNE)
        filtered_datasets.append(filtered_ds)

    return filtered_datasets

def merge_datasets(
    datasets: List[tf.data.Dataset],
    axis: str,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Merge multiple datasets by zipping them along the feature axis.

    Args:
        datasets: List of tf.data.Datasets to merge. Each should yield inputs_dict.
        axis: "examples" or "features".
        seed: optional RNG seed for the "examples" sampling, so the order in
            which datasets are interleaved is reproducible. None (default) keeps
            the previous random behavior.

    Returns:
        A merged tf.data.Dataset
    """
    if not datasets:
        raise ValueError("Must provide at least one dataset to merge")

    if axis == "features":
        # Zip datasets and apply merge function
        zipped = tf.data.Dataset.zip(tuple(datasets))
        return zipped.map(_merged_zipped_ds, num_parallel_calls=tf.data.AUTOTUNE)
    elif axis == "examples":
        datasets = _remove_unshared_features(datasets)
        return tf.data.Dataset.sample_from_datasets(datasets, seed=seed)
    else:
        raise ValueError('merge_datasets axis must be either "examples" or "features".'
                         'Got {}'.format(axis))


def build_merged_dataset(
        data_dirs,
        tfrecord_pattern,
        axis='examples', # examples or features
        shuffle=True,
        cache=False,
        rename_dict=None,
        batch_size=4
        seed=None,
        ):
    datasets = []
    for data_dir in data_dirs:
        ds = dataset_from_dir(
            data_dir,
            tfrecord_pattern=tfrecord_pattern,
            cache=cache,
            batch_size=batch_size,
            shuffle=False, # Shuffling will occur with overall dataset
            rename_dict=rename_dict,
            seed=seed,
        )
        datasets.append(ds)

    merged = merge_datasets(datasets, axis=axis, seed=seed)

    if shuffle:
        merged = merged.shuffle(buffer_size=128, seed=seed)

    return merged


# Alias for backward compatibility
dataset_from_gcs = dataset_from_dir
