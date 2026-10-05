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

import math
import os
import logging
from typing import List, Dict,  Optional, Callable

import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict

from aic_risk_modeling.train import data_norm, transforms

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


DEFAULT_PATCH_SIZE = 128  # fullgrid v2/v3 chips; used when the schema has no shape


def image_side(feature: schema_pb2.Feature) -> int:
    """Side of a square `im_` feature from its schema shape (flattened length).

    tfdv's infer_schema records the length of fixed-length features, so the chip
    size travels with the data (128 at 463 m, 64 at 1 km). Falls back to
    DEFAULT_PATCH_SIZE when the schema carries no shape.
    """
    if not feature.shape.dim:
        return DEFAULT_PATCH_SIZE
    n = int(feature.shape.dim[0].size)
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f'{feature.name}: length {n} is not a square image')
    return side


def schema_to_feature_spec(
    schema: schema_pb2.Schema,
    non_img_features: Optional[List[str]] = None,
    patch_size: Optional[int] = None
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
        patch_size: size each side of square patch; None (default) reads it from
            each image feature's schema shape (see `image_side`)

    Returns:
        Dict suitable for tf.io.parse_single_example
    """
    feature_spec = {}
    for feature in schema.feature:
        if feature.name.startswith('im_'):
            side = patch_size or image_side(feature)
            tf_size = [side, side]
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
    return schema_to_feature_spec(schema, patch_size=patch_size)

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


def _combine_output_bands(stacked, mode):
    """Reduce the trailing feature axis of a stacked output to one channel.

    Lets several label bands be merged into a single binary target, e.g. a
    union of two fire sensors that each miss fire the other sees. ``'any'``
    is the union (positive where any source band is positive), ``'all'`` the
    intersection. Boolean only.

    Args:
        stacked: tensor whose last axis indexes the source bands, as returned
            by `_stack_vars`.
        mode: 'any' or 'all'.

    Returns:
        Tensor with the trailing axis reduced away, same dtype as `stacked`.
    """
    reduce_fn = {'any': tf.reduce_any, 'all': tf.reduce_all}.get(mode)
    if reduce_fn is None:
        raise ValueError(
            f"Unknown output combine mode {mode!r}. Expected 'any' or 'all'.")
    combined = reduce_fn(tf.cast(stacked, tf.bool), axis=-1)
    return tf.cast(combined, stacked.dtype)


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

def build_type_weight_map(raw_type, type_weights, pos_weight):
    """Per-pixel loss weight map derived from a raw fire-type band.

    Background (type 0) gets weight 1.0; fire pixels (type > 0) default to
    ``pos_weight``; any type listed in ``type_weights`` overrides with its
    absolute weight. This lets specific fire types be up-weighted in the loss
    while the model stays binary fire/no-fire.

    Args:
        raw_type: integer tensor of raw fire-type values (e.g. im_viirs_type,
            values 0-4).
        type_weights: dict mapping fire-type value (int or str) to absolute
            per-pixel weight.
        pos_weight: weight applied to fire pixels whose type is not listed in
            ``type_weights``.

    Returns:
        A float32 tensor of per-pixel weights, same shape as ``raw_type``.
    """
    raw_type = tf.cast(raw_type, tf.int32)
    weight = tf.ones_like(raw_type, dtype=tf.float32)
    weight = tf.where(raw_type > 0,
                      tf.cast(pos_weight, tf.float32),
                      weight)
    for type_value, type_weight in type_weights.items():
        weight = tf.where(tf.equal(raw_type, int(type_value)),
                          tf.cast(type_weight, tf.float32),
                          weight)
    return weight

# --- confidence weighting from several fire products ------------------------
#
# The target is an OR of three imperfect detectors of the same latent event
# ("did this cell burn this year"): MCD64A1 burn scars, MOD14 active fire and
# VIIRS SNPP hotspots. They differ far more in what they MISS than in what they
# falsely claim, so a detection from any of them is strong, roughly
# interchangeable evidence, while a non-detection is weak evidence that varies a
# lot by product and by land cover. Treating the OR as a clean bit throws that
# structure away.
#
# Each pixel's evidence is summed as log-likelihood ratios under a latent-class
# (Hui-Walter / Dawid-Skene) measurement model whose parameters are fitted
# offline by scripts/analysis/fit_label_model.py and inlined into the config:
#
#   logit q = logit(pi_g)
#             + sum_r [ d_r log(s_rg/f_r) + (1-d_r) log((1-s_rg)/(1-f_r)) ]
#             + pair-dependence corrections + a co-detection |dDOY| term
#
# with s_rg the stratum-specific sensitivity, f_r the false-positive rate, and
# g a land-cover stratum. q is the posterior probability the cell burned.

_CONF_EPS = 1e-6


def _validate_confidence_config(cfg):
    """Eager checks on a 'confidence' sample_weight block; raises on nonsense.

    Runs at graph-construction time (not per batch) so a bad config fails at
    launch rather than producing silently wrong weights for a whole run.
    """
    products = cfg.get('products')
    if not products:
        raise ValueError("confidence sample_weight needs a non-empty 'products' list")
    n_strata = len(cfg['prior'])
    if n_strata < 1:
        raise ValueError("confidence sample_weight needs a non-empty 'prior'")
    strat = cfg.get('stratify')
    edges = (strat or {}).get('edges', [])
    if n_strata != len(edges) + 1:
        raise ValueError(
            f"'prior' has {n_strata} entries but 'stratify.edges' implies "
            f"{len(edges) + 1} strata")
    for value in cfg['prior']:
        if not 0.0 < value < 1.0:
            raise ValueError(f"prior {value} is not in (0, 1)")
    for prod in products:
        if not 0.0 < prod['fpr'] < 1.0:
            raise ValueError(f"{prod['name']}: fpr {prod['fpr']} is not in (0, 1)")
        if len(prod['sens']) != n_strata:
            raise ValueError(
                f"{prod['name']}: {len(prod['sens'])} sensitivities for "
                f"{n_strata} strata")
        for s in prod['sens']:
            if not 0.0 < s < 1.0:
                raise ValueError(f"{prod['name']}: sensitivity {s} is not in (0, 1)")
            if s <= prod['fpr']:
                raise ValueError(
                    f"{prod['name']}: sensitivity {s} <= fpr {prod['fpr']}, so a "
                    "detection would be evidence AGAINST fire")
    reliability = cfg.get('reliability')
    if reliability is not None and len(reliability) != n_strata:
        raise ValueError(
            f"'reliability' has {len(reliability)} entries for {n_strata} strata")
    for term in _doy_terms(cfg):
        if len(term['values']) != len(term['edges']) + 1:
            raise ValueError(
                "doy_llr 'values' must have one more entry than 'edges'")
        for pair in term['pairs']:
            if len(pair) != 2 or not all(0 <= i < len(products) for i in pair):
                raise ValueError(f"doy_llr pair {pair} is not a valid product index pair")


def _dilate_mask(mask, radius):
    """Max-pool a boolean mask with a (2*radius+1) square; identity at radius 0.

    Products sit on different native grids (MCD64 500 m, VIIRS 463 m, MOD14
    926.6 m against a ~555 m chip pixel) and MODIS scars sit ADJACENT to the
    active-fire detections that seeded them, so scoring agreement at exact pixel
    coincidence conflates registration and scale error with detection error.
    """
    if not radius:
        return mask
    x = tf.cast(mask, tf.float32)
    rank2 = len(x.shape) == 2
    if rank2:
        x = x[tf.newaxis, ...]
    x = tf.nn.max_pool2d(x[..., tf.newaxis], ksize=2 * int(radius) + 1,
                         strides=1, padding='SAME')[..., 0]
    if rank2:
        x = x[0]
    return x > 0.0


def _stratum_edges(strat):
    """Bin edges in the units the band will actually carry at loss time.

    Edges are configured in natural units (e.g. MapBiomas forest fraction in
    [0, 1]) because that is what the offline fit measured, but normalization
    runs BEFORE select_bands_transform, so a stratifier that is also a
    normalized model input arrives as a z-score. Mapping the edges once here is
    equivalent to de-normalizing every pixel and far cheaper.
    `resolve_stratifier_normalization` fills in 'normalized'.
    """
    edges = [float(e) for e in strat.get('edges', [])]
    norm = strat.get('normalized')
    if norm is None:
        return edges
    center, scale = float(norm['center']), float(norm['scale'])
    if scale == 0:
        return [e - center for e in edges]
    return [(e - center) / (scale + 1e-7) for e in edges]


def resolve_stratifier_normalization(sample_weight_config, normalize_list,
                                     robust_features, stats):
    """Fill in how the stratifier band is scaled at loss time, or raise.

    The band is read AFTER the normalizer has run, so silently comparing
    natural-unit edges against z-scores would put every pixel in one stratum and
    quietly disable stratification for a whole run. Rather than let that happen,
    resolve it from the same stats the normalizer used, and refuse if they are
    missing.

    Returns a shallow copy with stratify['normalized'] set (or the input
    unchanged when the band is not normalized).
    """
    cfg = sample_weight_config
    if (cfg or {}).get('mode') != 'confidence' or 'stratify' not in (cfg or {}):
        return cfg
    strat = cfg['stratify']
    name = strat['feature_name']
    if name not in set(normalize_list) or 'normalized' in strat:
        return cfg
    s = data_norm.get_norm_stats(stats, name)
    if not s:
        raise ValueError(
            f"confidence stratifier {name!r} is normalized by this config but "
            f"has no entry in the stats file, so its bin edges cannot be put on "
            f"the same scale. Give stratify.normalized {{center, scale}} "
            f"explicitly, or stratify on a band that is not normalized.")
    if name in set(robust_features or ()):
        center, scale = s['median'], (s.get('robust_scale') or s['stddev'])
    else:
        center, scale = s['mean'], s['stddev']
    out = dict(cfg)
    out['stratify'] = dict(strat, normalized={"center": float(center),
                                              "scale": float(scale)})
    return out


def _stratum_index(example, strat):
    """Per-pixel stratum index from a continuous band and a list of bin edges."""
    if strat is None:
        return None
    band = tf.cast(example[strat['feature_name']], tf.float32)
    idx = tf.zeros_like(band, dtype=tf.int32)
    for edge in _stratum_edges(strat):
        idx += tf.cast(band >= edge, tf.int32)
    return idx


def _per_stratum(values, idx, shape_ref):
    """Broadcast a per-stratum constant to a per-pixel tensor."""
    table = tf.constant(values, dtype=tf.float32)
    if idx is None:
        return tf.fill(tf.shape(shape_ref), table[0])
    return tf.gather(table, idx)


def _doy_terms(cfg):
    """Normalise cfg['doy_llr'] to a list of terms.

    A list lets each product pair carry its own |dDOY| table, which matters
    because MCD64A1 is seeded by MOD14 active fires -- their co-detection
    timing is partly an artefact of the algorithm, not independent corroboration
    -- while a VIIRS/MCD64 co-detection in the same week is real evidence.
    """
    doy = cfg.get('doy_llr')
    if doy is None:
        return []
    return [doy] if isinstance(doy, dict) else list(doy)


def _binned_lookup(delta, edges, values):
    """Piecewise-constant lookup: values[k] where edges[k-1] <= delta < edges[k]."""
    idx = tf.zeros_like(delta, dtype=tf.int32)
    for edge in edges:
        idx += tf.cast(delta >= float(edge), tf.int32)
    return tf.gather(tf.constant(values, dtype=tf.float32), idx)


def build_confidence_posterior(example, cfg):
    """Per-pixel posterior q that the cell burned, and the raw-OR union label.

    Args:
        example: the parsed feature dict, holding RAW day-of-year bands (the
            output prep binarizes a copy, so values here are untransformed).
        cfg: the 'confidence' sample_weight block (see _validate_confidence_config).

    Returns:
        (q, union) where q is float32 in (0, 1) and union is the bool OR of the
        undilated detections -- bit-identical to _combine_output_bands(..., 'any'),
        which is what the metrics and the hard eval label keep using.
    """
    products = cfg['products']
    idx = _stratum_index(example, cfg.get('stratify'))
    ref = tf.cast(example[products[0]['name']], tf.float32)

    llr = _per_stratum([math.log(p / (1.0 - p)) for p in cfg['prior']], idx, ref)

    raw = [tf.cast(example[p['name']], tf.float32) for p in products]
    hits = [band > 0.0 for band in raw]
    detected = [_dilate_mask(hit, p.get('dilate', 0))
                for hit, p in zip(hits, products)]

    for prod, det in zip(products, detected):
        f = float(prod['fpr'])
        pos = _per_stratum([math.log(s / f) for s in prod['sens']], idx, ref)
        neg = _per_stratum([math.log((1.0 - s) / (1.0 - f)) for s in prod['sens']],
                           idx, ref)
        llr += tf.where(det, pos, neg)

    # Pairwise dependence. MCD64A1 is partly DERIVED from MOD14 (the Collection 6
    # algorithm seeds its training samples and priors from a cumulative active-fire
    # composite), and MOD14/VIIRS share an early-afternoon overpass, so those
    # agreements are partly one vote counted twice and must be discounted.
    for pair in cfg.get('pair_llr', []):
        a, b = detected[pair['a']], detected[pair['b']]
        both = tf.logical_and(a, b)
        neither = tf.logical_and(tf.logical_not(a), tf.logical_not(b))
        llr += tf.where(both, float(pair.get('both', 0.0)),
                        tf.where(neither, float(pair.get('neither', 0.0)), 0.0))

    # Co-detection timing. Two products firing within the MCD64 8-day compositing
    # window is far stronger consensus than two firing six months apart, which in a
    # high-fire cell may be two unrelated fires. Undilated co-detection only, so
    # "whose day-of-year" is unambiguous.
    for term in _doy_terms(cfg):
        for a, b in term['pairs']:
            co = tf.logical_and(hits[a], hits[b])
            delta = tf.abs(raw[a] - raw[b])
            llr += tf.where(co, _binned_lookup(delta, term['edges'], term['values']),
                            tf.zeros_like(delta))

    union = hits[0]
    for hit in hits[1:]:
        union = tf.logical_or(union, hit)
    return tf.sigmoid(llr), union


def build_confidence_weight_map(example, cfg, pos_weight):
    """Per-pixel (loss weight, soft target) from several fire products.

    The loss we want per pixel is the pseudo-count form

        L = b * [ -P*q*log(p) - (1-q)*log(1-p) ]

    with P the pos_weight, q the posterior probability the cell burned and b a
    confidence in q. `weighted_bce` computes `(bce(target, pred) * weight).mean()`
    with ONE scalar per pixel, so P is folded into both returned tensors:

        target = P*q / (P*q + 1 - q)        weight = b * (P*q + 1 - q)

    which expands to exactly L. At q in {0, 1} and b = 1 this reduces to the
    current weighted_bce bit for bit (q=1 -> weight P, target 1; q=0 -> weight 1,
    target 0), so the feature is a strict superset of today's behaviour.
    `losses.deflate_probs(target, P)` still returns q exactly, so the area_ratio
    metric keeps its meaning.

    Two arms, selected by cfg['soft_label']:
      False  the target stays the hard union and b = P(the union bit is right)
             = q where the union fires, 1 - q elsewhere. This is the
             label-dependent-cost estimator for class-conditional label noise.
      True   the target is q itself and b defaults to 1 (optionally a per-stratum
             reliability), so the evidence lives in the target and is not counted
             twice.

    Returns:
        (weights, soft_target). soft_target is None in the hard-label arm, where
        it would equal the pipeline's existing union label.
    """
    q, union = build_confidence_posterior(example, cfg)
    conf = cfg.get('confidence', {})
    floor = float(conf.get('floor', 0.0))
    scale = float(conf.get('weight_scale', 1.0))
    soft_label = cfg.get('soft_label', False)

    if soft_label:
        target = q
        reliability = cfg.get('reliability')
        if reliability is None:
            b = tf.ones_like(q)
        else:
            b = _per_stratum(reliability,
                             _stratum_index(example, cfg.get('stratify')), q)
    else:
        target = tf.cast(union, tf.float32)
        b = tf.where(union, q, 1.0 - q)

    b = tf.maximum(b, floor) * scale
    p_w = tf.cast(pos_weight, tf.float32)
    denom = p_w * target + (1.0 - target)
    weights = b * denom
    if not soft_label:
        # target is already the pipeline's union label; emitting it again would
        # just duplicate `outputs`.
        return weights, None
    return weights, (p_w * target) / tf.maximum(denom, _CONF_EPS)

def _to_tuple_transform(
    example: Dict,
    input_feature_config: dict,
    output_feature_config: dict,
    sample_weight_config: Optional[dict] = None,
    pos_weight: float = 9.0,
):
    """Transform a parsed example into an (inputs, outputs[, weight[, soft]]) tuple.

    Returns:
        (inputs_dict, outputs_dict or outputs_tensor); plus a per-pixel
        sample_weight tensor when sample_weight_config is provided; plus a soft
        target when that config is a 'confidence' block with soft_label set.
        `outputs` is always the hard label, whatever the weighting.
    """

    # Input features first
    inputs = {}
    for feat_group in input_feature_config.keys():
        inputs[feat_group] = _single_feature_group_prep(
            example,
            input_feature_config[feat_group]
        )

    # Return outputs based on the combine mode and number of output bands
    prepped_outputs = _single_feature_group_prep(
        example,
        output_feature_config
    )
    combine = output_feature_config.get('combine')
    if combine is not None:
        # Several bands merged into one binary target (e.g. a two-sensor union)
        outputs = _combine_output_bands(prepped_outputs, combine)
    elif len(output_feature_config['feature_names']) == 1:
        # Single output: return as tensor
        outputs = prepped_outputs[...,0]
    else:
        # Multiple outputs: return as dict
        outputs = prepped_outputs

    if sample_weight_config is None:
        return inputs, outputs

    # Build a per-pixel loss weight map from the raw (untransformed) label bands.
    # The original `example` still holds raw values because the output prep
    # applies its binarizing transform to a copy (see apply_transforms).
    if sample_weight_config.get('mode') == 'confidence':
        weights, soft = build_confidence_weight_map(
            example, sample_weight_config, pos_weight)
        if soft is None:
            return inputs, outputs, weights
        # `outputs` stays the HARD union so metrics and the written prediction
        # rasters keep scoring against the frozen eval label; the soft target
        # rides along as a 4th element that only the loss reads.
        return inputs, outputs, weights, soft

    feature_name = sample_weight_config.get('feature_name', 'im_viirs_type')
    type_weights = sample_weight_config.get('type_weights', {})
    weights = build_type_weight_map(example[feature_name], type_weights, pos_weight)
    return inputs, outputs, weights

def select_bands_transform(
    dataset: tf.data.Dataset,
    input_feature_config: dict,
    output_feature_config: dict,
    sample_weight_config: Optional[dict] = None,
    pos_weight: float = 9.0,
) -> tf.data.Dataset:
    """Select input and output bands from a dataset of feature dicts, with optional transforms.

    Use this after merging datasets to split features into inputs/outputs.

    Args:
        dataset: A dataset yielding feature dicts (e.g., from dataset_from_dir or merge_datasets)
        input_feature_config:
        output_feature_config:
        sample_weight_config: optional per-pixel loss weighting block. Either
            the per-fire-type form ('feature_name', 'type_weights'), or
            {'mode': 'confidence', ...} to derive the weight (and optionally a
            soft target) from several fire products -- see
            build_confidence_weight_map.
        pos_weight: positive-class weight. Folded into the returned weights,
            since sample_weight replaces the loss's internal class weighting.
    Returns:
        A dataset yielding (inputs_dict, outputs_dict/tensor) 2-tuples; 3-tuples
        with a per-pixel sample_weight when sample_weight_config is given; or
        4-tuples that also carry a soft target when that config is a
        'confidence' block with soft_label set.
    """
    if (sample_weight_config or {}).get('mode') == 'confidence':
        # Eager, so a malformed measurement model fails at launch rather than
        # silently mis-weighting an entire run.
        _validate_confidence_config(sample_weight_config)

    def select_fn(example):
        return _to_tuple_transform(
            example, input_feature_config, output_feature_config,
            sample_weight_config=sample_weight_config, pos_weight=pos_weight)

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
        batch_size=4,
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
