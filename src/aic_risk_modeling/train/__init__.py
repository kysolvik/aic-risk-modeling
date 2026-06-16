"""Train

Utilities for building dataset and training models
"""

from .models import (
    get_unet,
    get_unet_lite,
    get_mlp,
    get_multi_scale_mlp_head,
    get_convlstm,
    get_convlstm_bottleneck,
    get_simple_convlstm,
    get_lstm,
    get_identity,
    get_mlp_for_fusion,
    get_transformer,
    decoder_fusion,
    decoder_mtsvit
)

from .data_loader import (
    load_schema_from_gcs,
    schema_to_feature_spec,
    dataset_from_dir,
    dataset_from_gcs,
    select_bands_transform,
    merge_datasets,
    apply_transforms,
    build_merged_dataset,
)

from .data_norm import (
    load_stats_from_text,
    load_stats_json,
    get_norm_stats,
    create_normalizer,
    get_normalize_list
)

from .data_stats import (
    compute_stats,
    write_stats,
)

from .losses import (
    weighted_bce,
    weighted_bce_dice,
    get_loss
)

from .metrics import (
    SegmentationMetrics,
)

from .trainer import (
    build_model,
    build_all_models,
    run,
    build_decoder,
    save_model,
    load_model
)

__all__ = [
    "load_schema_from_gcs",
    "schema_to_feature_spec",
    "dataset_from_dir",
    "dataset_from_gcs",
    "select_bands_transform",
    "merge_datasets",
    "apply_transforms",
    "get_unet",
    "get_unet_lite",
    "get_mlp",
    "get_mlp_for_fusion",
    "get_multi_scale_mlp_head",
    "get_convlstm",
    "get_simple_convlstm",
    "load_stats_from_text",
    "load_stats_json",
    "get_norm_stats",
    "get_normalize_list",
    "create_normalizer",
    "compute_stats",
    "write_stats",
    "build_merged_dataset",
    "weighted_bce",
    "weighted_bce_dice",
    "get_loss",
    "get_lstm",
    "get_convlstm_bottleneck",
    "get_transformer",
    "decoder_fusion",
    "decoder_mtsvit",
    "get_identity",
    "SegmentationMetrics",
    "build_model",
    "build_all_models",
    "build_decoder",
    "save_model",
    "load_model",
    "run"
]
