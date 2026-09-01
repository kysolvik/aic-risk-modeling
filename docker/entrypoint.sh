#!/usr/bin/env bash
# Container entrypoint: predict|attribute -> mosaic -> upload to GCS.
#
# Configured entirely by environment variables, because that is what
# `gcloud compute instances create-with-container --container-env` passes.
# Any extra arguments are forwarded verbatim to the script.
#
# MODE (predict) picks the script: `predict` -> predict.py, `attribute` ->
# attribute.py (per-driver Shapley/OAT rasters). The two share every flag below;
# attribute adds the DRIVERS/SHAPLEY/... vars, which are ignored in predict mode.
#
# Required: CONFIG_PATH CHECKPOINT DATA_DIR OUTPUT_URI
# Optional: MODE (predict) STATS_PATH TFRECORD_PATTERN MAX_CHIPS BATCH_SIZE SEED
#           EDGE_CROP (0) INVERT_YRES (1) MOSAIC (1) MOSAIC_NAME (preds)
#           UPLOAD_TILES (0) SCRATCH_DIR (/scratch/chips)
# Attribute-only: DRIVERS (gs:// driver spec) SHAPLEY (0) SHAPLEY_SAMPLES
#           POS_WEIGHT WRITE_MASK (0)
set -euo pipefail

: "${CONFIG_PATH:?must be set}"
: "${CHECKPOINT:?must be set}"
: "${DATA_DIR:?must be set}"
: "${OUTPUT_URI:?must be set}"

SCRATCH="${SCRATCH_DIR:-/scratch/chips}"
MOSAIC_DIR="${SCRATCH%/*}/mosaic"
mkdir -p "$SCRATCH"

# Upload on ANY exit path so a crash or a Spot preemption still leaves the
# finished chips in GCS instead of dying with the VM.
uploaded=0
sync_out() {
    [ "$uploaded" = "1" ] && return 0
    uploaded=1
    if [ "${UPLOAD_TILES:-0}" = "1" ]; then
        python /app/scripts/predict/sync_to_gcs.py \
            "$SCRATCH" "${OUTPUT_URI%/}/chips/" || true
    else
        echo "[entrypoint] UPLOAD_TILES=0, keeping the ~2 files/chip local"
    fi
    if [ -d "$MOSAIC_DIR" ]; then
        python /app/scripts/predict/sync_to_gcs.py \
            "$MOSAIC_DIR" "${OUTPUT_URI%/}/" || true
    fi
}
trap sync_out EXIT

MODE="${MODE:-predict}"
case "$MODE" in
    predict)   SCRIPT=/app/scripts/predict/predict.py ;;
    attribute) SCRIPT=/app/scripts/predict/attribute.py ;;
    *) echo "[entrypoint] unknown MODE='$MODE' (want predict|attribute)" >&2
       exit 2 ;;
esac

args=(--config_path "$CONFIG_PATH"
      --checkpoint "$CHECKPOINT"
      --data_dir "$DATA_DIR"
      --output_dir "$SCRATCH"
      --edge_crop "${EDGE_CROP:-0}")

# Use `if` blocks, not `[ ... ] && ...`: under `set -e` a trailing false test
# in a && chain is the script's exit status and would abort the run. Every flag
# below is accepted by BOTH scripts.
if [ -n "${STATS_PATH:-}" ];       then args+=(--stats_path "$STATS_PATH"); fi
if [ -n "${TFRECORD_PATTERN:-}" ]; then args+=(--tfrecord_pattern "$TFRECORD_PATTERN"); fi
if [ -n "${MAX_CHIPS:-}" ];        then args+=(--max_chips "$MAX_CHIPS"); fi
if [ -n "${BATCH_SIZE:-}" ];       then args+=(--batch_size "$BATCH_SIZE"); fi
if [ -n "${SEED:-}" ];             then args+=(--seed "$SEED"); fi
if [ "${INVERT_YRES:-1}" = "1" ];  then args+=(--invert_yres); fi

# Which per-chip prefixes the mosaic step should stitch. predict writes out/mask;
# attribute writes a single shap_ (Shapley) or attr_ (OAT) raster.
mosaic_prefixes="out mask"
if [ "$MODE" = "attribute" ]; then
    if [ -n "${DRIVERS:-}" ];         then args+=(--drivers "$DRIVERS"); fi
    if [ -n "${POS_WEIGHT:-}" ];      then args+=(--pos_weight "$POS_WEIGHT"); fi
    if [ "${SHAPLEY:-0}" = "1" ];     then args+=(--shapley); fi
    if [ -n "${SHAPLEY_SAMPLES:-}" ]; then args+=(--shapley_samples "$SHAPLEY_SAMPLES"); fi
    if [ "${WRITE_MASK:-0}" = "1" ];  then args+=(--write_mask); fi
    if [ "${SHAPLEY:-0}" = "1" ]; then mosaic_prefixes="shap"; else mosaic_prefixes="attr"; fi
fi

echo "[entrypoint] $MODE: $SCRIPT ${args[*]} $*"
python "$SCRIPT" "${args[@]}" "$@"

if [ "${MOSAIC:-1}" = "1" ]; then
    bash /app/scripts/predict/mosaic.sh \
        "$SCRATCH" "$MOSAIC_DIR" "${MOSAIC_NAME:-preds}" "$mosaic_prefixes"
fi

sync_out
echo "[entrypoint] done"
