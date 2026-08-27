#!/usr/bin/env bash
# Container entrypoint: predict -> mosaic -> upload to GCS.
#
# Configured entirely by environment variables, because that is what
# `gcloud compute instances create-with-container --container-env` passes.
# Any extra arguments are forwarded verbatim to predict.py.
#
# Required: CONFIG_PATH CHECKPOINT DATA_DIR OUTPUT_URI
# Optional: STATS_PATH TFRECORD_PATTERN MAX_CHIPS BATCH_SIZE SEED
#           EDGE_CROP (0) INVERT_YRES (1) MOSAIC (1) MOSAIC_NAME (preds)
#           UPLOAD_TILES (0) SCRATCH_DIR (/scratch/chips)
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

args=(--config_path "$CONFIG_PATH"
      --checkpoint "$CHECKPOINT"
      --data_dir "$DATA_DIR"
      --output_dir "$SCRATCH"
      --edge_crop "${EDGE_CROP:-0}")

# Use `if` blocks, not `[ ... ] && ...`: under `set -e` a trailing false test
# in a && chain is the script's exit status and would abort the run.
if [ -n "${STATS_PATH:-}" ];       then args+=(--stats_path "$STATS_PATH"); fi
if [ -n "${TFRECORD_PATTERN:-}" ]; then args+=(--tfrecord_pattern "$TFRECORD_PATTERN"); fi
if [ -n "${MAX_CHIPS:-}" ];        then args+=(--max_chips "$MAX_CHIPS"); fi
if [ -n "${BATCH_SIZE:-}" ];       then args+=(--batch_size "$BATCH_SIZE"); fi
if [ -n "${SEED:-}" ];             then args+=(--seed "$SEED"); fi
if [ "${INVERT_YRES:-1}" = "1" ];  then args+=(--invert_yres); fi

echo "[entrypoint] predict.py ${args[*]} $*"
python /app/scripts/predict/predict.py "${args[@]}" "$@"

if [ "${MOSAIC:-1}" = "1" ]; then
    bash /app/scripts/predict/mosaic.sh \
        "$SCRATCH" "$MOSAIC_DIR" "${MOSAIC_NAME:-preds}"
fi

sync_out
echo "[entrypoint] done"
