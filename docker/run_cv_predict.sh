#!/usr/bin/env bash
# Submit CV-protocol predictions to the aic-predict Cloud Run job: one async execution per
# (fold, eval year) of out/cv/protocol.csv. Fetch the results with download_cv_preds.sh.
# Usage: [DATA_VERSION=..] [FOLDS=..] [YEARS=..] [FORCE=1] run_cv_predict.sh <arch> <stage>
# YEARS rows are off-protocol (never score them); don't rerun while executions are live.
# Always runs the aic-predict:latest image (TAG is not read here).
source "$(dirname "$0")/_cv_common.sh"

FORCE="${FORCE:-0}"
YEARS="${YEARS:-}"

pin_image aic-predict latest
load_rows run_cv_predict
if [ -n "$YEARS" ]; then
    echo "[run_cv_predict] YEARS override: $YEARS (not in $PROTOCOL; never score a predict-only year)"
fi
[ -n "${FOLDS:-}" ] && echo "[run_cv_predict] FOLDS filter: $FOLDS"
echo "[run_cv_predict] $ARCH/$STAGE: ${#ROWS[@]} (fold, year) rows, image $IMG"

n_sub=0
for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r fold_id year config_gs model_gs stats_gs predict_root <<<"$line"
    out_gs="$GS/preds/cv/$ARCH/${fold_id}_${year}/"
    data_dir="$GS/data/fullgrid_${DATA_VERSION}/allpreds_${year}/"

    # preds_mask.tif is uploaded last, so it marks a finished run.
    if [ "$FORCE" != "1" ]; then
        if compgen -G "$predict_root/$year/chips/out_*.tif" > /dev/null; then
            echo "[predict] $fold_id $year local chips exist, skip"
            continue
        fi
        if gsutil -q stat "${out_gs}preds_mask.tif"; then
            echo "[predict] $fold_id $year finished in GCS, skip (download it)"
            continue
        fi
    fi

    # UPLOAD_TILES=1: the scorer needs per-chip rasters for the within-chip decomposition.
    echo "[predict] $fold_id $year -> $out_gs"
    gcloud run jobs execute aic-predict --region="$REGION" --project="$PROJECT" --async \
      --update-env-vars=\
CONFIG_PATH=${config_gs},\
CHECKPOINT=${model_gs},\
DATA_DIR=${data_dir},\
STATS_PATH=${stats_gs},\
OUTPUT_URI=${out_gs},\
PROFILE_TEMPLATE=${PROFILE_TEMPLATE},\
UPLOAD_TILES=1,EDGE_CROP=0,MOSAIC=1,BATCH_SIZE=4,OMP_NUM_THREADS=8
    n_sub=$((n_sub + 1))
done

echo "[run_cv_predict] submitted $n_sub execution(s). Watch with:"
echo "  gcloud run jobs executions list --job=aic-predict --region=$REGION --project=$PROJECT"
echo "then download with:"
echo "  DATA_VERSION=$DATA_VERSION docker/download_cv_preds.sh $ARCH $STAGE"
