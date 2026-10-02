#!/usr/bin/env bash
# Submit CV-protocol predictions to the aic-predict Cloud Run job: one async execution per
# (fold, eval year) of out/cv/protocol.csv. Fetch the results with download_cv_preds.sh.
# Usage: [DATA_VERSION=v3_patched] [FOLDS=..] [YEARS=..] [FORCE=1] run_cv_predict.sh <arch> <stage>
# YEARS rows are off-protocol (never score them); don't rerun while executions are live.
set -euo pipefail
cd "/home/ksolvik/research/firesat/risk_modeling/aic-risk-modeling"

ARCH="${1:?usage: run_cv_predict.sh <arch> <stage>}"
STAGE="${2:?usage: run_cv_predict.sh <arch> <stage>}"

REGION="${REGION:-us-east1}"
PROJECT="${PROJECT:-macedo-lab-general-9051}"
DATA_VERSION="${DATA_VERSION:-v3}"
TAG="latest" #"${TAG:-$(git rev-parse --short HEAD)}"
FORCE="${FORCE:-0}"
YEARS="${YEARS:-}"
GS="gs://aic-amazon"
PROTOCOL="out/cv/protocol.csv"
# Output CRS + pixel size: md_x/md_y are MODIS sinusoidal metres (463.3m), north-up.
PROFILE_TEMPLATE=/app/assets/example_v3.tif

# The image tracks src/ at build time; rebuild it (uv lock, gcloud builds submit) after src changes.
IMG=$REGION-docker.pkg.dev/$PROJECT/aic-containers/aic-predict:$TAG
gcloud run jobs update aic-predict --region="$REGION" --project="$PROJECT" --image="$IMG"
CURRENT=$(gcloud run jobs describe aic-predict --region="$REGION" --project="$PROJECT" \
            --format='value(spec.template.spec.template.spec.containers[0].image)')
if [ "$CURRENT" != "$IMG" ]; then
    echo "REFUSING: job image is '$CURRENT', expected '$IMG'." >&2
    exit 1
fi

mapfile -t ROWS < <(python3 docker/cv_protocol_rows.py "$ARCH" "$STAGE" "$PROTOCOL")
if [ "${#ROWS[@]}" -eq 0 ]; then
    echo "[run_cv_predict] no rows for arch=$ARCH stage=$STAGE in $PROTOCOL" >&2
    exit 1
fi
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
echo "  DATA_VERSION=$DATA_VERSION MOSAIC_DIR=\${MOSAIC_DIR:-out/label_mosaics_$DATA_VERSION} docker/download_cv_preds.sh $ARCH $STAGE"
