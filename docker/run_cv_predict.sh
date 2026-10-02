#!/usr/bin/env bash
# Submit CV-protocol predictions to the `aic-predict` Cloud Run job, one async
# execution per (fold, eval year). Does NOT wait or download: once the
# executions finish, fetch the tiles with
#
#   docker/download_cv_preds.sh <arch> <stage>
#
# Usage:
#   docker/run_cv_predict.sh <arch> <stage>
#     e.g. DATA_VERSION=v3_patched MOSAIC_DIR=out/label_mosaics_v3p_union4 \
#            docker/run_cv_predict.sh unet_v3p_union4 folds
#
# Rows come from out/cv/protocol.csv (via cv_protocol_rows.py), so this stays in
# sync with cv_make_folds.py.
#
# IMAGE: the job is pinned to aic-predict:$TAG (default: short HEAD). Build it
# first, or this refuses to run -- the image tracks src/ at build time, not the
# training sdist, and a stale image can load a checkpoint cleanly and still
# mispredict (see docker/README.md). Run `uv lock` first -- the Dockerfile uses
# `uv sync --locked`, so a stale uv.lock fails the build:
#   uv lock
#   gcloud builds submit --config=cloudbuild.yaml --region=us-east1 \
#     --project=macedo-lab-general-9051 --substitutions=_TAG=$(git rev-parse --short HEAD) .
#
# Notes:
# - UPLOAD_TILES=1 is required: the scorer globs per-chip out_/mask_ rasters for
#   the within-chip decomposition; the mosaic alone cannot give it.
# - STATS_PATH is passed explicitly (the pooled per-fold stats) to avoid the
#   normalization-skew trap, even though the v3 configs also embed stats_path.
# - Skips a (fold, year) whose chips are already local, or whose preds_mask.tif
#   is already in GCS (the container writes the mosaic only after predict.py
#   succeeds, and uploads it after the chips, so it marks a finished run).
#   FORCE=1 resubmits regardless. An execution that is still RUNNING has no
#   mosaic yet and WOULD be resubmitted -- don't rerun this while jobs are live.
# - YEARS overrides the protocol's eval years for every fold of <arch>/<stage>
#   (space/;/,-separated), e.g. a predict-only forecast year:
#     DATA_VERSION=v3_patched YEARS=2026 docker/run_cv_predict.sh <arch> final
#   These years are not in protocol.csv, so the scorer never sees them; fetch
#   them with the same YEARS (and FOLDS) set on download_cv_preds.sh. A
#   predict-only year's mask_*/preds_mask are placeholder labels (copies of
#   Y-1) -- never score them.
# - FOLDS restricts to these fold_ids, e.g. the last CV fold on the test years:
#     DATA_VERSION=v3_patched FOLDS=fwdpair_2022 YEARS="2024 2025" \
#       docker/run_cv_predict.sh <arch> folds
#   YEARS/FOLDS are applied in cv_protocol_rows.py, so both scripts agree.
set -euo pipefail
cd "/home/ksolvik/research/firesat/risk_modeling/aic-risk-modeling"

ARCH="${1:?usage: run_cv_predict.sh <arch> <stage>}"
STAGE="${2:?usage: run_cv_predict.sh <arch> <stage>}"

REGION="${REGION:-us-east1}"
PROJECT="${PROJECT:-macedo-lab-general-9051}"
DATA_VERSION="${DATA_VERSION:-v3}"      # data bucket suffix: fullgrid_<DATA_VERSION>
TAG="latest" #"${TAG:-$(git rev-parse --short HEAD)}"
FORCE="${FORCE:-0}"
YEARS="${YEARS:-}"                      # override eval years (see header)
GS="gs://aic-amazon"
PROTOCOL="out/cv/protocol.csv"
# Output CRS + pixel size: md_x/md_y are MODIS sinusoidal metres (463.3m), north-up.
PROFILE_TEMPLATE=/app/assets/example_v3.tif

# Pin the job to the immutable tag, then confirm it took.
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
