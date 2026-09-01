#!/usr/bin/env bash
# Launch the containerized Shapley attribution job, one year at a time.
set -euo pipefail

REGION='us-east1'
PROJECT='macedo-lab-general-9051'

MODEL=v44
DRIVERS=gs://aic-amazon/configs/attribution_drivers_v44.json
# Monte-Carlo permutation count for approximate Shapley (~N*S forwards instead
# of 2^N). Leave empty for exact enumeration; set e.g. 4 if s/chip is too high.
SHAPLEY_SAMPLES=

for y in 2023 2024 2025 2026;
do
    env_vars="MODE=attribute,SHAPLEY=1,DRIVERS=${DRIVERS}"
    env_vars="${env_vars},CONFIG_PATH=gs://aic-amazon/configs/mtsvit_test_${MODEL}.json"
    env_vars="${env_vars},CHECKPOINT=gs://aic-amazon/models/mtsvit_test_${MODEL}.pt"
    env_vars="${env_vars},DATA_DIR=gs://aic-amazon/data/fullgrid_v2/allpreds_${y}/"
    env_vars="${env_vars},OUTPUT_URI=gs://aic-amazon/attr/mtsvit_${MODEL}_${y}/"
    env_vars="${env_vars},EDGE_CROP=0,INVERT_YRES=1,MOSAIC=1,BATCH_SIZE=4,OMP_NUM_THREADS=8"
    if [ -n "$SHAPLEY_SAMPLES" ]; then
        env_vars="${env_vars},SHAPLEY_SAMPLES=${SHAPLEY_SAMPLES}"
    fi

    gcloud run jobs execute aic-attribute --region=$REGION --project=$PROJECT \
        --update-env-vars="$env_vars"
done
