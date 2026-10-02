#!/usr/bin/env bash
# Submit exact-Shapley driver attribution for a CV-protocol checkpoint to the aic-attribute
# Cloud Run job, split per year into parallel executions by tfrecord shard (~19 h/year unsplit).
# Usage: [DATA_VERSION=..] [YEARS=..] [GROUP_IDS=..] [OUT_TAG=..] run_cv_attribute.sh <arch> <stage>
# Chips upload even on failure; fetch and mosaic with download_cv_attr.sh (same OUT_TAG).
source "$(dirname "$0")/_cv_common.sh"

TAG="${TAG:-latest}"
DRIVERS="${DRIVERS:-gs://aic-amazon/configs/attribution_drivers_v3p_yeargain.json}"
GROUP_IDS="${GROUP_IDS:-}"
OUT_TAG="${OUT_TAG:-}"

# Last digit of the shard index, size-balanced per year (fullgrid_v3_patched).
declare -A GROUPS_BY_YEAR=(
    [2024]="469 28 03 157"
    [2025]="07 249 56 138"
    [2023]="04 25 19 3678"
    [2026]="045 17 239 68"
)

gsutil -q stat "$DRIVERS" || { echo "REFUSING: driver spec $DRIVERS not in GCS" >&2; exit 1; }
# Older images skip unknown spec keys, so a year_terms spec would silently run as the plain one.
if gsutil cat "$DRIVERS" | grep -q '"year_terms"' && [ "$TAG" = "latest" ]; then
    echo "REFUSING: $DRIVERS has year_terms; pin TAG to an image built with year_terms support" >&2
    exit 1
fi

pin_image aic-attribute "$TAG"
load_rows run_cv_attribute

# Validate every year's groups before submitting anything.
for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r fold_id year _rest <<<"$line"
    groups="${GROUPS_BY_YEAR[$year]:-}"
    [ -n "$groups" ] || { echo "REFUSING: no GROUPS_BY_YEAR entry for $year" >&2; exit 1; }
    data_dir="$GS/data/fullgrid_${DATA_VERSION}/allpreds_${year}/"
    LISTING="$(gsutil ls "${data_dir}*.tfrecord.gz")" python3 - "$year" "$groups" <<'PY' || exit 1
import os, re, sys
year, groups = sys.argv[1], sys.argv[2].split()
digits = [m.group(1)[-1] for l in os.environ["LISTING"].split()
          if (m := re.search(r"full-(\d+)-of-\d+\.tfrecord\.gz$", l.strip()))]
hits = {d: sum(d in g for g in groups) for d in digits}
bad = sorted({d for d, n in hits.items() if n != 1})
if not digits or bad:
    sys.exit(f"REFUSING: {year}: shard last-digits {bad} match != 1 group "
             f"({len(digits)} shards, groups {groups})")
print(f"[run_cv_attribute] {year}: {len(digits)} shards, groups {groups} OK")
PY
done

echo "[run_cv_attribute] $ARCH/$STAGE: ${#ROWS[@]} year(s), image $IMG, drivers $DRIVERS"
n_sub=0
for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r fold_id year config_gs model_gs stats_gs _predict_root <<<"$line"
    read -ra groups <<<"${GROUPS_BY_YEAR[$year]}"
    out_gs="$GS/attr/cv/$ARCH/${fold_id}_${year}${OUT_TAG}/"
    data_dir="$GS/data/fullgrid_${DATA_VERSION}/allpreds_${year}/"
    for gi in "${!groups[@]}"; do
        if [ -n "$GROUP_IDS" ] && [[ " $GROUP_IDS " != *" $gi "* ]]; then
            continue
        fi
        pattern="full-000?[${groups[$gi]}]-of-*.tfrecord.gz"
        echo "[attribute] $fold_id $year group $gi ($pattern) -> ${out_gs}chips/"
        gcloud run jobs execute aic-attribute --region="$REGION" --project="$PROJECT" --async \
          --update-env-vars=\
MODE=attribute,SHAPLEY=1,\
DRIVERS=${DRIVERS},\
CONFIG_PATH=${config_gs},\
CHECKPOINT=${model_gs},\
DATA_DIR=${data_dir},\
STATS_PATH=${stats_gs},\
OUTPUT_URI=${out_gs},\
TFRECORD_PATTERN=${pattern},\
PROFILE_TEMPLATE=${PROFILE_TEMPLATE},\
UPLOAD_TILES=1,MOSAIC=0,EDGE_CROP=0,BATCH_SIZE=4,OMP_NUM_THREADS=8
        n_sub=$((n_sub + 1))
    done
done

echo "[run_cv_attribute] submitted $n_sub execution(s). Watch with:"
echo "  gcloud run jobs executions list --job=aic-attribute --region=$REGION --project=$PROJECT"
echo "then download + mosaic with:"
echo "  YEARS=\"\$YEARS\" OUT_TAG=$OUT_TAG docker/download_cv_attr.sh $ARCH $STAGE"
