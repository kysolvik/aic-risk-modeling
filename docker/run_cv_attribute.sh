#!/usr/bin/env bash
# Submit Shapley driver attribution for a CV-protocol checkpoint to the
# `aic-attribute` Cloud Run job, split into parallel executions per year by
# tfrecord shard. Does NOT wait or download: once the executions finish, fetch
# and mosaic with
#
#   docker/download_cv_attr.sh <arch> <stage>
#
# Usage:
#   docker/run_cv_attribute.sh <arch> <stage>
#     e.g. DATA_VERSION=v3_patched YEARS="2024 2025 2023" \
#            docker/run_cv_attribute.sh factored_v3p_union4_monthlyattn_wide_yeargain final
#
# Why shards: exact Shapley over N driver groups is 2^N forwards per chip (32 for
# the 5-group v3p spec, measured 27 s/chip on 8 CPU threads, 2026-09-30), so one
# 8-vCPU task needs ~19 h per 2556-chip year. Each execution here gets the shards
# whose index ends in one of a set of digits (a TFRECORD_PATTERN glob, no image
# change); the digit sets are size-balanced per year from the shard byte counts
# (largest group <= 26% of a year at 4 groups). Shard counts differ by year
# (2023: 50, 2024: 54, 2025: 49, 2026: 61), so a year without a GROUPS_<year> entry is
# refused. Before submitting, the groups are checked against a GCS listing:
# every shard must match exactly one group.
#
# Notes:
# - Executions are submitted in YEARS order, so if the regional CPU quota bites,
#   the last year is what waits. 4 groups x 3 years = 12 x 8 vCPU.
# - UPLOAD_TILES=1 / MOSAIC=0: every execution uploads its own shap_ chips to
#   <out>/chips/ (chip names are unique, so groups never collide); no execution
#   sees the whole grid, so the mosaic is built locally by download_cv_attr.sh.
# - Chips are uploaded on exit (entrypoint trap), so a crashed or timed-out
#   execution still leaves its partial chips; download_cv_attr.sh flags short years.
# - GROUP_IDS (space-separated 0-based indices) resubmits only those groups of
#   each year, e.g. after one execution failed. Don't rerun this while executions
#   are live -- there is no per-group completion marker to skip on.
# - Bands are DEFLATED probabilities (risk, shapley_<driver>..., residual,
#   baseline). With the default spec the gamma year term and per-location year
#   gain are not players (md_year / md_single), so they sit in
#   risk_all_drivers_baseline; a spec with a `year_terms` block (e.g.
#   attribution_drivers_v3p_yeargain_yearsplit.json) hands their SOI /
#   prev-burn parts to climate_weather / fire_history instead.
# - OUT_TAG (e.g. _yearsplit) is appended to each <fold>_<year> output dir, so a
#   second driver spec never overwrites an earlier run; pass the same OUT_TAG to
#   download_cv_attr.sh.
set -euo pipefail
cd "/home/ksolvik/research/firesat/risk_modeling/aic-risk-modeling"

ARCH="${1:?usage: run_cv_attribute.sh <arch> <stage>}"
STAGE="${2:?usage: run_cv_attribute.sh <arch> <stage>}"

REGION="${REGION:-us-east1}"
PROJECT="${PROJECT:-macedo-lab-general-9051}"
DATA_VERSION="${DATA_VERSION:-v3_patched}"
TAG="${TAG:-latest}"
DRIVERS="${DRIVERS:-gs://aic-amazon/configs/attribution_drivers_v3p_yeargain.json}"
GROUP_IDS="${GROUP_IDS:-}"
OUT_TAG="${OUT_TAG:-}"
GS="gs://aic-amazon"
PROTOCOL="out/cv/protocol.csv"
PROFILE_TEMPLATE=/app/assets/example_v3.tif   # v3 grid, north-up, INVERT_YRES=0

# Last digit of the shard index, size-balanced per year (fullgrid_v3_patched).
declare -A GROUPS_BY_YEAR=(
    [2024]="469 28 03 157"
    [2025]="07 249 56 138"
    [2023]="04 25 19 3678"
    [2026]="045 17 239 68"     # predict-only year: 61 shards, 25/25/26/25%
)

gsutil -q stat "$DRIVERS" || { echo "REFUSING: driver spec $DRIVERS not in GCS" >&2; exit 1; }
# An image older than the year_terms support ignores the block (unknown spec keys
# are skipped), so a year-split spec would silently run as the plain one.
if gsutil cat "$DRIVERS" | grep -q '"year_terms"' && [ "$TAG" = "latest" ]; then
    echo "REFUSING: $DRIVERS has year_terms; pin TAG to an image built with year_terms support" >&2
    exit 1
fi

IMG=$REGION-docker.pkg.dev/$PROJECT/aic-containers/aic-predict:$TAG
gcloud run jobs update aic-attribute --region="$REGION" --project="$PROJECT" --image="$IMG"
CURRENT=$(gcloud run jobs describe aic-attribute --region="$REGION" --project="$PROJECT" \
            --format='value(spec.template.spec.template.spec.containers[0].image)')
if [ "$CURRENT" != "$IMG" ]; then
    echo "REFUSING: job image is '$CURRENT', expected '$IMG'." >&2
    exit 1
fi

mapfile -t ROWS < <(python3 docker/cv_protocol_rows.py "$ARCH" "$STAGE" "$PROTOCOL")
if [ "${#ROWS[@]}" -eq 0 ]; then
    echo "[run_cv_attribute] no rows for arch=$ARCH stage=$STAGE in $PROTOCOL" >&2
    exit 1
fi

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
UPLOAD_TILES=1,MOSAIC=0,EDGE_CROP=0,INVERT_YRES=0,BATCH_SIZE=4,OMP_NUM_THREADS=8
        n_sub=$((n_sub + 1))
    done
done

echo "[run_cv_attribute] submitted $n_sub execution(s). Watch with:"
echo "  gcloud run jobs executions list --job=aic-attribute --region=$REGION --project=$PROJECT"
echo "then download + mosaic with:"
echo "  YEARS=\"\$YEARS\" OUT_TAG=$OUT_TAG docker/download_cv_attr.sh $ARCH $STAGE"
