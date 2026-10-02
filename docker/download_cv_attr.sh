#!/usr/bin/env bash
# Download the per-chip Shapley rasters from run_cv_attribute.sh and mosaic each year locally.
# Usage: [YEARS=..] [FOLDS=..] [OUT_TAG=..] download_cv_attr.sh <arch> <stage>
# Years short of EXPECTED_CHIPS are not mosaicked (MOSAIC_PARTIAL=1 overrides). Safe to rerun.
source "$(dirname "$0")/_cv_common.sh"

EXPECTED_CHIPS="${EXPECTED_CHIPS:-2556}"
OUT_TAG="${OUT_TAG:-}"
export PATH="$PWD/.venv/bin:$PATH"

load_rows download_cv_attr

short=()
for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r fold_id year _rest <<<"$line"
    src="$GS/attr/cv/$ARCH/${fold_id}_${year}${OUT_TAG}/chips/"
    dst="out/cv/attr/$ARCH/${fold_id}${OUT_TAG}/$year"
    mkdir -p "$dst/chips"
    if gsutil -q ls "${src}shap_*.tif" > /dev/null 2>&1; then
        echo "[download] $fold_id $year <- $src"
        gsutil -m -q cp -n "${src}shap_*.tif" "$dst/chips/"
    else
        echo "[download] $fold_id $year: no chips in GCS yet"
    fi
    n=$(find "$dst/chips" -maxdepth 1 -name 'shap_*.tif' | wc -l)
    echo "[download] $fold_id $year: $n / $EXPECTED_CHIPS chips"
    if [ "$n" -lt "$EXPECTED_CHIPS" ] && [ "${MOSAIC_PARTIAL:-0}" != "1" ]; then
        short+=("$fold_id $year ($n)")
        continue
    fi
    if [ -f "$dst/attr_shap.tif" ] && [ "${FORCE_MOSAIC:-0}" != "1" ]; then
        echo "[mosaic] $dst/attr_shap.tif exists, skip"
        continue
    fi
    bash scripts/predict/mosaic.sh "$dst/chips" "$dst" attr shap
done

if [ "${#short[@]}" -gt 0 ]; then
    echo "[download_cv_attr] not mosaicked, short of $EXPECTED_CHIPS chips: ${short[*]}"
fi
