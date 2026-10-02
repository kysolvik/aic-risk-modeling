#!/usr/bin/env bash
# Download finished CV-protocol predictions (from run_cv_predict.sh) into the tree cv_collect_results.py scores.
# Usage: [DATA_VERSION=..] [MOSAIC_DIR=..] [YEARS=..] [FOLDS=..] download_cv_preds.sh <arch> <stage>
# A run is finished once preds_mask.tif exists; it is also copied to $MOSAIC_DIR/label_<year>.tif
# (skipped under YEARS: predict-only masks are placeholders). Safe to rerun.
set -euo pipefail
cd "/home/ksolvik/research/firesat/risk_modeling/aic-risk-modeling"

ARCH="${1:?usage: download_cv_preds.sh <arch> <stage>}"
STAGE="${2:?usage: download_cv_preds.sh <arch> <stage>}"

DATA_VERSION="${DATA_VERSION:-v3}"
GS="gs://aic-amazon"
PROTOCOL="out/cv/protocol.csv"
MOSAIC_DIR="${MOSAIC_DIR:-out/label_mosaics_${DATA_VERSION}}"

mapfile -t ROWS < <(python3 docker/cv_protocol_rows.py "$ARCH" "$STAGE" "$PROTOCOL")
if [ "${#ROWS[@]}" -eq 0 ]; then
    echo "[download_cv_preds] no rows for arch=$ARCH stage=$STAGE in $PROTOCOL" >&2
    exit 1
fi
mkdir -p "$MOSAIC_DIR"

pending=()
counts=()
for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r fold_id year _config _model _stats predict_root <<<"$line"
    out_gs="$GS/preds/cv/$ARCH/${fold_id}_${year}/"
    chips_dir="$predict_root/$year/chips"

    if ! compgen -G "$chips_dir/out_*.tif" > /dev/null; then
        if ! gsutil -q stat "${out_gs}preds_mask.tif"; then
            echo "[download] $fold_id $year not finished (no preds_mask.tif), pending"
            pending+=("$fold_id $year")
            continue
        fi
        echo "[download] $fold_id $year <- ${out_gs}chips/"
        mkdir -p "$chips_dir"
        gsutil -m -q cp "${out_gs}chips/*" "$chips_dir/"
    fi

    pred_tif="$predict_root/$year/preds_out.tif"
    if [ "${PRED_MOSAICS:-1}" = "1" ] && [ ! -f "$pred_tif" ]; then
        if gsutil -q cp "${out_gs}preds_out.tif" "$pred_tif"; then
            echo "[download] $fold_id $year <- ${out_gs}preds_out.tif"
        else
            echo "[download] $fold_id $year WARNING: no ${out_gs}preds_out.tif" >&2
        fi
    fi

    label_tif="$MOSAIC_DIR/label_${year}.tif"
    if [ -z "${YEARS:-}" ] && [ ! -f "$label_tif" ]; then
        gsutil -q cp "${out_gs}preds_mask.tif" "$label_tif"
    fi

    n=$(find "$chips_dir" -maxdepth 1 -name 'out_*.tif' | wc -l)
    counts+=("$n $fold_id $year")
done

echo
echo "[download_cv_preds] $ARCH/$STAGE chip counts (out_*.tif):"
if [ "${#counts[@]}" -gt 0 ]; then
    max=$(printf '%s\n' "${counts[@]}" | awk '{print $1}' | sort -n | tail -1)
    printf '%s\n' "${counts[@]}" | while read -r n f y; do
        flag=""; [ "$n" -lt "$max" ] && flag="   <-- SHORT (max $max)"
        printf '  %-28s %s  %6d%s\n' "$f" "$y" "$n" "$flag"
    done
fi
if [ "${#pending[@]}" -gt 0 ]; then
    echo "[download_cv_preds] ${#pending[@]} pending: ${pending[*]}"
    echo "  rerun this once they finish."
elif [ -n "${YEARS:-}" ]; then
    echo "[download_cv_preds] all ${#ROWS[@]} rows downloaded (YEARS override: off-protocol, not scored)."
else
    echo "[download_cv_preds] all ${#ROWS[@]} rows downloaded. Score with:"
    echo "  .venv/bin/python scripts/cross_validation/cv_collect_results.py --protocol $STAGE \\"
    echo "      --arch $ARCH --label_dir $MOSAIC_DIR --fetch_train_csv"
fi
