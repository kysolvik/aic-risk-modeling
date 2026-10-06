# Add the static bands (geebeam_static_463m.py) to one year's dynamic export
# (geebeam_cutoff_463m.py), keyed by md_id. Run from the repo root.
# Usage: merge_static.sh <target_year> [extra patch_features.py args, e.g. --dry_run]
# Output: gs://woodwell-aic-fire-risk/data/fullgrid_v5/merged/allpreds_<target_year>
# Local scratch: ~22 GB under $WORK_DIR (default ~/merge_static_work), removed at the end;
# keep it off /tmp (RAM-backed).
set -euo pipefail
target_year=$1
shift
ROOT=gs://woodwell-aic-fire-risk/data/fullgrid_v5
WORK_DIR=${WORK_DIR:-$HOME/merge_static_work}/$target_year
trap 'rm -rf "$WORK_DIR"' EXIT

python scripts/preprocessing/build_static_year.py \
    --static_dir "$ROOT/static" \
    --target_year "$target_year" \
    --output_dir "$WORK_DIR/static_year"

python scripts/preprocessing/patch_features.py \
    --corrected_dir "$WORK_DIR/static_year" \
    --data_dirs "$ROOT/dynamic/allpreds_$target_year" \
    --output_root "$ROOT/merged" \
    --cache_dir "$WORK_DIR/cache" \
    "$@"
