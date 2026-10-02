# Shared setup for the docker/*_cv_*.sh scripts; sourced, so it sees the caller's <arch> <stage>.
# Defines repo-root cwd, defaults, load_rows (protocol rows -> ROWS) and pin_image (sets IMG).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARCH="${1:?usage: $(basename "$0") <arch> <stage>}"
STAGE="${2:?usage: $(basename "$0") <arch> <stage>}"

REGION="${REGION:-us-east1}"
PROJECT="${PROJECT:-macedo-lab-general-9051}"
DATA_VERSION="${DATA_VERSION:-v3_patched}"
GS="gs://aic-amazon"
PROTOCOL="out/cv/protocol.csv"
# Output CRS + pixel size: md_x/md_y are MODIS sinusoidal metres (463.3m), north-up.
PROFILE_TEMPLATE=/app/assets/example_v3.tif

# load_rows NAME: one TSV row per (fold, year) of $ARCH/$STAGE into ROWS; exit if none.
load_rows() {
    mapfile -t ROWS < <(python3 docker/cv_protocol_rows.py "$ARCH" "$STAGE" "$PROTOCOL")
    if [ "${#ROWS[@]}" -eq 0 ]; then
        echo "[$1] no rows for arch=$ARCH stage=$STAGE in $PROTOCOL" >&2
        exit 1
    fi
}

# pin_image JOB TAG: point the Cloud Run job at aic-predict:TAG, then confirm it took.
# The image tracks src/ at build time; rebuild it (uv lock, gcloud builds submit) after src changes.
pin_image() {
    IMG=$REGION-docker.pkg.dev/$PROJECT/aic-containers/aic-predict:$2
    gcloud run jobs update "$1" --region="$REGION" --project="$PROJECT" --image="$IMG"
    local current
    current=$(gcloud run jobs describe "$1" --region="$REGION" --project="$PROJECT" \
                --format='value(spec.template.spec.template.spec.containers[0].image)')
    if [ "$current" != "$IMG" ]; then
        echo "REFUSING: job image is '$current', expected '$IMG'." >&2
        exit 1
    fi
}
