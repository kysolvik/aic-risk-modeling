#!/usr/bin/env bash
# Build & push the custom Vertex training image (prebuilt PyTorch GPU + mamba).
# Uses Cloud Build, so no local Docker daemon or multi-GB base-image pull needed.
#
# Usage:   bash scripts/build_container.sh
# Override defaults via env vars, e.g.:  REGION=us-east1 TAG=torch2.4-cu118 bash scripts/build_container.sh
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-east1}"
REPO="${REPO:-aic-training}"          # Artifact Registry docker repo
IMAGE="${IMAGE:-aic-mamba}"
TAG="${TAG:-torch2.4-cu118}"          # bump when the wheel or base image changes
URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE}:${TAG}"

echo "Project: ${PROJECT}"
echo "Image:   ${URI}"

# One-time repo creation (no-op if it already exists).
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${REGION}" \
  --description="AIC training images" 2>/dev/null || true

gcloud builds submit --tag "${URI}" --timeout=1800s .

echo
echo "Pushed: ${URI}"
echo "Launch a job with it:"
echo "  python scripts/train_vertex.py <gs://.../config.json> <display_name> --container_uri ${URI}"
