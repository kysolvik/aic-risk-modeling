# syntax=docker/dockerfile:1
#
# CPU-only prediction image for aic-risk-modeling.
#
# Built by Cloud Build (which has internet) and pulled by a Compute Engine VM
# that does not. At runtime the container reaches only Google APIs: Artifact
# Registry for its own image, GCS for the config, checkpoint, stats, TFRecords,
# and outputs.

########################### builder ###########################
FROM --platform=linux/amd64 python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=0

WORKDIR /src

# Dependency layer. Invalidated only by pyproject.toml / uv.lock, so editing a
# script does not rebuild ~2.5 GB of wheels. README.md is required because
# pyproject.toml declares readme = "README.md".
#
# `uv sync --locked` is what keeps this image CUDA-free: uv.lock pins
# torch==2.12.0+cpu from download.pytorch.org/whl/cpu via [tool.uv.sources] and
# resolves zero nvidia-* packages. A plain `pip install .[torch_cpu]` ignores
# that routing and pulls the CUDA wheel plus ~2.5 GB of CUDA runtime libraries.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --extra torch_cpu

# Project layer. --no-editable installs a real wheel into /opt/venv so the
# runtime stage needs no copy of src/.
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --extra torch_cpu

########################### runtime ###########################
FROM --platform=linux/amd64 python:3.11-slim-bookworm

# gdalbuildvrt / gdal_translate for the mosaic step. rasterio's wheels vendor
# their own libgdal and link it by RPATH, so this second GDAL does not conflict
# with it -- provided GDAL_DATA / PROJ_LIB / PROJ_DATA are never set globally,
# which would cross-wire the two stacks. Do not set them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gdal-bin \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TMPDIR=/scratch/tmp \
    TQDM_MININTERVAL=30

WORKDIR /app
COPY scripts/predict ./scripts/predict
# scripts/train/baseline_rf.py (tabular RF baseline) and
# scripts/analysis/compare_baselines.py (eval table) run in the same container
# as a Cloud Run job -- CPU-only, GCS in/out, no checkpoint.
COPY scripts/train ./scripts/train
COPY scripts/analysis ./scripts/analysis
COPY assets ./assets
COPY docker/entrypoint.sh /usr/local/bin/predict-entrypoint

# uid 1000 matches a typical host uid, which lets the local ADC bind-mount in
# the smoke test work without chmod.
RUN chmod 0755 /usr/local/bin/predict-entrypoint \
 && useradd --create-home --uid 1000 appuser \
 && mkdir -p /scratch/tmp \
 && chown -R appuser:appuser /scratch

USER appuser
ENV HOME=/home/appuser

ENTRYPOINT ["/usr/local/bin/predict-entrypoint"]
