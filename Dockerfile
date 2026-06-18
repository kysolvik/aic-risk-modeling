# syntax=docker/dockerfile:1
#
# Custom Vertex AI training image = the prebuilt PyTorch GPU container + the
# mamba-ssm / causal-conv1d CUDA wheels that `decoder_ssm` (SSMFusion) needs.
#
# Why a custom image: mamba-ssm and causal-conv1d are sdist-only on PyPI, so a
# plain `pip install` compiles from source (needs nvcc + --no-build-isolation)
# and will not install via Vertex's sdist core-deps path. We instead install the
# matching *prebuilt* wheels from their GitHub releases. The wheel tags
# (cuXXX / torchY.Z / cp3XX / cxx11abi) MUST match this base image's torch build.
#
# STEP 0 - confirm the base image's torch build, then pick wheels to match:
#   docker run --rm <BASE_IMAGE> python -c "import torch; print(torch.__version__)"
#   e.g. "2.4.0+cu118" -> the cu118 defaults below are correct.
#        "2.4.1+cu121" -> you need a cu12x torch2.4 wheel.
# Heads-up: as of this writing the published torch2.4 wheels for both packages
# are cu118 (CUDA 11.x) ONLY. If the base image is CUDA 12.x there is no matching
# prebuilt torch2.4 wheel -> either build from source (commented block below) or
# use a CUDA-11.8 / torch-2.4 base image.
ARG BASE_IMAGE=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
FROM ${BASE_IMAGE}

# Override with --build-arg to match the base image's exact CUDA / cxx11abi.
ARG CAUSAL_CONV1D_WHL=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
ARG MAMBA_SSM_WHL=https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

RUN pip install --no-cache-dir "${CAUSAL_CONV1D_WHL}" "${MAMBA_SSM_WHL}"

# Fail the build early on ABI mismatch: this loads the compiled CUDA extensions
# (no GPU needed to import them), so a wrong cuXXX/cxx11abi wheel errors here
# rather than silently at training time.
RUN python -c "import causal_conv1d, mamba_ssm; from mamba_ssm import Mamba; print('mamba_ssm', mamba_ssm.__version__, 'import OK')"

# --- Fallback: build from source (only if no matching prebuilt wheel exists) ---
# Requires the CUDA toolkit (nvcc) to be present in the base image.
# RUN MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE \
#     pip install --no-cache-dir --no-build-isolation \
#     "causal-conv1d>=1.4" "mamba-ssm>=2.2"
