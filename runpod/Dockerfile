# ============================================================================
# CRSE RunPod Serverless Worker — Dockerfile
# ============================================================================
# Builds a GPU-ready container with TRIBE v2 + CRSE pre-installed.
#
# Build (repo root):
#   docker build -t crse-worker:latest -f runpod/Dockerfile .
#
# Test locally:
#   docker run --gpus all -p 8000:8000 crse-worker:latest \
#       python -u runpod/handler.py --rp_serve_api
#
# Push to registry:
#   docker tag crse-worker:latest <your-registry>/crse-worker:latest
#   docker push <your-registry>/crse-worker:latest
#
# Notes:
# - README.md must be copied: pyproject.toml has readme = "README.md" (hatchling).
# - Handler is copied *after* pip install so editing handler.py alone reuses the
#   heavy install layer (BuildKit layer cache).
# - Set HF_TOKEN on the RunPod template so gated Hugging Face models can download.
# ============================================================================

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# ── System dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv ────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# ── TRIBE v2 (heavy); separate layer caches when only CRSE code changes ──
# TRIBE's default path runs `uvx whisperx`, which re-downloads PyTorch/CUDA into
# ~/.cache/uv on every cold worker and often exhausts RunPod's small root disk.
# We patch TRIBE to call the `whisperx` CLI and install whisperx into this image.
RUN git clone --depth 1 https://github.com/facebookresearch/tribev2.git /app/tribev2

COPY runpod/patch_tribev2_whisperx.py /tmp/patch_tribev2_whisperx.py
RUN python /tmp/patch_tribev2_whisperx.py

RUN uv pip install -e /app/tribev2 \
    && uv pip install whisperx

# ── CRSE package metadata + source (handler copied later on purpose) ───
COPY pyproject.toml README.md .
COPY crse/ crse/

# runpod + tribe-plot: cortical PNGs (surface_pngs_base64) on every job
RUN uv pip install -e ".[runpod,tribe-plot]"

COPY runpod/handler.py runpod/handler.py

# ── spaCy model (TRIBE / pipeline) ──────────────────────────────────────
RUN python -m spacy download en_core_web_sm

# ── Pre-download HuggingFace model weights (optional) ──────────────────
# Uncomment the lines below to bake model weights into the image.
# This makes cold starts faster but increases image size by ~2GB.
# You must set HF_TOKEN as a build arg for gated models.
#
# ARG HF_TOKEN
# RUN python -c "
# from huggingface_hub import hf_hub_download, login
# login(token='${HF_TOKEN}')
# hf_hub_download('facebook/tribev2', 'config.yaml')
# hf_hub_download('facebook/tribev2', 'best.ckpt')
# "

# ── Cache and tmp directories ──────────────────────────────────────────
# Keep Hugging Face + uv caches on /cache so they share the same volume as CRSE
# and do not fill /root (Whisper large-v3 alone is ~3GB; TRIBE adds more).
RUN mkdir -p /cache/huggingface /cache/uv /tmp/crse
ENV CRSE_CACHE_DIR="/cache"
ENV HF_HOME="/cache/huggingface"
ENV HUGGINGFACE_HUB_CACHE="/cache/huggingface"
ENV TRANSFORMERS_CACHE="/cache/huggingface"
ENV UV_CACHE_DIR="/cache/uv"
ENV CRSE_DEVICE="auto"
ENV CRSE_MODEL_ID="facebook/tribev2"

# Lightweight check (does not load TRIBE weights; worker loads them at startup)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD python -c "import crse; print('ok')" || exit 1

CMD ["python", "-u", "runpod/handler.py"]
