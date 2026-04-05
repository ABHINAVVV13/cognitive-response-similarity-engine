# ============================================================================
# CRSE RunPod Serverless Worker — Dockerfile
# ============================================================================
# Builds a GPU-ready container with TRIBE v2 + CRSE pre-installed.
#
# Build:
#   docker build -t crse-worker:latest -f runpod/Dockerfile .
#
# Test locally:
#   docker run --gpus all -p 8000:8000 crse-worker:latest \
#       python -u runpod/handler.py --rp_serve_api
#
# Push to registry:
#   docker tag crse-worker:latest <your-registry>/crse-worker:latest
#   docker push <your-registry>/crse-worker:latest
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

# Tell uv to install into the system Python (no venv in Docker)
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# ── Install TRIBE v2 ──────────────────────────────────────────────────
RUN git clone https://github.com/facebookresearch/tribev2.git /app/tribev2 \
    && uv pip install -e /app/tribev2

# ── Install CRSE dependencies ─────────────────────────────────────────
COPY pyproject.toml .
COPY crse/ crse/
COPY runpod/handler.py runpod/handler.py

RUN uv pip install -e . \
    && uv pip install runpod requests

# ── Install spaCy language model ───────────────────────────────────────
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
RUN mkdir -p /cache /tmp/crse
ENV CRSE_CACHE_DIR="/cache"
ENV CRSE_DEVICE="auto"
ENV CRSE_MODEL_ID="facebook/tribev2"

# ── Healthcheck ────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD python -c "import crse; print('ok')" || exit 1

# ── Run the handler ────────────────────────────────────────────────────
CMD ["python", "-u", "runpod/handler.py"]
