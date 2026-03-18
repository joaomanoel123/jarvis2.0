# ════════════════════════════════════════════════════════
#  JARVIS v2 — Multi-stage production Dockerfile
#
#  Stage 1 (builder): compile wheels in a build environment
#  Stage 2 (runtime): lean production image, non-root user
#
#  Build:
#    docker build -t jarvis-v2 .
#
#  Run (local):
#    docker run -p 8000:8000 \
#               -e LLM_MODEL_ID=tinyllama \
#               -e HF_TOKEN=your_token \
#               -v /tmp/jarvis_files:/tmp/jarvis_files \
#               jarvis-v2
# ════════════════════════════════════════════════════════

# ── Stage 1: Build ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Pre-copy only requirements to leverage Docker layer caching
COPY requirements.txt .

# Build wheels — no-cache keeps the image lean
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="JARVIS Team"
LABEL description="JARVIS v2 Multi-Agent AI Backend"

# Security: non-root user
RUN useradd --no-create-home --shell /bin/false --uid 10001 jarvis

WORKDIR /app

# Install runtime system libs (curl for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install pre-built wheels from builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels

# Copy application source
COPY --chown=jarvis:jarvis . .

# HuggingFace model cache — mapped to persistent disk in production
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chown -R jarvis:jarvis /app/.cache

# Sandboxed file directory for file_reader tool
RUN mkdir -p /tmp/jarvis_files && chown jarvis:jarvis /tmp/jarvis_files

# Drop privileges
USER jarvis

# Port
EXPOSE 8000

# Model can take 30-120 s to load on first start
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=120s \
    --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker — do NOT raise; multiple workers = multiple model copies = OOM
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
