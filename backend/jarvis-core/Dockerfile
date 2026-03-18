# JARVIS 2.0 — Multi-stage Dockerfile
FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── App ──────────────────────────────────────────────────────────────
FROM base AS app

RUN useradd -m -u 1001 jarvis
WORKDIR /app
COPY --chown=jarvis:jarvis . .
RUN mkdir -p /app/data && chown jarvis:jarvis /app/data

USER jarvis

EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:10000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
