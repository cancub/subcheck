FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/config/hf-cache \
    SUBCHECK_CACHE_DIR=/config/opensubtitles \
    SUBCHECK_HISTORY_FILE=/config/history.json \
    SUBCHECK_HOST=0.0.0.0 \
    SUBCHECK_PORT=8765 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install third-party dependencies in a cached layer first
RUN pip install --no-cache-dir \
    "faster-whisper>=1.0.0" \
    "pysrt>=1.1.1" \
    "jiwer>=3.0.0" \
    "numpy>=1.24" \
    "nvidia-cublas-cu12" \
    "nvidia-cudnn-cu12==9.*" \
    "opensubtitlescom"

COPY pyproject.toml README.md ./
COPY subcheck ./subcheck

RUN pip install --no-cache-dir --no-deps .

VOLUME ["/config", "/data"]
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/healthz || exit 1

ENTRYPOINT ["subcheck"]
CMD ["serve"]
