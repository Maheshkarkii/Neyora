# Production Multi-Stage Dockerfile for Nepali Voice Translator API
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEVICE=cpu \
    LOG_LEVEL=INFO \
    PORT=8000

WORKDIR /app

# Install system audio dependencies (libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, configs, and UI
COPY src/ ./src/
COPY configs/ ./configs/
COPY ui/ ./ui/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
