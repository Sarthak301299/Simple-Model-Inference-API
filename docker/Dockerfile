# Multi-stage Dockerfile for Simple Model Inference API
# Stage 1: Builder - Install dependencies and prepare Python environment
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements/ ./requirements
ARG REQUIREMENTS_FILE=requirements/cpu.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r ${REQUIREMENTS_FILE}

# Stage 2: Runtime - Minimal image with only necessary dependencies
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src" \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_NAME="microsoft/resnet-50" \
    INFERENCE_DEVICE="cpu" \
    API_HOST="0.0.0.0" \
    API_PORT="8000" \
    DEBUG="false"

# Install runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 modelapi && \
    mkdir -p /app /app/models && \
    chown -R modelapi:modelapi /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application source code
COPY --chown=modelapi:modelapi src/ ./src/
COPY --chown=modelapi:modelapi pyproject.toml .env.shared* .env* ./

# Switch to non-root user
USER modelapi

# Expose API port
EXPOSE ${API_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${API_PORT}/health/live', timeout=5)" || exit 1

# Run the application with Uvicorn
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
