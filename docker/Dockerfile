# Multi-stage Dockerfile for Simple Model Inference API
# Stage 1: Builder - Install dependencies and prepare Python environment
FROM python:3.12-slim AS builder

# Install uv from the distroless image
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

# Set working directory
WORKDIR /app

#Optimize uv performance for Docker
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency definition files first for optimal layer caching
COPY pyproject.toml uv.lock ./

ARG EXTRA=cpu

# Install dependencies into a self-contained virtual environment (.venv)
# Using cache mounting prevents downloading packages on every rebuild
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra ${EXTRA}

# Copy the rest of the application source code
COPY . .

# Complete the sync to install the project itself into the .venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra ${EXTRA}

# Stage 2: Runtime - Minimal image with only necessary dependencies
FROM python:3.12-slim

WORKDIR /app

# Create a secure, non-privileged system user
RUN useradd -u 8888 -m modelapi && \
    chown -R 8888:8888 /home/modelapi && \
    chmod -R 777 /home/modelapi

# Place the virtual environment's binaries directly onto the PATH
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    MODEL_NAME="microsoft/resnet-50" \
    INFERENCE_DEVICE="cpu" \
    API_HOST="0.0.0.0" \
    API_PORT="8000" \
    DEBUG="false"

# Copy only the compiled virtual environment from the builder stage
# Both stages must use identical Python base images so the binary paths match
COPY --from=builder --chown=modelapi:modelapi /app /app

# Switch to the non-root user for execution
USER modelapi

# Expose API port
EXPOSE ${API_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${API_PORT}/health/live', timeout=5)" || exit 1

# Run the application
CMD ["python", "-m", "src.app"]