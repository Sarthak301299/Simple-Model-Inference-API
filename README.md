# Simple Model Inference API

A lightweight, scalable REST API for image classification that uses pre-trained models from the Hugging Face transformers library. This application provides efficient batch processing with support for both GPU and CPU inference, making it ideal for production deployments.

## Features

- **REST API**: FastAPI-based HTTP interface for image classification
- **Batch Processing**: Efficient batch inference with dynamic batching to handle surge of requests
- **Decoupled Async-Compute Architecture**: Offloads compute-bound operations to a dedicated OS-level worker thread, isolating network I/O from hardware execution.
- **Isolated Batch Failures**: Failures during image processing ensures that the entire batch is not affected
- **Prometheus Metrics**: Real-time health and performance updates
- **Load Tested**: Simulated loads with locust with valid and invalid inputs to demonstrate clean execution
- **Test Automation**: CI workflow using GitHub Actions to automate testing
- **Supported Models**: Tested with ResNet-50
- **GPU Support**: Automatic GPU detection with CUDA support or CPU fallback
- **Configuration Management**: Environment-based configuration for flexible deployments
- **Production Ready**: Includes Docker containerization, health checks, and comprehensive logging
- **Comprehensive Testing**: High test coverage with pytest
- **Non-root Container**: Runs as non-privileged user for enhanced security

## Architecture

The application consists of three main components:

- **app.py**: FastAPI application with request handling and batch processing
- **model_manager.py**: Model loading, preprocessing, and inference logic
- **config.py**: Environment-based configuration management

## Prerequisites

- Python 3.12 or higher
- uv for dependency management
- Optional: NVIDIA GPU with CUDA support for accelerated inference

## Installation

### Local Development

Clone the repository:
```bash
git clone <repository-url>
cd Simple\ Model\ Inference\ API
```

Create a virtual environment and install dependencies (uv automatically creates venv):
```bash
uv sync --extra cpu   # For CPU (For deployment)
uv sync --extra gpu   # For GPU (For deployment)
uv sync --extra cpu --dev  # For CPU (For development)
uv sync --extra gpu --dev  # For GPU (For development)
```

Create a `.env` file for configuration (optional):
```bash
# .env file example
MODEL_NAME=microsoft/resnet-50
# Change to cuda for GPU
INFERENCE_DEVICE=cpu
LOG_LEVEL=INFO
...
```

### Docker

Build the Docker image:
```bash
docker build -f docker/Dockerfile -t simple-model-inference-api:cpu .
```
Build the Docker image for CUDA 12.6:

```bash
docker build -f docker/Dockerfile \
--build-arg REQUIREMENTS_FILE=requirements/cuda-cu126.txt \
-t simple-model-inference-api:cu126 .
```

Run the container:
```bash
# Can override values in .env like API_HOST and API_PORT if needed.
docker run --env-file .env -e API_HOST=0.0.0.0 -e API_PORT=8000 -p 8000:8000 simple-model-inference-api:cpu #Or :cu126
```

Run with Docker Compose (CPU):
```bash
docker compose -f docker/docker-compose.yml up --build
```

Run with Docker Compose (GPU):
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up --build
```

## Running the Application

### Local Development

```bash
python -m uvicorn src.app:app
```

The API will be available at `http://localhost:8000`

### Docker Container

```bash
docker run -p 8000:8000 \
  -e MODEL_NAME=microsoft/resnet-50 \
  -e INFERENCE_DEVICE=cpu \
  simple-model-inference-api:cpu
```

### GPU Support (Docker)

For GPU support, install NVIDIA Docker runtime and use:
```bash
docker run --gpus all -p 8000:8000 \
  -e INFERENCE_DEVICE=cuda \
  simple-model-inference-api:cu126
```

## API Endpoints

### Liveness Check
```http
GET /health/live
```
Returns application health status.

**Response:**
```json
{
"status": "alive"
}
```

### Readiness Check
```http
GET /health/live
```
Returns application service availability.

**Response:**
```json
{
"status": "ready"
}
```

### Startup Check
```http
GET /health/startup
```
Returns application startup status.

**Response:**
```json
{
"status": "ready"
}
```

### Information
```http
GET /info
Requires API_KEY if set in .env (or via environment variables)
```
Returns application health status.

**Response:**
```json
{
"model_name": "<model name>",
"inference_device": "<inference device>",
"model_path": "<model path>"
}
```

### Metrics
```http
GET /metrics
```
Returns prometheus metrics.

**Response:**
```json
{
"INFERENCE_LATENCY": "<Inference Latency>",
"QUEUE_DEPTH": "<Queue Depth>",
"REQUEST_COUNT": "<Number of HTTP Requests>"
,
...
}
```

### Prediction
```http
POST /predict
Requires API_KEY if set in .env (or via environment variables)
Content-Type: multipart/form-data

file: <image-file>
```

Upload an image for classification.

**Request:**
```bash
curl -H "X-API-Key: YOUR_API_KEY_HERE" -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict
```

**Response:**
```json

{"prediction":[["tench, Tinca tinca",10.069835662841797],["goldfish, Carassius auratus",-6.5064167976379395],["reel",-6.832561492919922],["yurt",-6.85081148147583],["sturgeon",-6.982334136962891]],"inference_time_ms":38.789903999713715}

```

###
## Configuration

Configuration is managed through environment variables. See `src/config.py` for all available options:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | microsoft/resnet-50 | Hugging Face model identifier |
| `INFERENCE_DEVICE` | cuda | Device for inference: `cuda` or `cpu` |
| `API_HOST` | 0.0.0.0 | API host address |
| `API_PORT` | 8000 | API port number |
| `DEBUG` | false | Enable debug logging |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `MAX_BATCH_SIZE` | 64 | Maximum batch size for inference |
| `BATCHING_TIMEOUT_MS` | 3 | Maximum wait time to fill a batch |
| `MAX_FILE_SIZE_MB` | 16 | Maximum upload file size |
| `TOP_K_PREDICTIONS` | 5 | Number of top predictions to return |

## Testing

Run the test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=. --cov-report=term-missing
```

## Docker Deployment Details

### Image Optimization

The Dockerfile uses a multi-stage build to minimize image size:
1. **Builder stage**: Installs dependencies in a full Python environment
2. **Runtime stage**: Uses only necessary runtime dependencies

### Security Features

- Non-root user (`modelapi`) for container execution
- Minimal runtime dependencies
- `.dockerignore` excludes unnecessary files
- No privileged operations
- Constant time API key comparison to prevent attackers from exploiting timing attacks

### Health Checks

The container includes a health check that:
- Runs every 30 seconds
- Times out after 10 seconds
- Waits 40 seconds before first check
- Retries up to 3 times

### Volume Management (Docker Compose)

- `model-cache`: Persistent Hugging Face model cache across restarts.

### Project Structure
```
.
├── src/
|   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   ├── config.py               # Configuration management
│   ├── inference_engine.py     # Inference engine decoupling compute from network
|   ├── metrics.py              # Prometheus metrics
|   └── model_manager.py        # Model loading and inference
├── tests/
|   ├── load/
|   |   └── locustfile.py       # Load testing script
│   ├── test_app.py
│   ├── test_config.py
│   ├── test_inference_engine.py
|   ├── test_metrics.py
|   └── test_model_manager.py
├── docker/
│   ├── Dockerfile              # Production Dockerfile
│   ├── docker-compose.yml      # Docker Compose configuration
│   └── docker-compose.gpu.yml  # Docker Compose for GPU override
├── github/
│   └── workflows/
|       └── ci.yml              # GitHub Action CI workflow.
├── pyproject.toml              # Project metadata, configuration, and dependencies
├── uv.lock                     # Cross-platform lockfile for uv
├── .gitignore                  # Don't commit to Git
├── .dockerignore               # Don't move to Docker container
├── .env_shared                 # Shared environment variables between development and production environments
├── .env.template               # Template for .env
└── README.md                   # This file
```

