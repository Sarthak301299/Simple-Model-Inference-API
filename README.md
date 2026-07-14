# Simple Model Inference API

A lightweight, scalable REST API for image classification that uses pre-trained models from the Hugging Face transformers library. This application provides efficient batch processing with support for both GPU and CPU inference, making it ideal for production deployments.

## Features

- **REST API**: FastAPI-based HTTP interface for image classification
- **Batch Processing**: Efficient batch inference with dynamic batching to handle surge of requests
- **Decoupled Async-Compute Architecture**: Offloads compute-bound operations to a dedicated OS-level worker thread, isolating network I/O from hardware execution.
- **Supported Models**: Tested with ResNet-50
- **GPU Support**: Automatic GPU detection with CUDA support or CPU fallback
- **Configuration Management**: Environment-based configuration for flexible deployments
- **Production Ready**: Includes Docker containerization, health checks, and comprehensive logging
- **Comprehensive Testing**: Full test coverage with pytest
- **Non-root Container**: Runs as non-privileged user for enhanced security

## Architecture

The application consists of three main components:

- **app.py**: FastAPI application with request handling and batch processing
- **model_manager.py**: Model loading, preprocessing, and inference logic
- **config.py**: Environment-based configuration management

## Prerequisites

- Python 3.12 or higher
- pip for dependency management
- Optional: NVIDIA GPU with CUDA support for accelerated inference

## Installation

### Local Development

Clone the repository:
```bash
git clone <repository-url>
cd Simple\ Model\ Inference\ API
```

Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements/cpu.txt
```

### Dependency sets

- `requirements/prod.txt`: API runtime dependencies excluding Torch backend selection.
- `requirements/cpu.txt`: CPU inference runtime.
- `requirements/cuda-cu126.txt`: CUDA 12.6 inference runtime.
- `requirements/dev.txt`: test/lint/type-check dependencies.
- `requirements/load.txt`: load-testing dependencies.
4. Create a `.env` file for configuration (optional):
```bash
# .env file example
MODEL_NAME=microsoft/resnet-50
# Change to cuda for GPU
INFERENCE_DEVICE=cpu
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO
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
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Docker Container

```bash
docker run -p 8000:8000 \
  -e MODEL_NAME=microsoft/resnet-50 \
  -e INFERENCE_DEVICE=cpu \
  simple-model-inference-api:latest
```

### GPU Support (Docker)

For GPU support, install NVIDIA Docker runtime and use:
```bash
docker run --gpus all -p 8000:8000 \
  -e INFERENCE_DEVICE=cuda \
  simple-model-inference-api:latest
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
"status": "ALIVE"
}
```

### Prediction
```http
POST /predict
Content-Type: multipart/form-data

file: <image-file>
```

Upload an image for classification.

**Request:**
```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict
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

### Health Checks

The container includes a health check that:
- Runs every 30 seconds
- Times out after 10 seconds
- Waits 40 seconds before first check
- Retries up to 3 times

### Volume Management (Docker Compose)

- `model-cache`: Persistent Hugging Face model cache across restarts
- `./models`: Local model weights volume (optional)



### Project Structure
```
.
├── src/
│   ├── app.py              # FastAPI application
│   ├── config.py           # Configuration management
│   └── model_manager.py    # Model loading and inference
├── tests/
│   ├── test_app.py
│   ├── test_config.py
│   └── test_model_manager.py
├── docker/
│   ├── Dockerfile          # Production Dockerfile
│   ├── docker-compose.yml  # Docker Compose configuration
│   └── .dockerignore       # Docker build exclusions
├── pyproject.toml          # Project metadata and configuration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

