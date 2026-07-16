"""
An FastAPI application that serves as an API for inference on an Image Classification model. The application should be able to handle
"""

from fastapi import (
    FastAPI,
    HTTPException,
    status,
    UploadFile,
    Request,
    File,
    Header,
    Depends,
)
from fastapi.responses import JSONResponse, Response
from PIL import Image
from contextlib import asynccontextmanager
import logging
import io
import queue
import hmac
import time
from src.config import Config
from src.inference_engine import InferenceEngine
from collections.abc import AsyncGenerator
from typing import List, Tuple, Dict, Callable
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from src.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    REJECTED_COUNT,
    QueueDepthCollector,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def read_with_limits(file: UploadFile, max_bytes: int) -> bytes:
    chunk_size = app.state.config.MAX_CHUNK_SIZE_MB * 1024 * 1024
    data = bytearray()
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {max_bytes} bytes",
            )
    return bytes(data)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize the model and background batching loop for the application lifetime."""
    # Start up the inference engine
    app.state.inf_engine = InferenceEngine(app.state.config)
    yield
    # Stop the inference engine and clean up the model on shutdown.
    if hasattr(app.state, "inf_engine"):
        app.state.inf_engine.shutdown()


app = FastAPI(
    name="Image Classification API",
    description="A FastAPI application that serves predictions from an image classification model.",
    lifespan=lifespan,
)
app.state.config = Config.from_env()
app.version = app.state.config.API_VERSION

REGISTRY.register(QueueDepthCollector(lambda: getattr(app.state, "inf_engine", None)))


@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware to collect metrics."""

    # Don't measure the metrics endpoint itself.
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(
        endpoint=request.url.path, status_code=response.status_code
    ).inc()
    return response


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> JSONResponse:
    """Return a response to confirm the service is alive."""

    engine = getattr(app.state, "inf_engine", None)
    if not engine:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "dead", "detail": "Inference Engine not initialized."},
        )
    if not engine.inference_thread.is_alive():
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "dead", "detail": "Inference worker thread crashed."},
        )
    if engine.shutdown_event.is_set():
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "dead", "detail": "Inference Engine is shutting down."},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "alive"},
    )


@app.get("/health/ready")
async def readyness_check() -> JSONResponse:
    """Report whether the application is ready to accept inference requests."""

    engine = getattr(app.state, "inf_engine", None)
    if not engine or not engine.ready or not engine.manager.model_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "detail": "Initializing Inference Engine"},
        )

    if engine.shutdown_event.is_set() or not engine.inference_thread.is_alive():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not ready",
                "detail": "Inference Engine is shutting down.",
            },
        )

    # The queue should be able to accept more work before requests are routed.
    if engine.inference_queue.full():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "detail": "Inference Queue Saturated"},
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})


@app.get("/health/startup")
async def startup_check() -> JSONResponse:
    """Report whether the application startup process has completed."""

    # The service is still starting until the model has been loaded.
    engine = getattr(app.state, "inf_engine", None)
    if not engine or not engine.ready or not engine.manager.model_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "detail": "Initializing Inference Engine"},
        )

    if engine.shutdown_event.is_set() or not engine.inference_thread.is_alive():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not ready",
                "detail": "Inference Engine is shutting down.",
            },
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = app.state.config.API_KEY
    if expected is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@app.get("/info", dependencies=[Depends(verify_api_key)])
async def info() -> JSONResponse:
    engine = getattr(app.state, "inf_engine", None)
    if not engine or not engine.ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "detail": "Initializing Inference Engine"},
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK, content=engine.manager.get_model_info()
    )


@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def handle_predict_request(
    request: Request,
    file: UploadFile = File(...),
) -> Dict[str, List[Tuple[str, float]] | float]:
    """Enqueue an inference request and await the resulting predictions."""
    engine: InferenceEngine | None = getattr(app.state, "inf_engine", None)
    if not engine or not engine.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Initializing Inference Engine",
        )

    if (file.content_type is not None) and (not file.content_type.startswith("image/")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Target"
        )

    # A fast path check for the queue before any reads.
    if engine.inference_queue.full():
        REJECTED_COUNT.inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Inference Queue is full",
            headers={"Retry-After": str(app.state.config.API_RETRY)},
        )

    max_bytes = app.state.config.MAX_FILE_SIZE_MB * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {max_bytes} bytes",
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Content-Length header",
            )

    try:
        img_bytes = await read_with_limits(file, max_bytes)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format: {str(e)}",
        )

    if not app.state.config.validate_image(
        image, app.state.config.MAX_IMAGE_DIMENSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image dimensions {image.size} are non-positive or exceed the maximum allowed {(app.state.config.MAX_IMAGE_DIMENSIONS[0], app.state.config.MAX_IMAGE_DIMENSIONS[1])}",
        )

    # Another check before enqueuing to account for the queue filling up during read.
    try:
        result: Tuple[List[Tuple[str, float]], float] = await engine.EnqueueRequest(
            image
        )
    except queue.Full:
        REJECTED_COUNT.inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Inference Queue is full",
            headers={"Retry-After": str(app.state.config.API_RETRY)},
        )
    return {"prediction": result[0], "inference_time_ms": result[1]}


@app.get("/")
def read_root() -> Dict[str, str]:
    """Return a simple welcome payload for the root endpoint."""
    return {"message": "Hello World"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=app.state.config.API_HOST,
        port=app.state.config.API_PORT,
        log_level=app.state.config.LOG_LEVEL,
    )
