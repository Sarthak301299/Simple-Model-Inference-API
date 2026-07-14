"""
An FastAPI application that serves as an API for inference on an Image Classification model. The application should be able to handle
"""

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from contextlib import asynccontextmanager
import logging
import io
from config import Config
from inference_engine import InferenceEngine
from collections.abc import AsyncGenerator
from typing import List, Tuple, Dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    if not engine or not engine.ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "detail": "Initializing Inference Engine"},
        )

    # The queue should be able to accept more work before requests are routed.
    if engine.inference_queue.qsize() > app.state.config.MAX_CONCURRENT_REQUESTS:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "ready", "detail": "Inference Queue Saturated"},
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})


@app.get("/health/startup")
def startup_check() -> JSONResponse:
    """Report whether the application startup process has completed."""

    # The service is still starting until the model has been loaded.
    engine = getattr(app.state, "inf_engine", None)
    if not engine or not engine.ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "detail": "Initializing Inference Engine"},
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})


@app.get("/info")
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


@app.post("/predict")
async def handle_predict_request(
    file: UploadFile = File(...),
) -> Dict[str, List[Tuple[str, float]] | float]:
    """Enqueue an inference request and await the resulting predictions."""
    engine = getattr(app.state, "inf_engine", None)
    if not engine or not engine.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Initializing Inference Engine",
        )

    if (file.content_type is not None) and (not file.content_type.startswith("image/")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Target"
        )

    # Create a ticket that will receive the prediction result from the inference engine.
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format: {str(e)}",
        )

    result: Tuple[List[Tuple[str, float]], float] = await engine.EnqueueRequest(image)
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
