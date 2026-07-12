"""
An FastAPI application that serves as an API for inference on an Image Classification model. The application should be able to handle
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image
from contextlib import asynccontextmanager
import logging
import asyncio
import time
import io
import base64
from config import Config
from model_manager import ModelManager
from collections.abc import AsyncIterator
from typing import List, Tuple, Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_model() -> None:
    """Initialize and load the image classification model into application state."""
    # Read the current configuration and create a model manager for the selected backend.
    config = app.state.config
    try:
        app.state.model_manager = ModelManager(
            config.MODEL_NAME, config.INFERENCE_DEVICE
        )
    except Exception as e:
        logger.error(
            f"Failed to Initialize the model {config.MODEL_NAME} on {config.INFERENCE_DEVICE} Exception {e}"
        )
        raise
    try:
        app.state.model_manager.load_model()
    except Exception as e:
        logger.error(f"Model loading failed. Exception {e}.")
        raise

    logger.info(
        f"Model {config.MODEL_NAME} loaded successfully on {config.INFERENCE_DEVICE}"
    )


async def perform_inference(
    batch_inputs: List[Dict[str, Any]],
) -> List[List[Tuple[str, float]]]:
    """Decode base64 image payloads, run inference, and return top-k predictions."""
    config = app.state.config
    model_manager = app.state.model_manager
    try:
        # Convert each incoming payload into a PIL image before preprocessing.
        image_batch: List[Image.Image] = []
        for payload in batch_inputs:
            if not isinstance(payload, dict):
                raise ValueError("Each inference payload must be a dictionary")
            b64_data = payload.get("image")
            if not isinstance(b64_data, str) or not b64_data.strip():
                raise ValueError(
                    "Each inference payload must include a non-empty image string"
                )
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
            img_bytes = base64.b64decode(b64_data)
            image = Image.open(io.BytesIO(img_bytes))
            image_batch.append(image)

        # Preprocess the images and produce predictions from the loaded model.
        if model_manager is not None:
            processed_image_batch = model_manager.preprocess_inputs(image_batch)
            logits = model_manager.predict(processed_image_batch)
            output = model_manager.top_k_from_logits(logits, config.TOP_K_PREDICTIONS)
        else:
            raise ValueError("Model Manager is not initialized during ")

    except Exception as e:
        logger.error(f"Exception {e} while performing inference.")
        raise
    return output


async def batch_processing_loop() -> None:
    """Continuously collect queued requests into batches and dispatch inference work."""
    config = app.state.config
    inference_queue = app.state.inference_queue
    shutdown_event = app.state.shutdown_event

    # Keep draining the queue until shutdown is requested and the queue is empty.
    while not shutdown_event.is_set() or not inference_queue.empty():
        try:
            batch: List[Tuple[Dict[str, Any], asyncio.Future[Any]]] = []
            if shutdown_event.is_set():
                try:
                    first_item = inference_queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
            else:
                try:
                    first_item = await asyncio.wait_for(
                        inference_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
            batch.append(first_item)
            window = 0 if shutdown_event.is_set() else config.BATCHING_TIMEOUT_MS
            start_time = time.perf_counter()
            while len(batch) < config.MAX_BATCH_SIZE:
                time_elapsed = time.perf_counter() - start_time
                time_left = window - time_elapsed
                if time_left <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        inference_queue.get(), timeout=max(0, time_left)
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            batch_inputs = [payload for (payload, _) in batch]
            topkoutputs = await perform_inference(batch_inputs)
            for i, (_, response_ticket) in enumerate(batch):
                if not response_ticket.done():
                    response_ticket.set_result(topkoutputs[i])
                inference_queue.task_done()
        except Exception as e:
            logger.warning(f"Error in batch processing loop: {e}")
            await asyncio.sleep(0.1)
    logger.info("Dynamic Batching Loop terminating cleanly")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the model and background batching loop for the application lifetime."""
    # Load the model once when the FastAPI app starts.
    init_model()
    model_manager = app.state.model_manager
    shutdown_event = app.state.shutdown_event
    batch_task = asyncio.create_task(batch_processing_loop())
    yield
    # Stop the batching loop and clean up the model on shutdown.
    shutdown_event.set()
    await batch_task
    if model_manager is not None:
        model_manager.cleanup_model()


app = FastAPI(
    name="Image Classification API",
    description="A FastAPI application that serves predictions from an image classification model.",
    lifespan=lifespan,
)
app.state.config = Config.from_env()
app.version = app.state.config.API_VERSION
app.state.model_manager = None
app.state.inference_queue = asyncio.Queue(
    maxsize=app.state.config.MAX_CONCURRENT_REQUESTS
)
app.state.shutdown_event = asyncio.Event()


@app.get("/health/live", status_code=status.HTTP_200_OK)
def liveness_check() -> JSONResponse:
    """Return a simple success response to confirm the service is alive."""
    # The service is considered healthy when this endpoint responds successfully.
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ALIVE"})


@app.get("/health/ready")
async def readyness_check() -> JSONResponse:
    """Report whether the application is ready to accept inference requests."""
    model_manager = app.state.model_manager
    inference_queue = app.state.inference_queue
    shutdown_event = app.state.shutdown_event

    # Reject readiness while the service is shutting down.
    if shutdown_event.is_set():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "SHUTDOWN"},
        )

    # The model must be loaded before the API can serve real requests.
    if not (model_manager is not None and model_manager.model_loaded):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "MODEL_LOADING"},
        )

    # The queue should be able to accept more work before requests are routed.
    if inference_queue.full():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "QUEUE SATURATED"},
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "READY"})


@app.get("/health/startup")
def startup_check() -> JSONResponse:
    """Report whether the application startup process has completed."""
    model_manager = app.state.model_manager
    is_started = model_manager is not None and model_manager.model_loaded

    # The service is still starting until the model has been loaded.
    if is_started:
        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"status": "STARTED"}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "STARTING"},
        )


@app.post("/predict")
async def handle_predict_request(
    payload: Dict[str, Any],
) -> Dict[str, List[Tuple[str, float]]]:
    """Enqueue an inference request and await the resulting predictions."""
    inference_queue = app.state.inference_queue
    shutdown_event = app.state.shutdown_event

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must be a JSON object.",
        )

    image_payload = payload.get("image")
    if not isinstance(image_payload, str) or not image_payload.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must include a non-empty 'image' field.",
        )

    # Reject requests once the application is shutting down.
    if shutdown_event.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )

    # Create a ticket that will receive the prediction result from the batch loop.
    ticket = asyncio.get_running_loop().create_future()
    try:
        inference_queue.put_nowait((payload, ticket))
    except asyncio.QueueFull:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Queue full."
        )
    result: List[Tuple[str, float]] = await ticket
    return {"prediction": result}


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
