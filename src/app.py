"""
An FastAPI application that serves as an API for inference on an Image Classification model. The application should be able to handle
"""

from fastapi import FastAPI, HTTPException, status, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from contextlib import asynccontextmanager
import logging
import asyncio
import time
import io
from config import Config
from model_manager import ModelManager
from collections.abc import AsyncGenerator
from typing import List, Tuple, Dict, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_model() -> None:
    """Initialize and load the image classification model into application state."""
    # Read the current configuration and create a model manager for the selected backend.
    config: Config = app.state.config
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
    image_batch: List[Image.Image],
) -> List[List[Tuple[str, float]]]:
    """Read image bytes, run inference, and return top-k predictions."""
    config: Config = app.state.config
    model_manager: ModelManager = app.state.model_manager
    try:
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


async def get_image_from_upload_file(
    upload: Tuple[UploadFile, asyncio.Future[List[Tuple[str, float]]]],
) -> Tuple[Image.Image, asyncio.Future[List[Tuple[str, float]]]]:
    img_bytes = await upload[0].read()
    stream = io.BytesIO(img_bytes)
    try:
        image: Image.Image = Image.open(stream)
        image.load()
    except Exception as e:
        logger.error(f"Error loading image from upload file. Exception {e}")
        upload[1].set_exception(
            HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded File is not an Image.",
            )
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded File is not an Image.",
        )
    return (image, upload[1])


async def batch_processing_loop() -> None:
    """Continuously collect queued requests into batches and dispatch inference work."""
    config: Config = app.state.config
    inference_queue: asyncio.Queue = app.state.inference_queue
    shutdown_event: asyncio.Event = app.state.shutdown_event

    # Keep draining the queue until shutdown is requested and the queue is empty.
    while not shutdown_event.is_set() or not inference_queue.empty():
        try:
            batch: List[Tuple[Image.Image, asyncio.Future[List[Tuple[str, float]]]]] = (
                []
            )
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

            try:
                batch.append(await get_image_from_upload_file(first_item))
            except Exception as e:
                logger.error(f"Exception {e} Getting first item from inference queue.")
                inference_queue.task_done()
                continue

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
                    try:
                        batch.append(await get_image_from_upload_file(item))
                    except Exception as e:
                        logger.error(
                            f"Exception {e} Getting item from inference queue."
                        )
                        inference_queue.task_done()
                        continue
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
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize the model and background batching loop for the application lifetime."""
    # Load the model once when the FastAPI app starts.
    init_model()
    model_manager: ModelManager = app.state.model_manager
    shutdown_event: asyncio.Event = app.state.shutdown_event
    batch_task: asyncio.Future = asyncio.create_task(batch_processing_loop())
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
    model_manager: ModelManager = app.state.model_manager
    inference_queue: asyncio.Queue = app.state.inference_queue
    shutdown_event: asyncio.Event = app.state.shutdown_event

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
    model_manager: ModelManager = app.state.model_manager
    is_started: bool = model_manager is not None and model_manager.model_loaded

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


@app.get("/info")
async def info() -> JSONResponse:
    model_manager: ModelManager = app.state.model_manager
    return JSONResponse(
        status_code=status.HTTP_200_OK, content=model_manager.get_model_info()
    )


@app.post("/predict")
async def handle_predict_request(
    file: UploadFile, metadata: Optional[str] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """Enqueue an inference request and await the resulting predictions."""
    inference_queue: asyncio.Queue = app.state.inference_queue
    shutdown_event: asyncio.Event = app.state.shutdown_event

    # Reject requests once the application is shutting down.
    if shutdown_event.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )

    # Create a ticket that will receive the prediction result from the batch loop.
    ticket = asyncio.get_running_loop().create_future()
    try:
        inference_queue.put_nowait((file, ticket))
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
