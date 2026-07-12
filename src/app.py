"""
An FastAPI application that serves as an API for inference on an Image Classification model. The application should be able to handle
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image
from contextlib import asynccontextmanager
import logging
import asyncio
import queue
import time
import io
import base64
from config import Config
from model_manager import ModelManager
from typing import List, Tuple, Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def init_model() -> None:
    """
    Initialize and Load the Image Classification model.
    """
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
    config = app.state.config
    model_manager = app.state.model_manager
    try:
        ImageBatch = []
        for payload in batch_inputs:
            b64Data = payload["image"]
            if "," in b64Data:
                b64Data = b64Data.split(",")[1]
            img_bytes = base64.b64decode(b64Data)
            image = Image.open(io.BytesIO(img_bytes))
            ImageBatch.append(image)

        # Preprocess the images
        if model_manager is not None:
            processed_Image_Batch = model_manager.preprocess_inputs(ImageBatch)
            logits = model_manager.predict(processed_Image_Batch)
            output = model_manager.top_k_from_logits(logits, config.TOP_K_PREDICTIONS)
        else:
            raise ValueError("Model Manager is not initialized during ")

    except Exception as e:
        logger.error(f"Exception {e} while performing inference.")
        raise
    return output


async def batch_processing_loop() -> None:
    config = app.state.config
    inference_queue = app.state.inference_queue
    shutdown_event = app.state.shutdown_event
    while not shutdown_event.is_set() or not inference_queue.empty():
        try:
            batch = []
            if shutdown_event.is_set():
                first_item = inference_queue.get_nowait()
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
async def lifespan(app: FastAPI):
    init_model()
    model_manager = app.state.model_manager
    shutdown_event = app.state.shutdown_event
    batch_task = asyncio.create_task(batch_processing_loop())
    yield
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
def liveness_check():
    """
    Liveness check endpoint to verify the application is running.
    """
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ALIVE"})


@app.get("/health/ready")
async def readyness_check():
    model_manager = app.state.model_manager
    inference_queue = app.state.inference_queue
    shutdown_event = app.state.shutdown_event
    if shutdown_event.is_set():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "SHUTDOWN"},
        )

    if not (model_manager is not None and model_manager.model_loaded):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "MODEL_LOADING"},
        )

    if inference_queue.full():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "QUEUE SATURATED"},
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "READY"})


@app.get("/health/startup")
def startup_check():
    model_manager = app.state.model_manager
    IS_STARTED = model_manager is not None and model_manager.model_loaded
    if IS_STARTED:
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
    inference_queue = app.state.inference_queue
    shutdown_event = app.state.shutdown_event
    if shutdown_event.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )
    ticket = asyncio.get_running_loop().create_future()
    try:
        inference_queue.put_nowait((payload, ticket))
    except queue.Full:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Queue full."
        )
    result: List[Tuple[str, float]] = await ticket
    return {"prediction": result}


@app.get("/")
def read_root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=app.state.config.API_HOST,
        port=app.state.config.API_PORT,
        log_level=app.state.config.LOG_LEVEL,
    )
