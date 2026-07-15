"""An inference engine that decouples the compute-heavy inference operations from network operations."""

import queue
import threading
import asyncio
import logging
import time
import torch
from PIL import Image
from typing import Tuple, List
from src.config import Config
from src.model_manager import ModelManager
from src.metrics import MODEL_READY, BATCH_SIZE, INFERENCE_LATENCY

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def safely_fail_future(
    f: asyncio.Future[Tuple[List[Tuple[str, float]], float]], err: Exception
) -> None:
    try:
        f.set_exception(RuntimeError(f"Inference Proceedure Failed Exception {err}"))
    except asyncio.InvalidStateError:
        pass


def safely_set_result(
    f: asyncio.Future[Tuple[List[Tuple[str, float]], float]],
    res: Tuple[List[Tuple[str, float]], float],
) -> None:
    try:
        f.set_result(res)
    except asyncio.InvalidStateError:
        pass


class InferenceEngine:
    def __init__(self, config: Config):
        self.config: Config = config
        self.ready: bool = False
        self.inference_queue: queue.Queue = queue.Queue(
            maxsize=self.config.MAX_CONCURRENT_REQUESTS
        )
        self.shutdown_event: threading.Event = threading.Event()
        self.init_model_manager()
        self.inference_thread: threading.Thread = threading.Thread(
            target=self.batch_processing_loop, daemon=True
        )
        self.inference_thread.start()

    def init_model_manager(self) -> None:
        """Initialize and load the image classification model into application state."""
        # Read the current configuration and create a model manager for the selected backend.
        try:
            self.manager = ModelManager(
                self.config.MODEL_NAME,
                self.config.INFERENCE_DEVICE,
                self.config.MODEL_PATH,
            )
        except Exception as e:
            logger.error(
                f"Failed to Initialize the model {self.config.MODEL_NAME} on {self.config.INFERENCE_DEVICE} Exception {e}"
            )
            raise
        try:
            self.manager.load_model()
        except Exception as e:
            logger.error(f"Model loading failed. Exception {e}.")
            raise

        logger.info(
            f"Model {self.config.MODEL_NAME} loaded successfully on {self.config.INFERENCE_DEVICE}"
        )

    def EnqueueRequest(
        self, image: Image.Image
    ) -> asyncio.Future[Tuple[List[Tuple[str, float]], float]]:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        future: asyncio.Future[Tuple[List[Tuple[str, float]], float]] = (
            loop.create_future()
        )
        try:
            self.inference_queue.put_nowait((image, loop, future))
        except queue.Full:
            logger.error("Failed to enqueue request.")
            del future
            raise queue.Full()
        return future

    def perform_inference(
        self,
        image_batch: List[Image.Image],
        loops: List[asyncio.AbstractEventLoop],
        futures: List[asyncio.Future[Tuple[List[Tuple[str, float]], float]]],
    ) -> None:
        """Read image bytes, run inference, and return top-k predictions, isolating failures where possible."""
        valid_tensors: List[torch.Tensor] = []  # each a (1,C,H,W) tensor
        valid_positions: List[int] = []
        for i, img in enumerate(image_batch):
            try:
                tensor = self.manager.preprocess_inputs(img)
            except Exception as e:
                loops[i].call_soon_threadsafe(safely_fail_future, futures[i], e)
                continue
            valid_tensors.append(tensor)
            valid_positions.append(i)

        if not valid_tensors:
            assert all(f.done() for f in futures)
            return None

        try:
            processed_image_batch = torch.cat(valid_tensors, dim=0)
            logits, inference_time_ms = self.manager.predict(processed_image_batch)
            output = self.manager.top_k_from_logits(
                logits, self.config.TOP_K_PREDICTIONS
            )
        except Exception as e:
            for pos in valid_positions:
                try:
                    loops[pos].call_soon_threadsafe(safely_fail_future, futures[pos], e)
                except Exception:
                    continue
            return None

        INFERENCE_LATENCY.observe(inference_time_ms / 1000)
        for local_i, pos in enumerate(valid_positions):
            result = (output[local_i], inference_time_ms)
            try:
                loops[pos].call_soon_threadsafe(safely_set_result, futures[pos], result)
            except Exception:
                continue

    def batch_processing_loop(self) -> None:
        """Continuously collect queued requests into batches and dispatch inference work."""

        self.ready = True
        MODEL_READY.set(1)
        # Keep draining the queue until shutdown is requested and the queue is empty.
        while not self.shutdown_event.is_set():
            batch: List[
                Tuple[
                    Image.Image,
                    asyncio.AbstractEventLoop,
                    asyncio.Future[Tuple[List[Tuple[str, float]], float]],
                ]
            ] = []
            start_time = time.perf_counter()
            try:
                first_item: (
                    Tuple[
                        Image.Image,
                        asyncio.AbstractEventLoop,
                        asyncio.Future[Tuple[List[Tuple[str, float]], float]],
                    ]
                    | None
                ) = self.inference_queue.get(block=True, timeout=1.0)
                if first_item is None:
                    logger.info(
                        "Inference Loop received termination signal. Exiting Loop."
                    )
                    break
                if first_item[0] is None or not self.config.validate_image(
                    first_item[0], self.config.MAX_IMAGE_DIMENSIONS
                ):
                    first_item[1].call_soon_threadsafe(
                        first_item[2].set_exception,
                        ValueError("Invalid Image Dimensions."),
                    )
                    continue
                else:
                    batch.append(first_item)

            except queue.Empty:
                continue

            while len(batch) < self.config.MAX_BATCH_SIZE:
                time_elapsed = time.perf_counter() - start_time
                time_left = (self.config.BATCHING_TIMEOUT_MS / 1000.0) - time_elapsed
                if time_left <= 0:
                    break
                try:
                    item: (
                        Tuple[
                            Image.Image,
                            asyncio.AbstractEventLoop,
                            asyncio.Future[Tuple[List[Tuple[str, float]], float]],
                        ]
                        | None
                    ) = self.inference_queue.get(block=True, timeout=time_left)
                    if item is None:
                        logger.info(
                            "Inference Loop received termination signal. Exiting Loop."
                        )
                        self.inference_queue.put(None)
                        break
                    if item[0] is None or not self.config.validate_image(
                        item[0], self.config.MAX_IMAGE_DIMENSIONS
                    ):
                        item[1].call_soon_threadsafe(
                            item[2].set_exception,
                            ValueError("Invalid Image Dimensions."),
                        )
                        continue
                    batch.append(item)
                except queue.Empty:
                    break

            if not batch:
                continue
            BATCH_SIZE.observe(len(batch))
            batch_inputs, loops, futures = map(list, zip(*batch))
            self.perform_inference(batch_inputs, loops, futures)
        logger.info("Dynamic Batching Loop terminating cleanly")

    def shutdown(self):
        logger.info("Inference Engine: Initiating graceful shutdown")
        self.shutdown_event.set()
        try:
            self.inference_queue.put(None, timeout=2.0)
        except queue.Full:
            logger.warning(
                "Inference queue is full, unable to enqueue sentinel. Worker will exit via shutdown_event poll."
            )
        self.inference_thread.join(timeout=5.0)
        if self.inference_thread.is_alive():
            logger.error(
                "Inference worker thread did not terminate within timeout. Exiting with potential leaks."
            )
        self.manager.cleanup_model()
        MODEL_READY.set(0)
        logger.info("Inference Engine: Shutdown complete")
