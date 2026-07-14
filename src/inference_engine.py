import queue
import threading
import asyncio
import logging
import time
from PIL import Image
from typing import Tuple, List
from config import Config
from model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, config: Config):
        self.config = config
        self.ready = False
        self.inference_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.init_model_manager()
        self.inference_thread = threading.Thread(
            target=self.batch_processing_loop, daemon=True
        )
        self.inference_thread.start()

    def init_model_manager(self) -> None:
        """Initialize and load the image classification model into application state."""
        # Read the current configuration and create a model manager for the selected backend.
        try:
            self.manager = ModelManager(
                self.config.MODEL_NAME, self.config.INFERENCE_DEVICE
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
        self.inference_queue.put((image, loop, future))
        return future

    def perform_inference(
        self,
        image_batch: List[Image.Image],
        loops: List[asyncio.AbstractEventLoop],
        futures: List[asyncio.Future[Tuple[List[Tuple[str, float]], float]]],
    ) -> Tuple[List[List[Tuple[str, float]]], float] | None:
        """Read image bytes, run inference, and return top-k predictions."""
        try:
            processed_image_batch = self.manager.preprocess_inputs(image_batch)
            logits, inference_time_ms = self.manager.predict(processed_image_batch)
            output = self.manager.top_k_from_logits(
                logits, self.config.TOP_K_PREDICTIONS
            )
        except Exception as e:
            for loop, future in zip(loops, futures):
                try:

                    def safely_fail_future(
                        f: asyncio.Future[
                            Tuple[List[Tuple[str, float]], float]
                        ] = future,
                        err: Exception = e,
                    ) -> None:
                        try:
                            f.set_exception(
                                RuntimeError(
                                    f"Inference Proceedure Failed Exception {err}"
                                )
                            )
                        except asyncio.InvalidStateError:
                            pass

                    loop.call_soon_threadsafe(safely_fail_future)
                except Exception:
                    continue
            return None
        return output, inference_time_ms

    def batch_processing_loop(self) -> None:
        """Continuously collect queued requests into batches and dispatch inference work."""

        self.ready = True
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
                if (
                    first_item[0] is None
                    or first_item[0].size[0] == 0
                    or first_item[0].size[1] == 0
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
                    if item[0] is None or item[0].size[0] == 0 or item[0].size[1] == 0:
                        item[1].call_soon_threadsafe(
                            item[2].set_exception,
                            ValueError("Invalid Image Dimensions."),
                        )
                        continue
                    batch.append(item)
                except queue.Empty:
                    break

            batch_inputs, loops, futures = map(list, zip(*batch))
            infout = self.perform_inference(batch_inputs, loops, futures)
            if infout is not None:
                topkoutputs, inference_time_ms = infout
                for i, (_, loop, future) in enumerate(batch):
                    try:
                        result = (topkoutputs[i], inference_time_ms)

                        def safely_set_result(
                            f: asyncio.Future[
                                Tuple[List[Tuple[str, float]], float]
                            ] = future,
                            res: Tuple[List[Tuple[str, float]], float] = result,
                        ) -> None:
                            try:
                                f.set_result(res)
                            except asyncio.InvalidStateError:
                                pass

                        loop.call_soon_threadsafe(safely_set_result)
                    except Exception:
                        continue
        logger.info("Dynamic Batching Loop terminating cleanly")

    def shutdown(self):
        logger.info("Inference Engine: Initiating graceful shutdown")
        self.shutdown_event.set()
        self.inference_queue.put(None)
        self.inference_thread.join(timeout=5.0)
        self.manager.cleanup_model()
        logger.info("Inference Engine: Shutdown complete")
