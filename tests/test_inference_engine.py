import torch
import queue
import threading
import asyncio
import pytest
from types import SimpleNamespace
from PIL import Image
from src.inference_engine import InferenceEngine, safely_set_result, safely_fail_future


class ImmediateLoop:
    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


class FakeManager:
    def __init__(self):
        self.model_loaded = True
        self.cleaned = False
        self.preprocess_calls = 0

    def preprocess_inputs(self, image):
        self.preprocess_calls += 1
        if getattr(image, "should_fail_preprocess", False):
            raise ValueError("bad image")
        return torch.zeros(1, 3, 2, 2)

    def predict(self, batch):
        if getattr(self, "fail_predict", False):
            raise RuntimeError("predict failed")
        return torch.ones(batch.shape[0], 3), 5.0

    def top_k_from_logits(self, logits, k):
        if getattr(self, "fail_top_k", False):
            raise RuntimeError("top-k failed")
        return [[("class", float(i))] for i in range(logits.shape[0])]

    def cleanup_model(self):
        self.cleaned = True


def make_engine(config=None):
    engine = object.__new__(InferenceEngine)
    engine.config = config or SimpleNamespace(
        TOP_K_PREDICTIONS=1,
        MAX_IMAGE_DIMENSIONS=(10, 10),
        MAX_BATCH_SIZE=2,
        BATCHING_TIMEOUT_MS=1,
        validate_image=lambda image, max_dims: True,
    )  # pyright: ignore [reportArgumentType] # type: ignore
    engine.manager = (  # pyright: ignore [reportArgumentType] # type: ignore
        FakeManager()
    )
    engine.inference_queue = queue.Queue(maxsize=4)
    engine.shutdown_event = threading.Event()
    engine.ready = False
    engine.inference_thread = SimpleNamespace(
        is_alive=lambda: False, join=lambda timeout=None: None
    )  # pyright: ignore [reportArgumentType] # type: ignore
    return engine


def new_future():
    loop = asyncio.new_event_loop()
    try:
        return loop, loop.create_future()
    except Exception:
        loop.close()
        raise


def close_future_loop(loop):
    loop.close()


def test_safely_set_result_ignores_completed_future():
    loop, future = new_future()
    try:
        future.set_result(([("old", 1.0)], 1.0))
        safely_set_result(future, ([("new", 2.0)], 2.0))
        assert future.result() == ([("old", 1.0)], 1.0)
    finally:
        close_future_loop(loop)


def test_safely_fail_future_ignores_completed_future():
    loop, future = new_future()
    try:
        future.set_result(([("ok", 1.0)], 1.0))
        safely_fail_future(future, RuntimeError("late"))
        assert future.result() == ([("ok", 1.0)], 1.0)
    finally:
        close_future_loop(loop)


def test_enqueue_request_puts_future_in_queue():
    engine = make_engine()

    async def run():
        image = Image.new("RGB", (1, 1))
        future = engine.EnqueueRequest(image)
        queued_image, queued_loop, queued_future = engine.inference_queue.get_nowait()
        assert queued_image is image
        assert queued_future is future
        assert queued_loop is asyncio.get_running_loop()

    asyncio.run(run())


def test_enqueue_request_raises_when_queue_full():
    engine = make_engine()
    engine.inference_queue = queue.Queue(maxsize=1)
    engine.inference_queue.put_nowait("already-full")

    async def run():
        with pytest.raises(queue.Full):
            engine.EnqueueRequest(Image.new("RGB", (1, 1)))

    asyncio.run(run())


def test_perform_inference_sets_results_for_valid_batch():
    engine = make_engine()
    loop1, future1 = new_future()
    loop2, future2 = new_future()
    try:
        engine.perform_inference(
            [Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))],
            [
                ImmediateLoop(),
                ImmediateLoop(),
            ],  # pyright: ignore [reportArgumentType] # type: ignore
            [future1, future2],
        )

        assert future1.result() == ([("class", 0.0)], 5.0)
        assert future2.result() == ([("class", 1.0)], 5.0)
    finally:
        close_future_loop(loop1)
        close_future_loop(loop2)


def test_perform_inference_isolates_preprocess_failure():
    engine = make_engine()
    good = Image.new("RGB", (1, 1))
    bad = Image.new("RGB", (1, 1))
    bad.should_fail_preprocess = (  # pyright: ignore [reportArgumentType] # type: ignore
        True
    )
    loop1, future1 = new_future()
    loop2, future2 = new_future()
    try:
        engine.perform_inference(
            [bad, good],
            [
                ImmediateLoop(),
                ImmediateLoop(),
            ],  # pyright: ignore [reportArgumentType] # type: ignore
            [future1, future2],
        )

        assert isinstance(future1.exception(), RuntimeError)
        assert future2.result() == ([("class", 0.0)], 5.0)
    finally:
        close_future_loop(loop1)
        close_future_loop(loop2)


@pytest.mark.parametrize("failure_attr", ["fail_predict", "fail_top_k"])
def test_perform_inference_fails_valid_futures_when_batch_stage_fails(failure_attr):
    engine = make_engine()
    setattr(engine.manager, failure_attr, True)
    loop, future = new_future()
    try:
        engine.perform_inference(
            [Image.new("RGB", (1, 1))],
            [ImmediateLoop()],  # pyright: ignore [reportArgumentType] # type: ignore
            [future],
        )

        assert isinstance(future.exception(), RuntimeError)
    finally:
        close_future_loop(loop)


def test_batch_processing_loop_processes_partial_batch_and_preserves_order():
    engine = make_engine(
        SimpleNamespace(
            TOP_K_PREDICTIONS=1,
            MAX_IMAGE_DIMENSIONS=(10, 10),
            MAX_BATCH_SIZE=2,
            BATCHING_TIMEOUT_MS=1,
            validate_image=lambda image, max_dims: True,
        )
    )
    loop1, future1 = new_future()
    loop2, future2 = new_future()
    try:
        engine.inference_queue.put((Image.new("RGB", (1, 1)), ImmediateLoop(), future1))
        engine.inference_queue.put((Image.new("RGB", (1, 1)), ImmediateLoop(), future2))
        engine.inference_queue.put(None)

        engine.batch_processing_loop()

        assert engine.ready is True
        assert future1.result() == ([("class", 0.0)], 5.0)
        assert future2.result() == ([("class", 1.0)], 5.0)
    finally:
        close_future_loop(loop1)
        close_future_loop(loop2)


def test_batch_processing_loop_rejects_invalid_image_dimensions():
    engine = make_engine(
        SimpleNamespace(
            TOP_K_PREDICTIONS=1,
            MAX_IMAGE_DIMENSIONS=(10, 10),
            MAX_BATCH_SIZE=2,
            BATCHING_TIMEOUT_MS=1,
            validate_image=lambda image, max_dims: False,
        )
    )
    loop, future = new_future()
    try:
        engine.inference_queue.put((Image.new("RGB", (1, 1)), ImmediateLoop(), future))
        engine.inference_queue.put(None)

        engine.batch_processing_loop()

        assert isinstance(future.exception(), ValueError)
    finally:
        close_future_loop(loop)


def test_shutdown_signals_thread_and_cleans_model():
    joined = []
    engine = make_engine()
    engine.inference_thread = SimpleNamespace(
        join=lambda timeout=None: joined.append(timeout), is_alive=lambda: False
    )  # pyright: ignore [reportArgumentType] # type: ignore

    engine.shutdown()

    assert engine.shutdown_event.is_set()
    assert joined == [5.0]
    assert (
        engine.manager.cleaned  # pyright: ignore [reportArgumentType] # type: ignore
        is True
    )


def test_shutdown_handles_full_queue_without_crashing():
    engine = make_engine()
    engine.inference_queue = queue.Queue(maxsize=1)
    engine.inference_queue.put_nowait("full")
    engine.inference_thread = SimpleNamespace(
        join=lambda timeout=None: None, is_alive=lambda: False
    )  # pyright: ignore [reportArgumentType] # type: ignore

    engine.shutdown()

    assert (
        engine.manager.cleaned  # pyright: ignore [reportArgumentType] # type: ignore
        is True
    )


def test_init_model_initializes_and_loads_model(monkeypatch):
    class FakeModelManager:
        def __init__(self, model_name, inference_device, model_path):
            self.model_name = model_name
            self.inference_device = inference_device
            self.model_loaded = False

        def load_model(self):
            self.model_loaded = True

        def cleanup_model(self):
            self.model_loaded = False

    FakeConfig = SimpleNamespace(
        MODEL_NAME="microsoft/resnet-50",
        INFERENCE_DEVICE="cpu",
        MODEL_PATH=None,
        MAX_CONCURRENT_REQUESTS=256,
    )

    monkeypatch.setattr("src.inference_engine.ModelManager", FakeModelManager)

    inf_engine = InferenceEngine(
        FakeConfig  # pyright: ignore [reportArgumentType] # type: ignore
    )

    inf_engine.init_model_manager()

    assert isinstance(inf_engine.manager, FakeModelManager)
    assert inf_engine.manager.model_name == "microsoft/resnet-50"
    assert inf_engine.manager.model_loaded is True
