import asyncio
import io
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

import app as app_module


@pytest.fixture(autouse=True)
def reset_app_state():
    app_module.app.state.config = SimpleNamespace(
        TOP_K_PREDICTIONS=3,
        MAX_BATCH_SIZE=2,
        BATCHING_TIMEOUT_MS=1,
        API_VERSION="1.0.0",
        API_HOST="127.0.0.1",
        API_PORT=8000,
        LOG_LEVEL="INFO",
    )
    app_module.app.state.model_manager = None
    app_module.app.state.inference_queue = asyncio.Queue(maxsize=4)
    app_module.app.state.shutdown_event = asyncio.Event()
    yield


def test_init_model_initializes_and_loads_model(monkeypatch):
    class FakeModelManager:
        def __init__(self, model_name, inference_device):
            self.model_name = model_name
            self.inference_device = inference_device
            self.model_loaded = False

        def load_model(self):
            self.model_loaded = True

        def cleanup_model(self):
            self.model_loaded = False

    monkeypatch.setattr(app_module, "ModelManager", FakeModelManager)
    app_module.app.state.config = SimpleNamespace(
        MODEL_NAME="microsoft/resnet-50",
        INFERENCE_DEVICE="cpu",
    )

    app_module.init_model()

    assert isinstance(app_module.app.state.model_manager, FakeModelManager)
    assert app_module.app.state.model_manager.model_name == "microsoft/resnet-50"
    assert app_module.app.state.model_manager.model_loaded is True


def test_perform_inference_processes_uploaded_image_bytes():
    class FakeModelManager:
        def preprocess_inputs(self, images):
            return [images]

        def predict(self, processed_inputs):
            return [processed_inputs], 12.42

        def top_k_from_logits(self, logits, k):
            return [[("cat", 0.99)]]

    image = Image.new("RGB", (8, 8), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    app_module.app.state.config = SimpleNamespace(TOP_K_PREDICTIONS=1)
    app_module.app.state.model_manager = FakeModelManager()

    result = asyncio.run(app_module.perform_inference([image]))

    assert result == ([[("cat", 0.99)]], 12.42)


def test_perform_inference_raises_when_model_manager_is_missing():
    image = Image.new("RGB", (4, 4), color="blue")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    app_module.app.state.model_manager = None
    app_module.app.state.config = SimpleNamespace(TOP_K_PREDICTIONS=1)

    with pytest.raises(ValueError):
        asyncio.run(app_module.perform_inference([image]))


def test_health_endpoints_report_expected_states():
    response = app_module.liveness_check()
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert response.body == b'{"status":"ALIVE"}'

    app_module.app.state.shutdown_event.set()
    assert asyncio.run(app_module.readyness_check()).status_code == 503

    app_module.app.state.shutdown_event = asyncio.Event()
    app_module.app.state.model_manager = None
    assert asyncio.run(app_module.readyness_check()).status_code == 503

    app_module.app.state.model_manager = SimpleNamespace(model_loaded=True)
    app_module.app.state.inference_queue = SimpleNamespace(full=lambda: True)
    assert asyncio.run(app_module.readyness_check()).status_code == 503

    app_module.app.state.inference_queue = SimpleNamespace(full=lambda: False)
    assert asyncio.run(app_module.readyness_check()).status_code == 200

    app_module.app.state.model_manager = SimpleNamespace(model_loaded=True)
    assert app_module.startup_check().status_code == 200

    app_module.app.state.model_manager = SimpleNamespace(model_loaded=False)
    assert app_module.startup_check().status_code == 503


def test_handle_predict_request_rejects_shutdown_and_queue_full():
    file = UploadFile(filename="image.jpg", file=io.BytesIO(b"dummy"))

    app_module.app.state.shutdown_event.set()
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.handle_predict_request(file, "dummy"))
    assert exc_info.value.status_code == 503

    class FullQueue:
        def put_nowait(self, item):
            raise asyncio.QueueFull

    app_module.app.state.shutdown_event = asyncio.Event()
    app_module.app.state.inference_queue = FullQueue()
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.handle_predict_request(file, "dummy"))
    assert exc_info.value.status_code == 429


def test_get_image_from_upload_file_rejects_non_images():
    async def run_test():
        future = asyncio.get_running_loop().create_future()
        upload_file = UploadFile(filename="bad.jpg", file=io.BytesIO(b"not an image"))
        with pytest.raises(HTTPException) as exc_info:
            await app_module.get_image_from_upload_file((upload_file, future))
        assert exc_info.value.status_code == 400
        assert future.done() is True
        assert isinstance(future.exception(), HTTPException)

    asyncio.run(run_test())


def test_lifespan_runs_init_and_cleanup(monkeypatch):
    cleanup_called = False

    class FakeModelManager:
        def __init__(self):
            self.model_loaded = True

        def cleanup_model(self):
            nonlocal cleanup_called
            cleanup_called = True

    async def fake_batch_processing_loop():
        return None

    monkeypatch.setattr(app_module, "init_model", lambda: None)
    monkeypatch.setattr(app_module, "batch_processing_loop", fake_batch_processing_loop)
    app_module.app.state.model_manager = FakeModelManager()
    app_module.app.state.shutdown_event = asyncio.Event()

    async def run_lifespan():
        async with app_module.lifespan(app_module.app):
            return None

    asyncio.run(run_lifespan())
    assert cleanup_called is True


def test_batch_processing_loop_handles_timeout_and_error(monkeypatch):
    calls = 0

    async def fake_wait_for(coro, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            app_module.app.state.shutdown_event.set()
            if hasattr(coro, "close"):
                coro.close()
            raise asyncio.TimeoutError
        return await coro

    async def fake_perform_inference(batch_inputs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.asyncio, "wait_for", fake_wait_for)
    monkeypatch.setattr(app_module, "perform_inference", fake_perform_inference)
    app_module.app.state.shutdown_event = asyncio.Event()
    app_module.app.state.inference_queue = asyncio.Queue()

    asyncio.run(app_module.batch_processing_loop())


def test_handle_predict_request_returns_prediction_for_enqueued_work():
    class ImmediateQueue:
        def put_nowait(self, item):
            uploaded_file, ticket = item
            assert uploaded_file.filename == "image.jpg"
            ticket.set_result(([("bird", 0.95)], 12.42))

    app_module.app.state.inference_queue = ImmediateQueue()
    app_module.app.state.shutdown_event = asyncio.Event()
    file = UploadFile(filename="image.jpg", file=io.BytesIO(b"dummy"))

    response = asyncio.run(app_module.handle_predict_request(file, "metadata"))

    assert response == {"prediction": [("bird", 0.95)], "inference_time_ms": 12.42}
