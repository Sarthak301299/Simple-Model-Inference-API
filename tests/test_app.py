import asyncio
import io
from types import SimpleNamespace
import queue
import pytest
import threading
from fastapi import HTTPException, UploadFile, Request
from PIL import Image
from starlette.datastructures import Headers

import src.app as app_module


def create_event(should_set: bool) -> threading.Event:
    event = threading.Event()
    if should_set:
        event.set()
    return event


def make_png(size=(2, 2), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size, color=0).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@pytest.fixture(autouse=True)
def reset_app_state():
    app_module.app.state.config = SimpleNamespace(
        API_KEY=None,
        API_RETRY=7,
        MAX_FILE_SIZE_MB=1,
        MAX_CHUNK_SIZE_MB=1,
        MAX_IMAGE_DIMENSIONS=(32, 32),
        validate_image=lambda image, maxdims: 0 < image.width <= maxdims[0]
        and 0 < image.height <= maxdims[1],
    )
    app_module.app.state.inf_engine = None
    yield


def test_verify_api_key_allows_when_not_configured():
    asyncio.run(app_module.verify_api_key(None))


@pytest.mark.parametrize("provided", [None, "wrong"])
def test_verify_api_key_rejects_missing_or_invalid_key(provided):
    app_module.app.state.config.API_KEY = "secret"

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.verify_api_key(provided))

    assert exc_info.value.status_code == 401


def test_verify_api_key_accepts_matching_key():
    app_module.app.state.config.API_KEY = "secret"

    asyncio.run(app_module.verify_api_key("secret"))


@pytest.mark.parametrize(
    "engine,shutdown,expected_status",
    [
        (None, False, 500),
        (
            SimpleNamespace(
                inference_thread=SimpleNamespace(
                    is_alive=lambda: False
                ), shutdown_event=threading.Event()
            ),
            False,
            500,
        ),
        (
            SimpleNamespace(
                inference_thread=SimpleNamespace(
                    is_alive=lambda: True
                ), shutdown_event=threading.Event()
            ),
            True,
            500,
        ),
        (
            SimpleNamespace(
                inference_thread=SimpleNamespace(
                    is_alive=lambda: True
                ), shutdown_event=threading.Event()
            ),
            False,
            200,
        ),
    ],
)
def test_liveness_states(engine, shutdown, expected_status):
    app_module.app.state.inf_engine = engine
    if app_module.app.state.inf_engine:
        if shutdown:
            app_module.app.state.inf_engine.shutdown_event.set()
    response = asyncio.run(app_module.liveness_check())

    assert response.status_code == expected_status


@pytest.mark.parametrize(
    "engine,shutdown,queuefull,expected_status",
    [
        (None, False, False, 503),
        (
            SimpleNamespace(
                ready=False,
                manager=SimpleNamespace(model_loaded=True),
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
                inference_queue=queue.Queue(maxsize=1),
            ),
            False,
            False,
            503,
        ),
        (
            SimpleNamespace(
                ready=True,
                manager=SimpleNamespace(model_loaded=False),
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
                inference_queue=queue.Queue(maxsize=1),
            ),
            False,
            False,
            503,
        ),
        (
            SimpleNamespace(
                ready=True,
                manager=SimpleNamespace(model_loaded=True),
                inference_thread=SimpleNamespace(is_alive=lambda: False),
                shutdown_event=threading.Event(),
                inference_queue=queue.Queue(maxsize=1),
            ),
            False,
            False,
            503,
        ),
        (
            SimpleNamespace(
                ready=True,
                manager=SimpleNamespace(model_loaded=True),
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
                inference_queue=queue.Queue(maxsize=1),
            ),
            True,
            False,
            503,
        ),
        (
            SimpleNamespace(
                ready=True,
                manager=SimpleNamespace(model_loaded=True),
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
                inference_queue=queue.Queue(maxsize=1),
            ),
            False,
            True,
            503,
        ),
        (
            SimpleNamespace(
                ready=True,
                manager=SimpleNamespace(model_loaded=True),
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
                inference_queue=queue.Queue(maxsize=1),
            ),
            False,
            False,
            200,
        ),
    ],
)
def test_readiness_states(engine, shutdown, queuefull, expected_status):
    app_module.app.state.inf_engine = engine

    if app_module.app.state.inf_engine:
        if shutdown:
            app_module.app.state.inf_engine.shutdown_event.set()
        if queuefull:
            app_module.app.state.inf_engine.inference_queue.put("item")

    response = asyncio.run(app_module.readyness_check())

    assert response.status_code == expected_status


@pytest.mark.parametrize(
    "engine,shutdown,expected_status",
    [
        (None, False, 503),
        (
            SimpleNamespace(
                ready=False,
                manager=SimpleNamespace(model_loaded=True),
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
            ),
            False,
            503,
        ),
        (
            SimpleNamespace(
                ready=True,
                manager=SimpleNamespace(model_loaded=False),
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
            ),
            False,
            503,
        ),
        (
            SimpleNamespace(
                ready=True,
                manager=SimpleNamespace(model_loaded=True),
                inference_thread=SimpleNamespace(is_alive=lambda: False),
                shutdown_event=threading.Event(),
            ),
            False,
            503,
        ),
        (
            SimpleNamespace(
                ready=True,
                manager=SimpleNamespace(model_loaded=True),
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
            ),
            True,
            503,
        ),
        (
            SimpleNamespace(
                ready=True,
                manager=SimpleNamespace(model_loaded=True),
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
            ),
            False,
            200,
        ),
    ],
)
def test_startup_states(engine, shutdown, expected_status):
    app_module.app.state.inf_engine = engine

    if app_module.app.state.inf_engine:
        if shutdown:
            app_module.app.state.inf_engine.shutdown_event.set()

    response = asyncio.run(app_module.startup_check())

    assert response.status_code == expected_status


def test_info_returns_model_metadata():
    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True,
        manager=SimpleNamespace(
            get_model_info=lambda: {"name": "fake", "device": "cpu"}
        ),
    )

    response = asyncio.run(app_module.info())

    assert response.status_code == 200
    assert response.body == b'{"name":"fake","device":"cpu"}'


@pytest.mark.parametrize(
    "len, exception, status_code",
    [("oops", ValueError, None), ("1048577", HTTPException, 413)],
)
def test_predict_rejects_invalid_content_length(len, exception, status_code):
    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True, inference_queue=queue.Queue()
    )
    request = Request(
        scope={"type": "http", "headers": [(b"content-length", len.encode("utf-8"))]}
    )
    file = UploadFile(
        filename="x.png",
        file=make_png(),
        headers=Headers({"content-type": "image/png"}),
    )

    with pytest.raises(exception) as exc:
        asyncio.run(app_module.handle_predict_request(request, file))

    if exc.value == HTTPException:
        assert exc.value.status_code == 413


def test_predict_rejects_invalid_image_bytes():
    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True, inference_queue=queue.Queue()
    )
    request = Request(scope={"type": "http", "headers": []})
    file = UploadFile(
        filename="x.png",
        file=io.BytesIO(b"not-image"),
        headers=Headers({"content-type": "image/png"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.handle_predict_request(request, file))

    assert exc_info.value.status_code == 400


def test_predict_rejects_too_large_dimensions():
    app_module.app.state.config.MAX_IMAGE_DIMENSIONS = (1, 1)
    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True, inference_queue=queue.Queue()
    )
    request = Request(scope={"type": "http", "headers": []})
    file = UploadFile(
        filename="x.png",
        file=make_png(size=(2, 2)),
        headers=Headers({"content-type": "image/png"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.handle_predict_request(request, file))

    assert exc_info.value.status_code == 400


def test_predict_returns_prediction_payload():
    async def enqueue(image):
        return ([("label", 0.9)], 12.5)

    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True, inference_queue=queue.Queue(), EnqueueRequest=enqueue
    )
    request = Request(scope={"type": "http", "headers": []})
    file = UploadFile(
        filename="x.png",
        file=make_png(),
        headers=Headers({"content-type": "image/png"}),
    )

    result = asyncio.run(app_module.handle_predict_request(request, file))

    assert result == {"prediction": [("label", 0.9)], "inference_time_ms": 12.5}


def test_predict_rejects_full_queue_before_reading_body():
    full_queue = queue.Queue(maxsize=1)
    full_queue.put("item")
    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True, inference_queue=full_queue
    )
    request = Request(scope={"type": "http", "headers": []})
    file = UploadFile(
        filename="x.png",
        file=make_png(),
        headers=Headers({"content-type": "image/png"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.handle_predict_request(request, file))

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "7"}


def test_predict_rejects_queue_full_during_enqueue():
    async def enqueue(_image):
        raise queue.Full

    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True, inference_queue=queue.Queue(), EnqueueRequest=enqueue
    )
    request = Request(scope={"type": "http", "headers": []})
    file = UploadFile(
        filename="x.png",
        file=make_png(),
        headers=Headers({"content-type": "image/png"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.handle_predict_request(request, file))

    assert exc_info.value.status_code == 429


def test_read_with_limits_allows_exact_limit():
    app_module.app.state.config.MAX_CHUNK_SIZE_MB = 1
    file = UploadFile(filename="payload", file=io.BytesIO(b"abc"))

    assert asyncio.run(app_module.read_with_limits(file, 3)) == b"abc"


def test_read_with_limits_rejects_limit_plus_one():
    app_module.app.state.config.MAX_CHUNK_SIZE_MB = 1
    file = UploadFile(filename="payload", file=io.BytesIO(b"abcd"))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.read_with_limits(file, 3))

    assert exc_info.value.status_code == 413


def test_root_endpoint_direct():
    assert app_module.read_root() == {"message": "Hello World"}


def test_handle_predict_request_rejects_shutdown_and_non_image():
    file = UploadFile(filename="image.jpg", file=io.BytesIO(b"dummy"))
    request = Request(scope={"type": "http", "headers": []})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.handle_predict_request(request, file))
    assert exc_info.value.status_code == 503

    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True, inference_queue=queue.Queue()
    )
    app_module.app.state.config = SimpleNamespace(
        MODEL_NAME="microsoft/resnet-50", INFERENCE_DEVICE="cpu", MAX_FILE_SIZE_MB=16
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.handle_predict_request(request, file))
    assert exc_info.value.status_code == 400


def test_lifespan_runs_init_and_cleanup(monkeypatch):
    shutdown_called = False

    class FakeInferenceEngine:
        def __init__(self, config):
            self.engine_started = True
            self.config = config

        def shutdown(self):
            nonlocal shutdown_called
            shutdown_called = True

    monkeypatch.setattr("src.app.InferenceEngine", FakeInferenceEngine)

    async def run_lifespan():
        async with app_module.lifespan(app_module.app):
            return None

    asyncio.run(run_lifespan())
    assert shutdown_called is True
