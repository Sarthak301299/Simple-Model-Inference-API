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
from src.config import Config


def make_health_engine(ready=True, model_loaded=True, thread_alive=True):
    return SimpleNamespace(
        ready=ready,
        manager=SimpleNamespace(model_loaded=model_loaded),
        inference_thread=SimpleNamespace(is_alive=lambda: thread_alive),
        shutdown_event=threading.Event(),
        inference_queue=queue.Queue(maxsize=1),
    )


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
        INFERENCE_TIMEOUT=60,
        API_HOST="0.0.0.0",
        API_PORT=8000,
        LOG_LEVEL="INFO",
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine,shutdown,expected_status",
    [
        (None, False, 500),
        (
            SimpleNamespace(
                inference_thread=SimpleNamespace(is_alive=lambda: False),
                shutdown_event=threading.Event(),
            ),
            False,
            500,
        ),
        (
            SimpleNamespace(
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
            ),
            True,
            500,
        ),
        (
            SimpleNamespace(
                inference_thread=SimpleNamespace(is_alive=lambda: True),
                shutdown_event=threading.Event(),
            ),
            False,
            200,
        ),
    ],
)
async def test_liveness_states(engine, shutdown, expected_status):
    app_module.app.state.inf_engine = engine
    if app_module.app.state.inf_engine:
        if shutdown:
            app_module.app.state.inf_engine.shutdown_event.set()
    response = await app_module.liveness_check()

    assert response.status_code == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "check_fn, engine, shutdown, queuefull, expected_status",
    [
        (app_module.readyness_check, None, False, False, 503),
        (app_module.startup_check, None, False, False, 503),
        (
            app_module.readyness_check,
            make_health_engine(ready=False),
            False,
            False,
            503,
        ),
        (app_module.startup_check, make_health_engine(ready=False), False, False, 503),
        (
            app_module.readyness_check,
            make_health_engine(model_loaded=False),
            False,
            False,
            503,
        ),
        (
            app_module.startup_check,
            make_health_engine(model_loaded=False),
            False,
            False,
            503,
        ),
        (
            app_module.readyness_check,
            make_health_engine(thread_alive=False),
            False,
            False,
            503,
        ),
        (
            app_module.startup_check,
            make_health_engine(thread_alive=False),
            False,
            False,
            503,
        ),
        (app_module.readyness_check, make_health_engine(), True, False, 503),
        (app_module.startup_check, make_health_engine(), True, False, 503),
        (app_module.readyness_check, make_health_engine(), False, False, 200),
        (app_module.startup_check, make_health_engine(), False, False, 200),
        (
            app_module.readyness_check,
            make_health_engine(),
            False,
            True,
            503,
        ),  # readiness-only: queue full
    ],
)
async def test_health_endpoint_states(
    check_fn, engine, shutdown, queuefull, expected_status
):
    app_module.app.state.inf_engine = engine
    if engine:
        if shutdown:
            engine.shutdown_event.set()
        if queuefull:
            engine.inference_queue.put("item")

    response = await check_fn()

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
    "content_length, exception, status_code",
    [("oops", HTTPException, 400), ("1048577", HTTPException, 413)],
)
def test_predict_rejects_invalid_content_length(content_length, exception, status_code):
    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True, inference_queue=queue.Queue()
    )
    request = Request(
        scope={
            "type": "http",
            "headers": [(b"content-length", content_length.encode("utf-8"))],
        }
    )
    file = UploadFile(
        filename="x.png",
        file=make_png(),
        headers=Headers({"content-type": "image/png"}),
    )

    with pytest.raises(exception) as exc:
        asyncio.run(app_module.handle_predict_request(request, file))

    if status_code is not None:
        assert exc.value.status_code == status_code


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

    app_module.app.state.config = Config()

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

    app_module.app.state.config = Config()

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


@pytest.mark.asyncio
async def test_handle_predict_request_rejects_shutdown_and_non_image():
    file = UploadFile(filename="image.jpg", file=io.BytesIO(b"dummy"))
    request = Request(scope={"type": "http", "headers": []})

    with pytest.raises(HTTPException) as exc_info:
        await app_module.handle_predict_request(request, file)
    assert exc_info.value.status_code == 503


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


@pytest.mark.asyncio
async def test_metrics_middleware_completion():
    temp = SimpleNamespace(status_code=200)

    async def tempf(input):
        return temp

    resp = await app_module.metrics_middleware(
        request=SimpleNamespace(
            url=SimpleNamespace(path="/notmetrics")
        ),  # pyright: ignore[reportArgumentType] # type: ignore
        call_next=tempf,
    )
    assert temp == resp


@pytest.mark.asyncio
async def test_metrics_middleware_returns_on_metric_endpoint():
    temp = SimpleNamespace(status_code=200)

    async def tempf(input):
        return temp

    resp = await app_module.metrics_middleware(
        request=SimpleNamespace(
            url=SimpleNamespace(path="/metrics")
        ),  # pyright: ignore[reportArgumentType] # type: ignore
        call_next=tempf,
    )
    assert temp == resp


@pytest.mark.asyncio
async def test_metrics_returns_successfully():
    return await app_module.metrics()


@pytest.mark.asyncio
async def test_info_fails_without_engine():
    response = await app_module.info()
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_predict_raises_httpexception_when_bad_content_type():
    app_module.app.state.inf_engine = SimpleNamespace(ready=True)
    request = Request(scope={"type": "http"})
    file = UploadFile(
        filename="x.png",
        file=make_png(),
        headers=Headers({"content-type": "text"}),
    )
    with pytest.raises(HTTPException) as exc:
        await app_module.handle_predict_request(request, file)

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_predict_propagates_413_from_read_with_limits_without_rewrapping():
    app_module.app.state.config.MAX_FILE_SIZE_MB = 1
    app_module.app.state.config.MAX_CHUNK_SIZE_MB = 1
    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True, inference_queue=queue.Queue()
    )
    oversized = io.BytesIO(
        b"x" * (2 * 1024 * 1024)
    )  # 2MB body, no Content-Length header set
    request = Request(scope={"type": "http", "headers": []})
    file = UploadFile(
        filename="x.png", file=oversized, headers=Headers({"content-type": "image/png"})
    )

    with pytest.raises(HTTPException) as exc_info:
        await app_module.handle_predict_request(request, file)

    assert exc_info.value.status_code == 413


def test_main_calls_uvicorn_run_with_configured_settings(monkeypatch):
    calls = {}

    def fake_run(app, host, port, log_level):
        calls["host"] = host
        calls["port"] = port
        calls["log_level"] = log_level

    monkeypatch.setattr(app_module.uvicorn, "run", fake_run)

    app_module.main()

    assert calls["host"] == app_module.app.state.config.API_HOST
    assert calls["port"] == app_module.app.state.config.API_PORT
    assert calls["log_level"] == app_module.app.state.config.LOG_LEVEL


@pytest.mark.asyncio
async def test_predict_succeeds_within_timeout():
    async def enqueue(image):
        return ([("label", 0.9)], 12.5)

    app_module.app.state.config.INFERENCE_TIMEOUT = 5.0
    app_module.app.state.inf_engine = SimpleNamespace(
        ready=True, inference_queue=queue.Queue(), EnqueueRequest=enqueue
    )
    request = Request(scope={"type": "http", "headers": []})
    file = UploadFile(
        filename="x.png",
        file=make_png(),
        headers=Headers({"content-type": "image/png"}),
    )

    result = await app_module.handle_predict_request(request, file)

    assert result == {"prediction": [("label", 0.9)], "inference_time_ms": 12.5}


@pytest.mark.asyncio
async def test_predict_raises_504_when_enqueue_future_never_resolves():
    """Simulates a future whose notifying loop is closed (or a stuck worker) —
    the caller must not hang forever; asyncio.wait_for is what actually
    guarantees that, independent of anything perform_inference/
    batch_processing_loop can do across threads."""

    def enqueue(image):
        loop = asyncio.get_running_loop()
        return loop.create_future()  # deliberately never resolved

    app_module.app.state.config.INFERENCE_TIMEOUT = 0.05
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
        await app_module.handle_predict_request(request, file)

    assert exc_info.value.status_code == 504
