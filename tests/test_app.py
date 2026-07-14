import asyncio
import io
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

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
    app_module.app.state.inf_engine = None
    yield


def test_handle_predict_request_rejects_shutdown_and_non_image():
    file = UploadFile(filename="image.jpg", file=io.BytesIO(b"dummy"))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.handle_predict_request(file))
    assert exc_info.value.status_code == 503

    app_module.app.state.inf_engine = SimpleNamespace(ready=True)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_module.handle_predict_request(file))
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

    monkeypatch.setattr("app.InferenceEngine", FakeInferenceEngine)

    async def run_lifespan():
        async with app_module.lifespan(app_module.app):
            return None

    asyncio.run(run_lifespan())
    assert shutdown_called is True
