import asyncio
import io
from types import SimpleNamespace
from PIL import Image
import torch
from inference_engine import InferenceEngine


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

    FakeConfig = SimpleNamespace(
        MODEL_NAME="microsoft/resnet-50",
        INFERENCE_DEVICE="cpu",
        MAX_CONCURRENT_REQUESTS=256
    )

    monkeypatch.setattr("inference_engine.ModelManager", FakeModelManager)

    inf_engine = InferenceEngine(FakeConfig)  # pyright: ignore [reportArgumentType]

    inf_engine.init_model_manager()

    assert isinstance(inf_engine.manager, FakeModelManager)
    assert inf_engine.manager.model_name == "microsoft/resnet-50"
    assert inf_engine.manager.model_loaded is True


def test_perform_inference_processes_uploaded_image_bytes(monkeypatch):
    class FakeModelManager:
        def __init__(self, model_name, inference_device):
            self.model_name = model_name
            self.inference_device = inference_device
            self.model_loaded = False

        def load_model(self):
            self.model_loaded = True

        def preprocess_inputs(self, images):
            return torch.randn(1, 3, 224,224)

        def predict(self, processed_inputs):
            return [processed_inputs], 12.42

        def top_k_from_logits(self, logits, k):
            return [[("cat", 0.99)]]

    image = Image.new("RGB", (8, 8), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    FakeConfig = SimpleNamespace(
        MODEL_NAME="microsoft/resnet-50", INFERENCE_DEVICE="cpu", TOP_K_PREDICTIONS=1, MAX_CONCURRENT_REQUESTS=256
    )
    FakeLoop = asyncio.new_event_loop()
    FakeFuture = FakeLoop.create_future()

    monkeypatch.setattr("inference_engine.ModelManager", FakeModelManager)

    inf_engine = InferenceEngine(FakeConfig)  # pyright: ignore [reportArgumentType]

    result = inf_engine.perform_inference([image], [FakeLoop], [FakeFuture])

    assert result == ([[("cat", 0.99)]], 12.42)
