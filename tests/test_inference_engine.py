from types import SimpleNamespace
from inference_engine import InferenceEngine


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

    monkeypatch.setattr("inference_engine.ModelManager", FakeModelManager)

    inf_engine = InferenceEngine(FakeConfig)  # pyright: ignore [reportArgumentType]

    inf_engine.init_model_manager()

    assert isinstance(inf_engine.manager, FakeModelManager)
    assert inf_engine.manager.model_name == "microsoft/resnet-50"
    assert inf_engine.manager.model_loaded is True
