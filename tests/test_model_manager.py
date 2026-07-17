import types
from types import SimpleNamespace
import pytest
import torch
import json
import numpy as np
from PIL import Image

from src.model_manager import TorchModelManager, ONNXModelManager, ModelManager


def test_model_manager_cannot_be_instantiated_directly():
    """ABC guard: only concrete subclasses should be constructible."""
    with pytest.raises(TypeError):
        ModelManager(
            model_name="whatever"
        )  # pyright: ignore[reportArgumentType] # type: ignore


def test_torch_model_manager_is_a_model_manager():
    assert issubclass(TorchModelManager, ModelManager)
    assert issubclass(ONNXModelManager, ModelManager)


def test_subclass_missing_load_model_cannot_be_instantiated():
    """Guards the ABC contract itself: a subclass that forgets to implement
    `predict` (or `load_model`) should fail at construction, not silently
    inherit a no-op — this is the exact mistake a future ONNXModelManager-
    style addition could make."""

    class IncompleteManager(ModelManager):
        def load_model(self):
            pass

        # predict() deliberately not implemented

    with pytest.raises(TypeError):
        IncompleteManager(
            model_name="whatever"
        )  # pyright: ignore[reportArgumentType] # type: ignore


def test_preprocess_inputs_and_top_k_are_shared_not_overridden():
    """Documents intent: these live on the base class and every backend
    should get identical behavior without re-implementing them."""
    assert "preprocess_inputs" not in TorchModelManager.__dict__
    assert "top_k_from_logits" not in TorchModelManager.__dict__
    assert "preprocess_inputs" not in ONNXModelManager.__dict__
    assert "top_k_from_logits" not in ONNXModelManager.__dict__


class FakeProcessor:
    def __init__(self):
        self.called_with = None

    def __call__(self, images, return_tensors="pt"):
        self.called_with = list(images)
        return {"pixel_values": torch.zeros(len(images), 3, 224, 224)}


class FakeModel:
    def __init__(self, id2label=None):
        self.config = SimpleNamespace(
            id2label=id2label or {0: "zero", 1: "one", 2: "two"}
        )
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        pass

    def __call__(self, inputs):
        return SimpleNamespace(logits=torch.tensor([[0.1, 2.0, 0.5]] * inputs.shape[0]))


class FakeInferenceSession:
    """Stands in for onnxruntime.InferenceSession."""

    def __init__(self, path, providers=None):
        self.path = path
        self.providers = providers
        self.run_calls = []

    def run(self, output_names, input_feed):
        self.run_calls.append((output_names, input_feed))
        batch_size = input_feed["pixel_values"].shape[0]
        num_classes = 3
        # deterministic, distinguishable-per-row logits, same convention as
        # FakeHFModel in the torch-backend tests, so parity-style comparisons
        # between backends are easy to write later.
        logits = np.stack(
            [
                np.full((num_classes,), float(i), dtype=np.float32)
                for i in range(batch_size)
            ]
        )
        return [logits]


@pytest.fixture
def fake_ort(monkeypatch):
    monkeypatch.setattr(
        "src.model_manager.onnxruntime",
        types.SimpleNamespace(
            InferenceSession=FakeInferenceSession,
        ),
    )


def test_load_model_creates_session_with_configured_providers_and_wrong_name(
    fake_ort, tmp_path
):
    onnx_path = tmp_path / "microsoft-resnet-50.onnx"
    json_path = tmp_path / "microsoft-resnet-50_config.json"
    onnx_path.write_bytes(b"fake onnx bytes")
    json_path.write_bytes(
        json.dumps({"id2label": {1: "cat", 2: "dog"}}).encode("utf-8")
    )

    manager = ONNXModelManager(
        model_name="microsoft/resnet-50",
        device="cpu",
        model_path=str(tmp_path),
        inference_backend="onnx",
    )
    manager.load_model()

    assert manager.model_loaded is True
    assert (
        manager.session.path  # pyright: ignore[reportArgumentType] # type: ignore
        == str(onnx_path)
    )
    assert (
        manager.session.providers  # pyright: ignore[reportArgumentType] # type: ignore
        == ["CPUExecutionProvider"]
    )


def test_load_model_raises_on_missing_json(fake_ort, tmp_path):
    onnx_path = tmp_path / "microsoft-resnet-50.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")

    manager = ONNXModelManager(
        model_name="microsoft/resnet-50",
        device="cpu",
        model_path=str(tmp_path),
        inference_backend="onnx",
    )
    with pytest.raises(FileNotFoundError):
        manager.load_model()


def test_load_model_defaults_to_cpu_provider_when_unset(fake_ort, tmp_path):
    onnx_path = tmp_path / "resnet-50.onnx"
    json_path = tmp_path / "resnet-50_config.json"
    onnx_path.write_bytes(b"fake onnx bytes")
    json_path.write_bytes(
        json.dumps({"id2label": {1: "cat", 2: "dog"}}).encode("utf-8")
    )

    manager = ONNXModelManager(model_name="resnet-50", model_path=str(tmp_path))
    manager.load_model()

    assert (
        manager.session.providers  # pyright: ignore[reportArgumentType] # type: ignore
        == ["CPUExecutionProvider"]
    )


def test_init_model_raises_clear_error_when_onnx_file_missing(fake_ort, tmp_path):

    with pytest.raises(FileNotFoundError, match="does not exist"):
        _ = ONNXModelManager(
            model_name="resnet-50",
            device="cpu",
            model_path=tmp_path,
            inference_backend="onnx",
        )


@pytest.mark.parametrize(
    "providers, session_path", [(None, "test.onnx"), (["CPUExecutionProvider"], None)]
)
def test_load_model_raises_value_error_when_invalid_setups(
    fake_ort, tmp_path, providers, session_path
):
    manager = object.__new__(ONNXModelManager)
    manager.model_name = "resnet-50"
    manager.model_path = str(tmp_path)
    manager.device = torch.device("cpu")
    manager.inference_backend = "onnx"
    manager.providers = providers
    manager.session_path = session_path

    with pytest.raises(ValueError):
        manager.load_model()


def test_load_model_raises_key_error_when_missing_id2label(
    fake_ort, tmp_path, monkeypatch
):
    onnx_path = tmp_path / "resnet-50.onnx"
    json_path = tmp_path / "resnet-50_config.json"
    onnx_path.write_bytes(b"fake onnx bytes")
    json_path.write_bytes(
        json.dumps({"id2label": {1: "cat", 2: "dog"}}).encode("utf-8")
    )
    monkeypatch.setattr("src.model_manager.json.load", lambda input: {"0": "cat"})
    manager = object.__new__(ONNXModelManager)
    manager.model_name = "resnet-50"
    manager.model_path = str(tmp_path)
    manager.device = torch.device("cpu")
    manager.inference_backend = "onnx"
    manager.providers = ["CPUExecutionProvider"]
    manager.session_path = (
        onnx_path  # pyright: ignore[reportArgumentType] # type: ignore
    )

    with pytest.raises(KeyError):
        manager.load_model()


def test_predict_raises_value_error_when_missing_session():
    manager = object.__new__(ONNXModelManager)
    manager.model_loaded = True
    manager.session = None

    with pytest.raises(ValueError):
        manager.predict(torch.zeros(1))


def test_init_model_raises_value_error_when_incorrect_tensorrt(fake_ort, tmp_path):
    onnx_path = tmp_path / "resnet-50.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    with pytest.raises(ValueError):
        manager = ONNXModelManager("resnet-50", "cpu", tmp_path, "tensorrt")
        manager.load_model()


def test_load_model_selects_cuda_provider_when_device_is_cuda(
    fake_ort, tmp_path, monkeypatch
):
    monkeypatch.setattr("src.model_manager.torch.cuda.is_available", lambda: True)
    onnx_path = tmp_path / "resnet-50.onnx"
    json_path = tmp_path / "resnet-50_config.json"
    onnx_path.write_bytes(b"fake onnx bytes")
    json_path.write_bytes(
        json.dumps({"id2label": {1: "cat", 2: "dog"}}).encode("utf-8")
    )

    manager = ONNXModelManager(
        model_name="resnet-50", device="cuda", model_path=str(tmp_path)
    )
    manager.load_model()

    assert (
        manager.session.providers  # pyright: ignore[reportArgumentType] # type: ignore
        == ["CUDAExecutionProvider"]
    )


def test_load_model_selects_tensorrt_provider_with_cache_config(
    fake_ort, tmp_path, monkeypatch
):
    monkeypatch.setattr("src.model_manager.torch.cuda.is_available", lambda: True)
    onnx_path = tmp_path / "resnet-50.onnx"
    json_path = tmp_path / "resnet-50_config.json"
    onnx_path.write_bytes(b"fake onnx bytes")
    json_path.write_bytes(
        json.dumps({"id2label": {1: "cat", 2: "dog"}}).encode("utf-8")
    )

    manager = ONNXModelManager(
        model_name="resnet-50",
        device="cuda",
        model_path=str(tmp_path),
        inference_backend="tensorrt",
    )

    assert manager.providers == [
        (
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": ".tensorrtcache/",
            },
        ),
        "CUDAExecutionProvider",
    ]

    manager.load_model()
    assert (
        manager.session.providers  # pyright: ignore[reportArgumentType] # type: ignore
        == manager.providers
    )


def test_predict_returns_logits_tensor_and_latency(fake_ort, tmp_path):
    onnx_path = tmp_path / "resnet-50.onnx"
    json_path = tmp_path / "resnet-50_config.json"
    onnx_path.write_bytes(b"fake onnx bytes")
    json_path.write_bytes(
        json.dumps({"id2label": {1: "cat", 2: "dog"}}).encode("utf-8")
    )
    manager = ONNXModelManager(model_name="resnet-50", model_path=str(tmp_path))
    manager.load_model()

    inputs = torch.zeros(2, 3, 224, 224)
    logits, latency_ms = manager.predict(inputs)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 3)
    assert isinstance(latency_ms, float)
    assert latency_ms >= 0.0


def test_predict_passes_numpy_pixel_values_to_session_run(fake_ort, tmp_path):

    onnx_path = tmp_path / "resnet-50.onnx"
    json_path = tmp_path / "resnet-50_config.json"
    onnx_path.write_bytes(b"fake onnx bytes")
    json_path.write_bytes(
        json.dumps({"id2label": {1: "cat", 2: "dog"}}).encode("utf-8")
    )
    manager = ONNXModelManager(model_name="resnet-50", model_path=str(tmp_path))
    manager.load_model()

    inputs = torch.ones(1, 3, 224, 224)
    manager.predict(inputs)

    output_names, input_feed = (
        manager.session.run_calls[  # pyright: ignore[reportArgumentType] # type: ignore
            0
        ]
    )
    assert isinstance(input_feed["pixel_values"], np.ndarray)
    assert input_feed["pixel_values"].shape == (1, 3, 224, 224)


def test_predict_raises_clear_error_when_model_not_loaded(fake_ort, tmp_path):
    onnx_path = tmp_path / "resnet-50.onnx"
    json_path = tmp_path / "resnet-50_config.json"
    onnx_path.write_bytes(b"fake onnx bytes")
    json_path.write_bytes(
        json.dumps({"id2label": {1: "cat", 2: "dog"}}).encode("utf-8")
    )
    manager = ONNXModelManager(model_name="resnet-50", model_path=tmp_path)
    with pytest.raises(ValueError, match="not loaded"):
        manager.predict(torch.zeros(1, 3, 224, 224))


def test_cleanup_resets_state(fake_ort, tmp_path):
    onnx_path = tmp_path / "resnet-50.onnx"
    json_path = tmp_path / "resnet-50_config.json"
    onnx_path.write_bytes(b"fake onnx bytes")
    json_path.write_bytes(
        json.dumps({"id2label": {1: "cat", 2: "dog"}}).encode("utf-8")
    )
    manager = ONNXModelManager(model_name="resnet-50", model_path=str(tmp_path))
    manager.load_model()

    manager.cleanup_model()

    assert manager.model_loaded is False
    assert manager.session is None


def test_constructor_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("src.model_manager.torch.cuda.is_available", lambda: False)

    manager = TorchModelManager(device="cuda")

    assert manager.device == torch.device("cpu")


def test_constructor_stores_model_path_without_loading():
    manager = TorchModelManager(model_path="/tmp/model.pt", device="cpu")

    assert manager.model_path == "/tmp/model.pt"
    assert manager.model_loaded is False


def test_local_model_loading(monkeypatch):
    class MockProcessorLoader:
        @staticmethod
        def from_pretrained(name, local_files_only):
            return True

    class MockModelLoader:
        @staticmethod
        def from_pretrained(name, local_files_only):
            return SimpleNamespace(
                to=lambda inp: True,
                config=SimpleNamespace(idtolabel={0: "cat", 1: "dog"}),
            )

    monkeypatch.setattr("src.model_manager.AutoImageProcessor", MockProcessorLoader)
    monkeypatch.setattr(
        "src.model_manager.AutoModelForImageClassification", MockModelLoader
    )
    manager = TorchModelManager(model_path=".", device="cpu")
    manager.load_model()
    assert manager.model_loaded is True


def test_load_model_processor_failure_leaves_model_unloaded(monkeypatch):
    class FailingProcessor:
        @staticmethod
        def from_pretrained(name):
            raise RuntimeError("processor failed")

    monkeypatch.setattr("src.model_manager.AutoImageProcessor", FailingProcessor)
    manager = TorchModelManager(device="cpu", model_path=".")

    with pytest.raises(FileNotFoundError):
        manager.load_model()

    assert manager.model_loaded is False


def test_preprocess_inputs_accepts_list_and_preserves_batch_size():
    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()
    manager.image_processor = FakeProcessor()
    images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]

    tensor = manager.preprocess_inputs(images)

    assert tensor.shape[0] == 2
    assert manager.image_processor.called_with == images


def test_preprocess_inputs_rejects_empty_list():
    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()
    manager.image_processor = FakeProcessor()

    with pytest.raises(ValueError):
        manager.preprocess_inputs([])


def test_preprocess_inputs_rejects_missing_processor():
    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()
    manager.image_processor = None

    with pytest.raises(TypeError):
        manager.preprocess_inputs(Image.new("RGB", (2, 2)))


def test_predict_returns_logits_and_positive_latency():
    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()

    logits, latency_ms = manager.predict(torch.zeros(2, 3, 224, 224))

    assert logits.shape == (2, 3)
    assert latency_ms >= 0


def test_predict_raises_when_model_output_has_no_logits():
    class NoLogitsModel:
        def eval(self):
            pass

        def __call__(self, inputs):
            return SimpleNamespace(scores=torch.zeros(1, 3))

    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = NoLogitsModel()

    with pytest.raises(AttributeError):
        manager.predict(torch.zeros(1, 3, 224, 224))


def test_top_k_returns_descending_labels_and_scores():
    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel(id2label={0: "zero", 1: "one", 2: "two"})
    manager.id2label = {0: "zero", 1: "one", 2: "two"}

    result = manager.top_k_from_logits(torch.tensor([[0.2, 3.0, 1.0]]), k=3)

    assert [label for label, _ in result[0]] == ["one", "two", "zero"]
    assert [score for _, score in result[0]] == [3.0, 1.0, 0.20000000298023224]


def test_top_k_rejects_bool_k():
    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()

    with pytest.raises(TypeError):
        manager.top_k_from_logits(
            torch.zeros(1, 3), k=True
        )  # pyright: ignore[reportArgumentType]


def test_top_k_rejects_k_larger_than_class_count():
    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()

    with pytest.raises(ValueError):
        manager.top_k_from_logits(torch.zeros(1, 3), k=4)


def test_top_k_falls_back_for_missing_label_index():
    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel(id2label={0: "zero"})
    manager.id2label = {0: "zero"}

    result = manager.top_k_from_logits(torch.tensor([[0.0, 10.0]]), k=1)

    assert result[0][0][0] == "1"


def test_cleanup_model_is_idempotent_when_not_loaded(monkeypatch):
    manager = TorchModelManager(device="cpu")

    manager.cleanup_model()
    manager.cleanup_model()

    assert manager.model_loaded is False


def test_cleanup_model_deletes_loaded_state(monkeypatch):
    collected = []
    monkeypatch.setattr("src.model_manager.gc.collect", lambda: collected.append(True))
    monkeypatch.setattr("src.model_manager.torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("src.model_manager.torch.cuda.empty_cache", lambda: None)
    manager = TorchModelManager(device="cuda")
    manager.model_loaded = True
    manager.model = FakeModel()
    manager.image_processor = FakeProcessor()

    # Force CUDA device here for coverage in CPU-only systems
    manager.device = torch.device("cuda")
    manager.cleanup_model()

    assert manager.model_loaded is False
    assert collected == [True]
    assert manager.model is None
    assert manager.image_processor is None


def test_init_validation():
    # invalid model_name
    with pytest.raises(ValueError):
        TorchModelManager(model_name=None)
    with pytest.raises(ValueError):
        TorchModelManager(model_name="not-a-model")
    with pytest.raises(ValueError):
        TorchModelManager(device="not-a-device")


def test_load_model_monkeypatched(monkeypatch):
    manager = TorchModelManager()

    # monkeypatch AutoImageProcessor and AutoModelForImageClassification
    monkeypatch.setattr(
        "src.model_manager.AutoImageProcessor",
        types.SimpleNamespace(from_pretrained=lambda name: FakeProcessor()),
    )
    monkeypatch.setattr(
        "src.model_manager.AutoModelForImageClassification",
        types.SimpleNamespace(from_pretrained=lambda name: FakeModel()),
    )

    # Should not raise
    manager.load_model()
    assert isinstance(manager.image_processor, FakeProcessor)
    assert isinstance(manager.model, FakeModel)
    assert manager.model.moved_to == manager.device


def test_top_k_from_logits(monkeypatch):
    manager = TorchModelManager()
    # provide a fake loaded model with id2label
    manager.model = FakeModel(id2label={0: "a", 1: "b", 2: "c"})
    manager.id2label = {0: "a", 1: "b", 2: "c"}
    manager.model_loaded = True

    logits = torch.tensor([[0.1, 2.0, 0.5], [1.0, 0.2, 0.3]])
    top2 = manager.top_k_from_logits(logits, k=2)
    assert isinstance(top2, list)
    assert len(top2) == 2
    # check label names and float scores
    for row in top2:
        for label, score in row:
            assert isinstance(label, str)
            assert isinstance(score, float)


def test_top_k_invalid_k_values():
    manager = TorchModelManager()
    manager.model = FakeModel()
    manager.model_loaded = True
    logits = torch.zeros(2, 3)

    with pytest.raises(TypeError):
        manager.top_k_from_logits(logits, k=0)
    with pytest.raises(TypeError):
        manager.top_k_from_logits(logits, k=-1)
    with pytest.raises(TypeError):
        manager.top_k_from_logits(
            logits, k="two"  # pyright: ignore [reportArgumentType] # type: ignore
        )


def test_top_k_missing_id2label_raises():
    """id2label defaults to None until load_model() populates it."""
    manager = TorchModelManager()
    manager.model_loaded = True
    with pytest.raises(AttributeError):
        manager.top_k_from_logits(torch.zeros(1, 3), k=1)


def test_methods_raise_if_not_loaded():
    manager = TorchModelManager()
    # ensure model unset
    manager.model = None
    img = Image.new("RGB", (5, 5))
    with pytest.raises(ValueError):
        manager.preprocess_inputs(img)
    with pytest.raises(ValueError):
        manager.predict(torch.zeros(1, 3, 224, 224))
    with pytest.raises(ValueError):
        manager.top_k_from_logits(torch.zeros(1, 3), k=1)


def test_preprocess_inputs_invalid_type():
    manager = TorchModelManager()
    manager.model = FakeModel()
    manager.model_loaded = True
    manager.image_processor = FakeProcessor()

    # None input
    with pytest.raises(TypeError):
        manager.preprocess_inputs(None)

    # wrong datatype (int)
    with pytest.raises(TypeError):
        manager.preprocess_inputs(
            123  # pyright: ignore [reportArgumentType] # type: ignore
        )


@pytest.mark.parametrize(
    "bad_input, expected_exception",
    [(None, TypeError), ("not-a-tensor", AttributeError)],
)
def test_predict_rejects_invalid_inputs(bad_input, expected_exception):
    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()

    with pytest.raises(expected_exception):
        manager.predict(bad_input)


def test_top_k_from_logits_invalid():
    manager = TorchModelManager()
    manager.model = FakeModel()
    manager.model_loaded = True

    # non-tensor input
    with pytest.raises(TypeError):
        manager.top_k_from_logits(
            [1, 2, 3], k=1  # pyright: ignore [reportArgumentType] # type: ignore
        )

    # wrong-dim tensor
    with pytest.raises(ValueError):
        manager.top_k_from_logits(torch.zeros(2, 3, 4), k=1)


def test_get_model_info():
    manager = TorchModelManager(
        model_name="microsoft/resnet-50", device="cpu", model_path=None
    )
    info = manager.get_model_info()
    assert isinstance(info, dict)
    assert set(info.keys()) == {"name", "device", "model_path"}
    assert info["name"] == "microsoft/resnet-50"


def test_cleanup_backend_fallback_works():
    class MockModelManager(ModelManager):
        def load_model(self):
            pass

        def predict(self, inputs):
            return inputs

    manager = MockModelManager()
    manager._cleanup_backend()


def test_missing_id2label_fails_load(monkeypatch):
    manager = object.__new__(TorchModelManager)
    manager.model_name = "microsoft/resnet-50"
    manager.device = torch.device("cpu")
    manager.model_path = None

    class MockProcessorLoader:
        @staticmethod
        def from_pretrained(name):
            return True

    class MockModelLoader:
        @staticmethod
        def from_pretrained(name):
            return SimpleNamespace(
                to=lambda inp: inp,
            )

    monkeypatch.setattr("src.model_manager.AutoImageProcessor", MockProcessorLoader)
    monkeypatch.setattr(
        "src.model_manager.AutoModelForImageClassification", MockModelLoader
    )
    with pytest.raises(AttributeError):
        manager.load_model()
