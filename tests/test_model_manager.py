import types
from types import SimpleNamespace
import pytest
import torch
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


def test_load_model_creates_session_with_configured_providers(fake_ort, tmp_path):
    onnx_path = tmp_path / "resnet-50.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")

    manager = ONNXModelManager(
        model_name="resnet-50",
        device="cuda",
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
        == ["CUDAExecutionProvider"]
    )


def test_load_model_defaults_to_cpu_provider_when_unset(fake_ort, tmp_path):
    onnx_path = tmp_path / "resnet-50.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")

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
    manager.providers = ["CPUExecutionProvider"]
    manager.session_path = None  # pyright: ignore[reportArgumentType] # type: ignore

    with pytest.raises(ValueError):
        manager.load_model()


def test_predict_returns_logits_tensor_and_latency(fake_ort, tmp_path):
    onnx_path = tmp_path / "resnet-50.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
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
    onnx_path.write_bytes(b"fake onnx bytes")
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
    onnx_path.write_bytes(b"fake onnx bytes")
    manager = ONNXModelManager(model_name="resnet-50", model_path=tmp_path)
    with pytest.raises(ValueError, match="not loaded"):
        manager.predict(torch.zeros(1, 3, 224, 224))


def test_cleanup_resets_state(fake_ort, tmp_path):
    onnx_path = tmp_path / "resnet-50.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    manager = ONNXModelManager(model_name="resnet-50", model_path=str(tmp_path))
    manager.load_model()

    manager.cleanup_model()

    assert manager.model_loaded is False


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
            return SimpleNamespace(to=lambda inp: True)

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


def test_load_model_moves_model_to_device_and_marks_loaded(monkeypatch):
    monkeypatch.setattr(
        "src.model_manager.AutoImageProcessor",
        types.SimpleNamespace(from_pretrained=lambda name: FakeProcessor()),
    )
    monkeypatch.setattr(
        "src.model_manager.AutoModelForImageClassification",
        types.SimpleNamespace(from_pretrained=lambda name: FakeModel()),
    )
    manager = TorchModelManager(device="cpu")

    manager.load_model()

    assert manager.model_loaded is True
    assert manager.model.moved_to == torch.device("cpu")


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


def test_predict_rejects_non_tensor_input():
    manager = TorchModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()

    with pytest.raises(AttributeError):
        manager.predict("not-a-tensor")  # pyright: ignore[reportArgumentType]


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
    manager = TorchModelManager(device="cuda")
    manager.model_loaded = True
    manager.model = FakeModel()
    manager.image_processor = FakeProcessor()

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


def test_preprocess_and_predict(monkeypatch):
    manager = TorchModelManager()
    # set a fake model and fake processor directly (skip load_model)
    manager.model = FakeModel()
    manager.model_loaded = True
    manager.image_processor = FakeProcessor()

    img = Image.new("RGB", (10, 10), color="red")
    inputs = manager.preprocess_inputs(img)
    for inp in inputs:
        assert inp.device.type == manager.device.type

    logits, inference_time_ms = manager.predict(inputs)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[1] == 3
    assert isinstance(inference_time_ms, float)


def test_top_k_from_logits(monkeypatch):
    manager = TorchModelManager()
    # provide a fake loaded model with id2label
    manager.model = FakeModel(id2label={0: "a", 1: "b", 2: "c"})
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


def test_preprocess_inputs_sequence_with_non_image():
    manager = TorchModelManager()
    manager.model = FakeModel()
    manager.model_loaded = True
    manager.image_processor = FakeProcessor()

    # sequence containing a non-image element should raise
    with pytest.raises(TypeError):
        manager.preprocess_inputs(None)


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
    manager = TorchModelManager()
    # model.config exists but has no id2label attribute
    manager.model = SimpleNamespace(config=SimpleNamespace())
    manager.model_loaded = True
    with pytest.raises(AttributeError):
        manager.top_k_from_logits(torch.zeros(1, 3), k=1)


def test_top_k_id2label_index_fallback():
    manager = TorchModelManager()
    # provide id2label as a sequence too small to cover indices -> triggers except
    manager.model = FakeModel(id2label=["only"])
    manager.model_loaded = True
    logits = torch.tensor([[0.1, 2.0, 0.5]])
    res = manager.top_k_from_logits(logits, k=3)
    # index 0 maps to the provided item, out-of-range indices fallback
    assert [lbl for lbl, _ in res[0]] == ["1", "2", "only"]


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


def test_predict_invalid_inputs():
    manager = TorchModelManager()
    manager.model = FakeModel()
    manager.model_loaded = True

    # None -> TypeError (explicit check)
    with pytest.raises(TypeError):
        manager.predict(None)  # pyright: ignore [reportArgumentType] # type: ignore


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
