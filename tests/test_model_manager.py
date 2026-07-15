import tempfile
import types
from types import SimpleNamespace
import pytest
import torch
from PIL import Image

from src.model_manager import ModelManager


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


def test_constructor_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("src.model_manager.torch.cuda.is_available", lambda: False)

    manager = ModelManager(device="cuda")

    assert manager.device == torch.device("cpu")


def test_constructor_stores_model_path_without_loading():
    manager = ModelManager(model_path="/tmp/model.pt", device="cpu")

    assert manager.model_path == "/tmp/model.pt"
    assert manager.model_loaded is False


def test_load_model_processor_failure_leaves_model_unloaded(monkeypatch):
    class FailingProcessor:
        @staticmethod
        def from_pretrained(name):
            raise RuntimeError("processor failed")

    monkeypatch.setattr("src.model_manager.AutoImageProcessor", FailingProcessor)
    manager = ModelManager(device="cpu")

    with pytest.raises(RuntimeError):
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
    manager = ModelManager(device="cpu")

    manager.load_model()

    assert manager.model_loaded is True
    assert manager.model.moved_to == torch.device("cpu")


def test_preprocess_inputs_accepts_list_and_preserves_batch_size():
    manager = ModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()
    manager.image_processor = FakeProcessor()
    images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]

    tensor = manager.preprocess_inputs(images)

    assert tensor.shape[0] == 2
    assert manager.image_processor.called_with == images


def test_preprocess_inputs_rejects_empty_list():
    manager = ModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()
    manager.image_processor = FakeProcessor()

    with pytest.raises(ValueError):
        manager.preprocess_inputs([])


def test_preprocess_inputs_rejects_missing_processor():
    manager = ModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()
    manager.image_processor = None

    with pytest.raises(TypeError):
        manager.preprocess_inputs(Image.new("RGB", (2, 2)))


def test_predict_rejects_non_tensor_input():
    manager = ModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()

    with pytest.raises(AttributeError):
        manager.predict("not-a-tensor")  # pyright: ignore[reportArgumentType]


def test_predict_returns_logits_and_positive_latency():
    manager = ModelManager(device="cpu")
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

    manager = ModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = NoLogitsModel()

    with pytest.raises(AttributeError):
        manager.predict(torch.zeros(1, 3, 224, 224))


def test_top_k_returns_descending_labels_and_scores():
    manager = ModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel(id2label={0: "zero", 1: "one", 2: "two"})

    result = manager.top_k_from_logits(torch.tensor([[0.2, 3.0, 1.0]]), k=3)

    assert [label for label, _ in result[0]] == ["one", "two", "zero"]
    assert [score for _, score in result[0]] == [3.0, 1.0, 0.20000000298023224]


def test_top_k_rejects_bool_k():
    manager = ModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()

    with pytest.raises(TypeError):
        manager.top_k_from_logits(
            torch.zeros(1, 3), k=True
        )  # pyright: ignore[reportArgumentType]


def test_top_k_rejects_k_larger_than_class_count():
    manager = ModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel()

    with pytest.raises(ValueError):
        manager.top_k_from_logits(torch.zeros(1, 3), k=4)


def test_top_k_falls_back_for_missing_label_index():
    manager = ModelManager(device="cpu")
    manager.model_loaded = True
    manager.model = FakeModel(id2label={0: "zero"})

    result = manager.top_k_from_logits(torch.tensor([[0.0, 10.0]]), k=1)

    assert result[0][0][0] == "1"


def test_cleanup_model_is_idempotent_when_not_loaded():
    manager = ModelManager(device="cpu")

    manager.cleanup_model()
    manager.cleanup_model()

    assert manager.model_loaded is False


def test_cleanup_model_deletes_loaded_state(monkeypatch):
    collected = []
    monkeypatch.setattr("src.model_manager.gc.collect", lambda: collected.append(True))
    manager = ModelManager(device="cpu")
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
        ModelManager(model_name=None)
    with pytest.raises(ValueError):
        ModelManager(model_name="not-a-model")
    with pytest.raises(ValueError):
        ModelManager(device="not-a-device")


def test_load_model_monkeypatched(monkeypatch):
    manager = ModelManager()

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
    manager = ModelManager()
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
    manager = ModelManager()
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
    manager = ModelManager()
    manager.model = FakeModel()
    manager.model_loaded = True
    manager.image_processor = FakeProcessor()

    # sequence containing a non-image element should raise
    with pytest.raises(TypeError):
        manager.preprocess_inputs(None)


def test_top_k_invalid_k_values():
    manager = ModelManager()
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
    manager = ModelManager()
    # model.config exists but has no id2label attribute
    manager.model = SimpleNamespace(config=SimpleNamespace())
    manager.model_loaded = True
    with pytest.raises(AttributeError):
        manager.top_k_from_logits(torch.zeros(1, 3), k=1)


def test_top_k_id2label_index_fallback():
    manager = ModelManager()
    # provide id2label as a sequence too small to cover indices -> triggers except
    manager.model = FakeModel(id2label=["only"])
    manager.model_loaded = True
    logits = torch.tensor([[0.1, 2.0, 0.5]])
    res = manager.top_k_from_logits(logits, k=3)
    # index 0 maps to the provided item, out-of-range indices fallback
    assert [lbl for lbl, _ in res[0]] == ["1", "2", "only"]


def test_methods_raise_if_not_loaded():
    manager = ModelManager()
    # ensure model unset
    manager.model = None
    img = Image.new("RGB", (5, 5))
    with pytest.raises(ValueError):
        manager.preprocess_inputs(img)
    with pytest.raises(ValueError):
        manager.predict(torch.zeros(1, 3, 224, 224))
    with pytest.raises(ValueError):
        manager.top_k_from_logits(torch.zeros(1, 3), k=1)


def test_load_model_file_not_found(monkeypatch):
    """If HF loading fails and no local model_path is provided, load_model should raise FileNotFoundError."""
    manager = ModelManager()
    # return a valid processor but make the model manager raise
    monkeypatch.setattr(
        "src.model_manager.AutoImageProcessor",
        types.SimpleNamespace(from_pretrained=lambda name: FakeProcessor()),
    )

    class Fail:
        @staticmethod
        def from_pretrained(name):
            raise RuntimeError("simulated HF failure")

    monkeypatch.setattr("src.model_manager.AutoModelForImageClassification", Fail)

    with pytest.raises(FileNotFoundError):
        manager.load_model()


def test_load_model_falls_back_to_local_weights_when_hf_load_fails(monkeypatch):
    manager = ModelManager(model_name="microsoft/resnet-50", device="cpu")
    monkeypatch.setattr(
        "src.model_manager.AutoImageProcessor",
        types.SimpleNamespace(from_pretrained=lambda name: FakeProcessor()),
    )

    class LocalModel(FakeModel):
        def load_state_dict(self, state):
            self.loaded_state = state

    class FailThenLoad:
        calls = 0

        @classmethod
        def from_pretrained(cls, name):
            cls.calls += 1
            if cls.calls == 1:
                raise RuntimeError("simulated HF failure")
            return LocalModel()

    monkeypatch.setattr(
        "src.model_manager.AutoModelForImageClassification", FailThenLoad
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save({"state": torch.tensor([1.0])}, tmp.name)
        manager.model_path = tmp.name

    manager.load_model()

    assert manager.model_loaded is True
    assert manager.model.loaded_state["state"].item() == 1.0


def test_preprocess_inputs_invalid_type():
    manager = ModelManager()
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
    manager = ModelManager()
    manager.model = FakeModel()
    manager.model_loaded = True

    # None -> TypeError (explicit check)
    with pytest.raises(TypeError):
        manager.predict(None)  # pyright: ignore [reportArgumentType] # type: ignore


def test_top_k_from_logits_invalid():
    manager = ModelManager()
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
    manager = ModelManager(
        model_name="microsoft/resnet-50", device="cpu", model_path=None
    )
    info = manager.get_model_info()
    assert isinstance(info, dict)
    assert set(info.keys()) == {"name", "device", "model_path"}
    assert info["name"] == "microsoft/resnet-50"
