import tempfile
import types
from types import SimpleNamespace
import pytest
import torch
from PIL import Image

from model_manager import ModelManager


class FakeProcessor:
    def __init__(self):
        self.called_with = None

    def __call__(self, images, return_tensors="pt"):
        # return a fake pixel_values tensor shaped (batch,3,224,224)
        batch = len(images)
        self.called_with = list(images)
        return {"pixel_values": torch.zeros(batch, 3, 224, 224)}


class FakeModel:
    def __init__(self, id2label=None):
        self.config = SimpleNamespace(
            id2label=id2label or {0: "zero", 1: "one", 2: "two"}
        )
        self.moved_to = None

    def to(self, device):
        # record device move
        self.moved_to = device
        return self

    def eval(self):
        pass

    def __call__(self, inputs):
        batch_size = inputs.shape[0]
        # return logits for three classes
        logits = torch.tensor([[0.1, 2.0, 0.5]] * batch_size)
        return SimpleNamespace(logits=logits)


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
        "model_manager.AutoImageProcessor",
        types.SimpleNamespace(from_pretrained=lambda name: FakeProcessor()),
    )
    monkeypatch.setattr(
        "model_manager.AutoModelForImageClassification",
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
            logits, k="two"  # pyright: ignore [reportArgumentType]
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
        "model_manager.AutoImageProcessor",
        types.SimpleNamespace(from_pretrained=lambda name: FakeProcessor()),
    )

    class Fail:
        @staticmethod
        def from_pretrained(name):
            raise RuntimeError("simulated HF failure")

    monkeypatch.setattr("model_manager.AutoModelForImageClassification", Fail)

    with pytest.raises(FileNotFoundError):
        manager.load_model()


def test_load_model_falls_back_to_local_weights_when_hf_load_fails(monkeypatch):
    manager = ModelManager(model_name="microsoft/resnet-50", device="cpu")
    monkeypatch.setattr(
        "model_manager.AutoImageProcessor",
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

    monkeypatch.setattr("model_manager.AutoModelForImageClassification", FailThenLoad)

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
        manager.preprocess_inputs(123)  # pyright: ignore [reportArgumentType]


def test_predict_invalid_inputs():
    manager = ModelManager()
    manager.model = FakeModel()
    manager.model_loaded = True

    # None -> TypeError (explicit check)
    with pytest.raises(TypeError):
        manager.predict(None)  # pyright: ignore [reportArgumentType]


def test_top_k_from_logits_invalid():
    manager = ModelManager()
    manager.model = FakeModel()
    manager.model_loaded = True

    # non-tensor input
    with pytest.raises(TypeError):
        manager.top_k_from_logits(
            [1, 2, 3], k=1  # pyright: ignore [reportArgumentType]
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
