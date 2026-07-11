import types
from types import SimpleNamespace
import pytest
import torch
from PIL import Image

from model_loader import ModelLoader


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

    def __call__(self, **inputs):
        batch_size = inputs["pixel_values"].shape[0]
        # return logits for three classes
        logits = torch.tensor([[0.1, 2.0, 0.5]] * batch_size)
        return SimpleNamespace(logits=logits)


def test_init_validation():
    # invalid model_name
    with pytest.raises(ValueError):
        ModelLoader(model_name=None)
    with pytest.raises(ValueError):
        ModelLoader(model_name="not-a-model")
    with pytest.raises(ValueError):
        ModelLoader(device="not-a-device")


def test_load_model_monkeypatched(monkeypatch):
    loader = ModelLoader()

    # monkeypatch AutoImageProcessor and AutoModelForImageClassification
    monkeypatch.setattr(
        "model_loader.AutoImageProcessor",
        types.SimpleNamespace(from_pretrained=lambda name: FakeProcessor()),
    )
    monkeypatch.setattr(
        "model_loader.AutoModelForImageClassification",
        types.SimpleNamespace(from_pretrained=lambda name: FakeModel()),
    )

    # Should not raise
    loader.load_model()
    assert isinstance(loader.image_processor, FakeProcessor)
    assert isinstance(loader.model, FakeModel)
    assert loader.model.moved_to == loader.device


def test_preprocess_and_predict(monkeypatch):
    loader = ModelLoader()
    # set a fake model and fake processor directly (skip load_model)
    loader.model = FakeModel()
    loader.image_processor = FakeProcessor()

    img = Image.new("RGB", (10, 10), color="red")
    inputs = loader.preprocess_inputs(img)
    assert "pixel_values" in inputs
    assert inputs["pixel_values"].device.type == loader.device.type

    logits = loader.predict(inputs)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[1] == 3


def test_top_k_from_logits(monkeypatch):
    loader = ModelLoader()
    # provide a fake loaded model with id2label
    loader.model = FakeModel(id2label={0: "a", 1: "b", 2: "c"})

    logits = torch.tensor([[0.1, 2.0, 0.5], [1.0, 0.2, 0.3]])
    top2 = loader.top_k_from_logits(logits, k=2)
    assert isinstance(top2, list)
    assert len(top2) == 2
    # check label names and float scores
    for row in top2:
        for label, score in row:
            assert isinstance(label, str)
            assert isinstance(score, float)


def test_preprocess_inputs_sequence_with_non_image():
    loader = ModelLoader()
    loader.model = FakeModel()
    loader.image_processor = FakeProcessor()

    # sequence containing a non-image element should raise
    with pytest.raises(TypeError):
        loader.preprocess_inputs(
            [Image.new("RGB", (2, 2)), 123]  # pyright: ignore [reportArgumentType]
        )


def test_top_k_invalid_k_values():
    loader = ModelLoader()
    loader.model = FakeModel()
    logits = torch.zeros(2, 3)

    with pytest.raises(TypeError):
        loader.top_k_from_logits(logits, k=0)
    with pytest.raises(TypeError):
        loader.top_k_from_logits(logits, k=-1)
    with pytest.raises(TypeError):
        loader.top_k_from_logits(
            logits, k="two"  # pyright: ignore [reportArgumentType]
        )


def test_top_k_missing_id2label_raises():
    loader = ModelLoader()
    # model.config exists but has no id2label attribute
    loader.model = SimpleNamespace(config=SimpleNamespace())
    with pytest.raises(AttributeError):
        loader.top_k_from_logits(torch.zeros(1, 3), k=1)


def test_top_k_id2label_index_fallback():
    loader = ModelLoader()
    # provide id2label as a sequence too small to cover indices -> triggers except
    loader.model = FakeModel(id2label=["only"])
    logits = torch.tensor([[0.1, 2.0, 0.5]])
    res = loader.top_k_from_logits(logits, k=3)
    # index 0 maps to the provided item, out-of-range indices fallback
    assert [lbl for lbl, _ in res[0]] == ["1", "2", "only"]


def test_methods_raise_if_not_loaded():
    loader = ModelLoader()
    # ensure model unset
    loader.model = None
    img = Image.new("RGB", (5, 5))
    with pytest.raises(ValueError):
        loader.preprocess_inputs(img)
    with pytest.raises(ValueError):
        loader.predict({"pixel_values": torch.zeros(1, 3, 224, 224)})
    with pytest.raises(ValueError):
        loader.top_k_from_logits(torch.zeros(1, 3), k=1)


def test_load_model_file_not_found(monkeypatch):
    """If HF loading fails and no local model_path is provided, load_model should raise FileNotFoundError."""
    loader = ModelLoader()
    # return a valid processor but make the model loader raise
    monkeypatch.setattr(
        "model_loader.AutoImageProcessor",
        types.SimpleNamespace(from_pretrained=lambda name: FakeProcessor()),
    )

    class Fail:
        @staticmethod
        def from_pretrained(name):
            raise RuntimeError("simulated HF failure")

    monkeypatch.setattr("model_loader.AutoModelForImageClassification", Fail)

    with pytest.raises(FileNotFoundError):
        loader.load_model()


def test_preprocess_inputs_invalid_type():
    loader = ModelLoader()
    loader.model = FakeModel()
    loader.image_processor = FakeProcessor()

    # None input
    with pytest.raises(TypeError):
        loader.preprocess_inputs(None)

    # wrong datatype (int)
    with pytest.raises(TypeError):
        loader.preprocess_inputs(123)  # pyright: ignore [reportArgumentType]


def test_predict_invalid_inputs():
    loader = ModelLoader()
    loader.model = FakeModel()

    # None -> TypeError (explicit check)
    with pytest.raises(TypeError):
        loader.predict(None)  # pyright: ignore [reportArgumentType]

    # Missing expected key will raise KeyError inside FakeModel
    with pytest.raises(KeyError):
        loader.predict({"wrong_key": torch.zeros(1, 3, 224, 224)})


def test_top_k_from_logits_invalid():
    loader = ModelLoader()
    loader.model = FakeModel()

    # non-tensor input
    with pytest.raises(TypeError):
        loader.top_k_from_logits([1, 2, 3], k=1)  # pyright: ignore [reportArgumentType]

    # wrong-dim tensor
    with pytest.raises(ValueError):
        loader.top_k_from_logits(torch.zeros(2, 3, 4), k=1)


def test_get_model_info():
    loader = ModelLoader(
        model_name="microsoft/resnet-50", device="cpu", model_path=None
    )
    info = loader.get_model_info()
    assert isinstance(info, dict)
    assert set(info.keys()) == {"name", "device", "model_path"}
    assert info["name"] == "microsoft/resnet-50"
