"""Tests for scripts/export_onnx.py.

All Hugging Face / torch.onnx.export / onnxruntime calls are mocked — these
tests verify the script's own orchestration and file-naming logic, not real
model export (which needs network access, real weights, and real tracing).
"""

import json
import types
from types import SimpleNamespace

import pytest
import torch

from scripts.export_onnx import export, main, resolve_output_paths, verify_parity
from src.config import Config


def make_config(**overrides) -> Config:
    defaults = dict(
        MODEL_NAME="microsoft/resnet-50",
        MODEL_PATH="onnx_models",
        MAX_BATCH_SIZE=64,
    )
    return SimpleNamespace(
        **{**defaults, **overrides}
    )  # pyright: ignore[reportArgumentType] # type: ignore


class FakeImageProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(1, 3, 224, 224)}

    @classmethod
    def from_pretrained(cls, name):
        return cls()


_DEFAULT_ID2LABEL = {0: "cat", 1: "dog"}


def make_fake_torch_model_cls(id2label=_DEFAULT_ID2LABEL, logits=None):
    """Builds a fake AutoModelForImageClassification.from_pretrained() replacement.

    id2label defaults to a real dict. Pass id2label=None or id2label={} explicitly
    to simulate a model with no usable id2label — unlike a plain `id2label=None`
    default parameter, this doesn't collapse "not specified" and "explicitly None"
    into the same case.
    """
    resolved_id2label = id2label
    resolved_logits = logits if logits is not None else torch.tensor([[0.1, 5.0]])

    class FakeTorchModel:
        def __init__(self):
            self.config = SimpleNamespace(id2label=resolved_id2label)

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(logits=resolved_logits)

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    return FakeTorchModel


# ---------------------------------------------------------------------------
# resolve_output_paths
# ---------------------------------------------------------------------------


def test_resolve_output_paths_sanitizes_slash(tmp_path):
    config = make_config(MODEL_NAME="microsoft/resnet-50", MODEL_PATH=str(tmp_path))

    name, onnx_path, json_path = resolve_output_paths(config)

    assert name == "microsoft-resnet-50"
    assert onnx_path == tmp_path / "microsoft-resnet-50.onnx"
    assert json_path == tmp_path / "microsoft-resnet-50_config.json"


def test_resolve_output_paths_leaves_slash_free_name_unchanged(tmp_path):
    config = make_config(MODEL_NAME="resnet-50", MODEL_PATH=str(tmp_path))

    name, onnx_path, json_path = resolve_output_paths(config)

    assert name == "resnet-50"
    assert onnx_path == tmp_path / "resnet-50.onnx"


def test_resolve_output_paths_creates_output_directory(tmp_path):
    output_dir = tmp_path / "does" / "not" / "exist"
    config = make_config(MODEL_PATH=str(output_dir))

    resolve_output_paths(config)

    assert output_dir.is_dir()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_writes_onnx_and_config_files(monkeypatch, tmp_path):
    config = make_config(MODEL_PATH=str(tmp_path), MAX_BATCH_SIZE=8)
    monkeypatch.setattr(
        "scripts.export_onnx.AutoModelForImageClassification",
        make_fake_torch_model_cls(id2label={0: "cat", 1: "dog"}),
    )

    export_calls = []

    def fake_onnx_export(**kwargs):
        export_calls.append(kwargs)
        # torch.onnx.export writes the file itself; simulate that side effect
        # since nothing else in export() creates the .onnx file.
        kwargs["f"].write_bytes(b"fake onnx bytes")

    monkeypatch.setattr("scripts.export_onnx.torch.onnx.export", fake_onnx_export)

    export(config)

    assert len(export_calls) == 1
    call = export_calls[0]
    assert call["f"] == tmp_path / "microsoft-resnet-50.onnx"
    assert call["input_names"] == ["pixel_values"]
    assert call["output_names"] == ["logits"]
    # Structure only here — the actual Dim(min=..., max=...) values are
    # verified precisely in test_export_uses_configured_max_batch_size_for_dynamic_dim.
    assert set(call["dynamic_shapes"].keys()) == {"pixel_values"}
    assert set(call["dynamic_shapes"]["pixel_values"].keys()) == {0}

    json_path = tmp_path / "microsoft-resnet-50_config.json"
    assert json_path.exists()
    with open(json_path) as f:
        written = json.load(f)
    assert written == {"id2label": {"0": "cat", "1": "dog"}}


def test_export_uses_configured_max_batch_size_for_dynamic_dim(monkeypatch, tmp_path):
    config = make_config(MODEL_PATH=str(tmp_path), MAX_BATCH_SIZE=16)
    monkeypatch.setattr(
        "scripts.export_onnx.AutoModelForImageClassification",
        make_fake_torch_model_cls(),
    )

    captured_dim = {}

    def fake_dim(name, min, max):
        captured_dim["min"] = min
        captured_dim["max"] = max
        return SimpleNamespace(name=name, min=min, max=max)

    monkeypatch.setattr("scripts.export_onnx.torch.export.Dim", fake_dim)
    monkeypatch.setattr(
        "scripts.export_onnx.torch.onnx.export",
        lambda **kwargs: kwargs["f"].write_bytes(b"fake onnx bytes"),
    )

    export(config)

    assert captured_dim == {"min": 1, "max": 16}


def test_export_raises_before_exporting_when_id2label_missing(monkeypatch, tmp_path):
    """A model with no id2label should fail loudly at export time, not produce
    an artifact that crashes ONNXModelManager later at serving time."""
    config = make_config(MODEL_PATH=str(tmp_path))
    monkeypatch.setattr(
        "scripts.export_onnx.AutoModelForImageClassification",
        make_fake_torch_model_cls(id2label=None),
    )

    export_calls = []
    monkeypatch.setattr(
        "scripts.export_onnx.torch.onnx.export",
        lambda **kwargs: export_calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="id2label"):
        export(config)

    assert export_calls == []  # must fail before attempting the (expensive) export
    assert not (tmp_path / "microsoft-resnet-50.onnx").exists()
    assert not (tmp_path / "microsoft-resnet-50_config.json").exists()


def test_export_raises_when_id2label_is_empty_dict(monkeypatch, tmp_path):
    """Empty dict is falsy but not None — should be treated the same as missing."""
    config = make_config(MODEL_PATH=str(tmp_path))
    monkeypatch.setattr(
        "scripts.export_onnx.AutoModelForImageClassification",
        make_fake_torch_model_cls(id2label={}),
    )

    with pytest.raises(ValueError, match="id2label"):
        export(config)


# ---------------------------------------------------------------------------
# verify_parity
# ---------------------------------------------------------------------------


class FakeInferenceSession:
    def __init__(self, path, logits=None, providers=None):
        self.path = path
        self.providers = providers
        self._logits = logits if logits is not None else [[0.1, 5.0]]

    def run(self, output_names, input_feed):
        import numpy as np

        return [np.array(self._logits, dtype="float32")]


def _patch_common(monkeypatch, torch_logits, onnx_logits):
    monkeypatch.setattr("scripts.export_onnx.AutoImageProcessor", FakeImageProcessor)
    monkeypatch.setattr(
        "scripts.export_onnx.AutoModelForImageClassification",
        make_fake_torch_model_cls(logits=torch_logits),
    )
    monkeypatch.setattr(
        "scripts.export_onnx.onnxruntime",
        types.SimpleNamespace(
            InferenceSession=lambda path, providers=None: FakeInferenceSession(
                path, logits=onnx_logits, providers=providers
            )
        ),
    )


def test_verify_parity_passes_when_top1_agrees(monkeypatch, tmp_path, capsys):
    onnx_path = tmp_path / "microsoft-resnet-50.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    config = make_config(MODEL_PATH=str(tmp_path))

    _patch_common(
        monkeypatch,
        torch_logits=torch.tensor([[0.1, 5.0]]),  # top-1 = index 1
        onnx_logits=[[0.05, 4.9]],  # top-1 = index 1, slightly different values
    )

    verify_parity(config)  # should not raise / not exit

    assert "Parity OK" in capsys.readouterr().out


def test_verify_parity_exits_nonzero_when_top1_disagrees(monkeypatch, tmp_path):
    onnx_path = tmp_path / "microsoft-resnet-50.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    config = make_config(MODEL_PATH=str(tmp_path))

    _patch_common(
        monkeypatch,
        torch_logits=torch.tensor([[9.0, 0.1]]),  # top-1 = index 0
        onnx_logits=[[0.1, 9.0]],  # top-1 = index 1 -- mismatch
    )

    with pytest.raises(SystemExit) as exc_info:
        verify_parity(config)

    assert exc_info.value.code == 1


def test_verify_parity_warns_but_does_not_exit_when_logits_drift_within_top1_agreement(
    monkeypatch, tmp_path, capsys
):
    onnx_path = tmp_path / "microsoft-resnet-50.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    config = make_config(MODEL_PATH=str(tmp_path))

    _patch_common(
        monkeypatch,
        torch_logits=torch.tensor([[0.1, 5.0]]),
        onnx_logits=[
            [0.1, 5.5]
        ],  # top-1 still agrees, but drifts beyond default atol=1e-3
    )

    verify_parity(config)  # should not exit

    output = capsys.readouterr().out
    assert "PARITY WARNING" in output
    assert "Parity OK" in output


def test_verify_parity_loads_session_from_correct_onnx_path(monkeypatch, tmp_path):
    onnx_path = tmp_path / "microsoft-resnet-50.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    config = make_config(MODEL_PATH=str(tmp_path))

    seen_paths = []

    monkeypatch.setattr("scripts.export_onnx.AutoImageProcessor", FakeImageProcessor)
    monkeypatch.setattr(
        "scripts.export_onnx.AutoModelForImageClassification",
        make_fake_torch_model_cls(logits=torch.tensor([[0.1, 5.0]])),
    )

    def fake_session(path, providers=None):
        seen_paths.append(path)
        return FakeInferenceSession(path, logits=[[0.1, 5.0]], providers=providers)

    monkeypatch.setattr(
        "scripts.export_onnx.onnxruntime",
        types.SimpleNamespace(InferenceSession=fake_session),
    )

    verify_parity(config)

    assert seen_paths == [str(onnx_path)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_runs_export_then_verify_parity_by_default(monkeypatch):
    calls = []
    fake_config = make_config()

    monkeypatch.delenv("SKIP_PARITY_CHECK", raising=False)
    monkeypatch.setattr("scripts.export_onnx.Config.from_env", lambda: fake_config)
    monkeypatch.setattr(
        "scripts.export_onnx.export", lambda config: calls.append(("export", config))
    )
    monkeypatch.setattr(
        "scripts.export_onnx.verify_parity",
        lambda config: calls.append(("verify_parity", config)),
    )

    main()

    assert calls == [("export", fake_config), ("verify_parity", fake_config)]


def test_main_skips_verify_parity_when_env_flag_set(monkeypatch, capsys):
    calls = []
    fake_config = make_config()

    monkeypatch.setenv("SKIP_PARITY_CHECK", "true")
    monkeypatch.setattr("scripts.export_onnx.Config.from_env", lambda: fake_config)
    monkeypatch.setattr(
        "scripts.export_onnx.export", lambda config: calls.append(("export", config))
    )
    monkeypatch.setattr(
        "scripts.export_onnx.verify_parity",
        lambda config: calls.append(("verify_parity", config)),
    )

    main()

    assert calls == [("export", fake_config)]  # verify_parity must not run
    assert "Skipped parity verification" in capsys.readouterr().out


def test_main_treats_skip_flag_case_insensitively(monkeypatch):
    calls = []
    fake_config = make_config()

    monkeypatch.setenv("SKIP_PARITY_CHECK", "TRUE")
    monkeypatch.setattr("scripts.export_onnx.Config.from_env", lambda: fake_config)
    monkeypatch.setattr(
        "scripts.export_onnx.export", lambda config: calls.append(("export", config))
    )
    monkeypatch.setattr(
        "scripts.export_onnx.verify_parity",
        lambda config: calls.append(("verify_parity", config)),
    )

    main()

    assert calls == [("export", fake_config)]


def test_main_does_not_skip_on_unrelated_env_value(monkeypatch):
    """Guards against a typo'd or truthy-looking-but-wrong value silently
    skipping the check — only the exact "true" string (case-insensitive)
    should skip it."""
    calls = []
    fake_config = make_config()

    monkeypatch.setenv("SKIP_PARITY_CHECK", "1")
    monkeypatch.setattr("scripts.export_onnx.Config.from_env", lambda: fake_config)
    monkeypatch.setattr(
        "scripts.export_onnx.export", lambda config: calls.append(("export", config))
    )
    monkeypatch.setattr(
        "scripts.export_onnx.verify_parity",
        lambda config: calls.append(("verify_parity", config)),
    )

    main()

    assert calls == [("export", fake_config), ("verify_parity", fake_config)]
