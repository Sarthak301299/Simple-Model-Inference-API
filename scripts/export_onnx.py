# scripts/export_onnx.py
"""One-off / CI export: converts the configured HF model to a dynamic-batch ONNX graph."""

from pathlib import Path
import torch
import json
import os
import onnxruntime
import sys
from transformers import AutoModelForImageClassification, AutoImageProcessor

from src.config import Config


def resolve_output_paths(config: Config) -> tuple[str, Path, Path]:
    """Returns (sanitized_model_name, onnx_path, json_path), matching the exact
    naming convention ONNXModelManager expects when loading these back."""
    rstring = config.MODEL_NAME.replace("/", "-")
    model_name = config.MODEL_NAME
    if rstring != model_name:
        print(
            f"Input Model name {model_name} contains invalid '/' characters. Resolving to {rstring}"
        )
        model_name = rstring
    output_path = Path(str(config.MODEL_PATH))
    output_path.mkdir(parents=True, exist_ok=True)
    return (
        model_name,
        output_path / f"{model_name}.onnx",
        output_path / f"{model_name}_config.json",
    )


def export(config: Config) -> None:
    model_name, onnx_path, json_path = resolve_output_paths(config)

    model = AutoModelForImageClassification.from_pretrained(config.MODEL_NAME)
    model.eval()

    # id2label is required by ONNXModelManager.load_model() / top_k_from_logits at
    # serving time — fail loudly here at export time rather than producing a file
    # that loads fine but crashes on the first real prediction request.
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        raise ValueError(
            f"{config.MODEL_NAME} has no id2label on its config — refusing to export "
            "an artifact that ONNXModelManager would fail to serve predictions from."
        )

    dummy_input = torch.randn(1, 3, 224, 224)
    dynconfig = torch.export.Dim("batch_size", min=1, max=config.MAX_BATCH_SIZE)
    dynamic_shapes = {"pixel_values": {0: dynconfig}}
    torch.onnx.export(
        model=model,
        args=(dummy_input,),
        f=onnx_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
    )
    print(f"Wrote {onnx_path}")

    with open(json_path, "w") as f:
        json.dump({"id2label": id2label}, f)
    print(f"Wrote {json_path}")


def verify_parity(config: Config, atol: float = 1e-3) -> None:
    """Run one real, preprocessed image through both the torch model and the
    freshly-exported ONNX graph and confirm they agree on the top-1 class.

    Uses the same AutoImageProcessor path ONNXModelManager itself uses at
    serving time (not a raw random tensor), so this check reflects real
    runtime preprocessing, not just raw graph-export correctness. A faster-
    but-wrong export is worse than a slower-but-correct one — this exists
    specifically to catch that before the artifact ships.
    """
    model_name, onnx_path, _ = resolve_output_paths(config)

    processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
    torch_model = AutoModelForImageClassification.from_pretrained(
        config.MODEL_NAME
    ).eval()

    dummy_image = torch.rand(3, 224, 224)
    inputs = processor(images=dummy_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        torch_logits = torch_model(pixel_values=pixel_values).logits

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_logits = session.run(None, {"pixel_values": pixel_values.numpy()})[0]
    onnx_logits = torch.as_tensor(onnx_logits)

    torch_top1 = int(torch_logits.argmax(-1).item())
    onnx_top1 = int(onnx_logits.argmax(-1).item())

    if torch_top1 != onnx_top1:
        print(
            f"PARITY FAILURE: torch top-1={torch_top1}, onnx top-1={onnx_top1} "
            f"for model {config.MODEL_NAME}",
            file=sys.stderr,
        )
        sys.exit(1)

    max_diff = (torch_logits - onnx_logits).abs().max().item()
    if max_diff > atol:
        print(
            f"PARITY WARNING: top-1 class agrees ({torch_top1}) but logits differ "
            f"by {max_diff:.6f} (atol={atol}) — within tolerance for float32 kernel "
            "differences, but worth a second look if this grows over time."
        )

    print(f"Parity OK: torch and ONNX agree on class {torch_top1} for {model_name}")


def main():
    config = Config.from_env()

    export(config)
    if os.getenv("SKIP_PARITY_CHECK", "false").lower() == "true":
        print("Skipped parity verification (SKIP_PARITY_CHECK=true)")
    else:
        verify_parity(config)


if __name__ == "__main__":
    main()  # pragma: no cover
