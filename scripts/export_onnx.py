# scripts/export_onnx.py
"""One-off / CI export: converts the configured HF model to a dynamic-batch ONNX graph."""

from pathlib import Path
import torch
import json
from transformers import AutoModelForImageClassification

from src.config import Config


def export(config: Config):
    rstring = config.MODEL_NAME.replace("/", "-")
    model_name = config.MODEL_NAME
    if rstring != model_name:
        print(
            f"Input Model name {model_name} contains invalid '/' characters. Resolving to {rstring}"
        )
        model_name = rstring
    output_path = Path(str(config.MODEL_PATH))
    output_path.mkdir(parents=True, exist_ok=True)
    full_name = output_path / f"{model_name}.onnx"
    full_json_name = output_path / f"{model_name}_config.json"

    model = AutoModelForImageClassification.from_pretrained(config.MODEL_NAME)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    dynconfig = torch.export.Dim("batch_size", min=1, max=config.MAX_BATCH_SIZE)
    dynamic_shapes = {"pixel_values": {0: dynconfig}}
    torch.onnx.export(
        model=model,
        args=(dummy_input,),
        f=full_name,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
    )
    id2label = getattr(model.config, "id2label", None)
    jsonconfig = {"id2label": id2label}
    with open(full_json_name, "w") as f:
        json.dump(jsonconfig, f)


if __name__ == "__main__":
    config = Config.from_env()

    export(config)
