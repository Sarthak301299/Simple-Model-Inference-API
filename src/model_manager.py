"""
Load a pre-trained model, pre-process images to match model input and make predictions.
"""

import os
import logging
import time
import torch
import gc
import onnxruntime
import json
from transformers import AutoModelForImageClassification, AutoImageProcessor
from typing import Any, Dict, List, Tuple
from PIL import Image
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelManager(ABC):
    """
    Class for Model Functions.
    """

    model_name: str
    device: torch.device
    model: Any = None
    model_path: str | None = None
    image_processor: Any = None
    model_loaded: bool = False
    inference_backend: str = "torch"
    id2label: Dict[int, str] | None = None

    def __init__(
        self,
        model_name: str | None = "microsoft/resnet-50",
        device: str = "cpu",
        model_path: str | None = None,
        inference_backend: str = "torch",
    ) -> None:
        """
        Initialize the manager with configuration only — do not load the model here.

        The actual model and image processor are loaded when `load_model()` is
        called. Keeping `__init__` lightweight makes instantiation faster and
        easier to test.
        """
        if model_name is None:
            raise ValueError("model_name must be provided")
        if model_name not in [
            "microsoft/resnet-50",
            "google/mobilenet_v2_1.4_224",
            "resnet-50",
            "mobilenet_v2",
        ]:
            raise ValueError(
                "model_name must be 'microsoft/resnet-50', 'google/mobilenet_v2_1.4_224', 'resnet-50', or 'mobilenet_v2'"
            )
        if device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")

        self.model_name = model_name
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and device == "cuda"
            else torch.device("cpu")
        )
        self.model_path = model_path
        self.inference_backend = inference_backend
        logger.debug(
            "Initialized ModelManager for %s on %s",
            self.model_name,
            self.device,
        )

    @abstractmethod
    def load_model(self) -> None: ...

    """
    Load the image processor and model weights.

    We first check for the model and image processor in the provided valid model path.
    If no valid path is provided or the model/image_processor does not exist in the path use the HuggingFace cache or download.
    """

    @abstractmethod
    def predict(self, inputs) -> tuple[torch.Tensor, float]: ...

    """
    Run the model forward pass and return logits.

    Args:
        inputs: Preprocessed input sequence (from `preprocess_inputs`).

    Returns:
        A `torch.Tensor` of logits with shape (batch_size, num_classes) and the inference time.
    """

    def _cleanup_backend(self) -> None:
        """Override for backend-specific teardown. No-op by default"""
        pass

    def preprocess_inputs(
        self, inputs: Image.Image | List[Image.Image] | None = None
    ) -> torch.Tensor:
        """
        Convert PIL image(s) into model-ready tensors.

        Args:
            inputs: A single `PIL.Image` or a list of `PIL.Image` objects.

        Returns:
            A list of `torch.Tensor` values on  containing processed pixel values loaded on the device.
        """
        if not self.model_loaded:
            raise ValueError(
                "Model is not loaded. Please load the model before preprocessing inputs."
            )
        if inputs is None:
            raise TypeError("Input images are not provided")

        # Normalize to a sequence of images
        if isinstance(inputs, Image.Image):
            inputs = [inputs]

        if len(inputs) == 0:
            raise ValueError("Input image sequence must not be empty")

        # Use the AutoImageProcessor to prepare tensors and move to device
        logger.debug("Preprocessing %d image(s)", len(inputs))
        processed_inputs: torch.Tensor = self.image_processor(
            inputs, return_tensors="pt"
        )["pixel_values"].to(self.device)
        logger.debug("Preprocessing completed for %d image(s)", len(inputs))
        return processed_inputs

    def get_model_info(self) -> Dict[str, str | None]:
        """Return basic metadata about the manager configuration.

        Note this does not indicate whether the model is currently loaded.
        """
        return {
            "name": self.model_name,
            "device": str(self.device),
            "model_path": self.model_path,
        }

    def top_k_from_logits(
        self, logits: torch.Tensor, k: int
    ) -> List[List[Tuple[str, float]]]:
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.dim() != 2:
            raise ValueError("logits must be a 2D tensor [batch, score]")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise TypeError("k must be a positive integer")
        if k > logits.size(-1):
            raise ValueError("k cannot exceed the number of classes in logits")
        if not self.model_loaded:
            raise ValueError(
                "Model is not loaded. Please call load_model() before calling top_k_from_logits."
            )

        if self.id2label is None:
            raise AttributeError("Model config has no 'id2label' mapping")

        values, indices = logits.topk(k=k, dim=-1)
        results: List[List[Tuple[str, float]]] = []
        for vals_row, inds_row in zip(values, indices):
            row: List[Tuple[str, float]] = []
            for val, idx in zip(vals_row, inds_row):
                label_idx = (
                    int(idx.item()) if isinstance(idx, torch.Tensor) else int(idx)
                )
                try:
                    if isinstance(self.id2label, dict):
                        label = self.id2label.get(label_idx, str(label_idx))
                    else:
                        label = self.id2label[label_idx]
                except Exception:
                    label = str(label_idx)
                row.append((label, float(val.item())))
            results.append(row)
        return results

    def cleanup_model(self) -> None:
        if self.model_loaded:
            self.model = None
            self.image_processor = None
            self.model_loaded = False
            self._cleanup_backend()
            gc.collect()


class TorchModelManager(ModelManager):
    def load_model(self):
        logger.info("Loading model %s onto %s", self.model_name, self.device)
        local_model_dir = (
            self.model_path
            if self.model_path is not None and os.path.isdir(self.model_path)
            else None
        )
        if local_model_dir:
            logger.info("Model Path Exists, Attempting load")
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    local_model_dir, local_files_only=True
                )
                self.model = AutoModelForImageClassification.from_pretrained(
                    local_model_dir, local_files_only=True
                )
            except Exception as e:
                self.model = None
                self.image_processor = None
                logger.info(
                    f"Local model loading failed {e}. Falling back to HuggingFace cache or download."
                )

        if not self.model:
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    self.model_name
                )
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name
                )
            except Exception as exc:
                raise FileNotFoundError(
                    f"Model loading failure Model Name {self.model_name} Model path {self.model_path}. Exception {exc}"
                )

        self.model.to(self.device)
        try:
            self.id2label = getattr(self.model.config, "id2label", None)
        except AttributeError as e:
            self.id2label = None
            logger.error("Model does not have id2label attribute. Error: {e}")
            raise e
        self.model_loaded = True
        logger.info("Model %s loaded successfully", self.model_name)

    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if not self.model_loaded:
            raise ValueError(
                "Model is not loaded. Please load the model before doing predictions."
            )

        # Run in eval mode with no gradient computation
        logger.debug("Starting prediction for %d input item(s)", len(inputs))
        start_time = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Prediction completed in %.3f ms", latency_ms)
        return outputs.logits, latency_ms

    def _cleanup_backend(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class ONNXModelManager(ModelManager):
    providers: onnxruntime.SessionOptions = []
    session: onnxruntime.InferenceSession
    session_path: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rstring = self.model_name.replace("/", "-")
        if rstring != self.model_name:
            logger.warning(
                f"Input Model name {self.model_name} contains invalid '/' characters. Resolving to {rstring}"
            )
            self.model_name = rstring
        if (
            self.model_path is None
            or not os.path.isdir(self.model_path)
            or not os.path.isfile(
                os.path.join(self.model_path, self.model_name) + ".onnx"
            )
        ):
            raise (
                FileNotFoundError(
                    f"Model path {self.model_path}/{self.model_name}.onnx does not exist. Required for ONNX/TensorRT"
                )
            )
        self.session_path = os.path.join(self.model_path, self.model_name) + ".onnx"
        if self.inference_backend == "tensorrt":
            if self.device != torch.device("cuda"):
                raise (ValueError(f"Device {self.device} must be CUDA for TensorRT"))
            self.providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": ".tensorrtcache/",
                    },
                ),
                "CUDAExecutionProvider",
            ]
        elif self.device == torch.device("cuda"):
            self.providers = ["CUDAExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]

    def load_model(self):
        logger.info("Loading model %s onto %s", self.model_name, self.device)
        if self.session_path and self.providers:
            self.session = onnxruntime.InferenceSession(
                self.session_path, self.providers
            )
        else:
            raise ValueError(
                "Model path and inference backend must be provided for ONNX/TensorRT"
            )
        try:
            with open(
                os.path.join(str(self.model_path), f"{self.model_name}_config.json")
            ) as f:
                config = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Config file not found: {e}")
            raise e
        try:
            self.id2label = {int(k): v for k, v in config["id2label"].items()}
        except KeyError as e:
            logger.error(f"Invalid config file: {e}")
            raise e
        self.model_loaded = True
        logger.info("Model %s loaded successfully", self.model_name)

    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if not self.model_loaded:
            raise ValueError(
                "Model is not loaded. Please load the model before doing predictions."
            )

        # Run in eval mode with no gradient computation
        logger.debug("Starting prediction for %d input item(s)", len(inputs))
        start_time = time.perf_counter()
        outputs = self.session.run(None, {"pixel_values": inputs.numpy()})
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Prediction completed in %.3f ms", latency_ms)
        return torch.from_numpy(outputs[0]), latency_ms


InferenceBackendMapping = {
    "torch": TorchModelManager,
    "onnx": ONNXModelManager,
    "tensorrt": ONNXModelManager,
}
