"""
Load a pre-trained model, pre-process images to match model input and make predictions.
"""

import os
import logging
import time
import torch
import gc
from transformers import AutoModelForImageClassification, AutoImageProcessor
from typing import Any, Dict, List, Tuple
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Class for Model Functions.
    """

    model_name: str
    device: torch.device
    model: Any = None
    model_path: str | None = None
    image_processor: Any = None
    model_loaded: bool = False

    def __init__(
        self,
        model_name: str | None = "microsoft/resnet-50",
        device: str = "cuda",
        model_path: str | None = None,
    ) -> None:
        """
        Initialize the manager with configuration only — do not load the model here.

        The actual model and image processor are loaded when `load_model()` is
        called. Keeping `__init__` lightweight makes instantiation faster and
        easier to test.
        """
        if model_name is None:
            raise ValueError("model_name must be provided")
        if model_name not in ["microsoft/resnet-50", "google/mobilenet_v2_1.4_224"]:
            raise ValueError(
                "model_name must be 'microsoft/resnet-50' or 'google/mobilenet_v2_1.4_224'"
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
        logger.debug(
            "Initialized ModelManager for %s on %s",
            self.model_name,
            self.device,
        )

    def load_model(self) -> None:
        """
        Load the image processor and model weights.

        This may perform network I/O (download) if the model is not available
        locally. After loading the model is moved to `self.device`.
        """
        logger.info("Loading model %s onto %s", self.model_name, self.device)
        # Load image processor from the pretrained model repository
        # (this may download tokenizer/config if not cached)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        try:
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name
            )
        except Exception:
            exc: Exception = Exception()
            logger.warning(
                "Falling back to local model weights from %s",
                self.model_path,
            )
            if self.model is None:
                try:
                    self.model = AutoModelForImageClassification.from_pretrained(
                        self.model_name
                    )
                except Exception as retry_exc:
                    exc = retry_exc
            if self.model_path is not None and os.path.exists(self.model_path):
                if self.model is None:
                    raise RuntimeError(
                        "Unable to initialize a model instance from the pretrained source."
                    ) from exc
                state = torch.load(self.model_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.model.load_state_dict(state)
            else:
                raise FileNotFoundError(f"Model path {self.model_path}") from exc
        self.model.to(self.device)
        self.model_loaded = True
        logger.info("Model %s loaded successfully", self.model_name)

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

    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Run the model forward pass and return logits.

        Args:
            inputs: Preprocessed input sequence (from `preprocess_inputs`).

        Returns:
            A `torch.Tensor` of logits with shape (batch_size, num_classes) and the inference time.
        """

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

        id2label = getattr(self.model.config, "id2label", None)
        if id2label is None:
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
                    if isinstance(id2label, dict):
                        label = id2label.get(label_idx, str(label_idx))
                    else:
                        label = id2label[label_idx]
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
            gc.collect()
            if self.device == torch.device("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
