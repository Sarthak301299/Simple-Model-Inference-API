"""
Configuration file for the model inference.
This file contains the necessary parameters and settings required to run the model inference process.
It includes paths to model name and path, inference device, API connection settings, and other relevant configurations.
Uses certain default values for parameters if not specified via the .env file.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional, Tuple, cast
from PIL import Image


@dataclass
class Config:
    MODEL_NAME: str = "microsoft/resnet-50"
    MODEL_PATH: Optional[str] = None
    INFERENCE_DEVICE: str = "cuda"
    INFERENCE_BACKEND: str = "torch"
    API_HOST: str = "0.0.0.0"
    API_KEY: Optional[str] = None
    API_PORT: int = 8000
    API_VERSION: str = "1.0.1"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    MAX_FILE_SIZE_MB: int = 16
    MAX_CHUNK_SIZE_MB: int = 1
    MAX_IMAGE_DIMENSIONS: Tuple[int, int] = (4096, 4096)
    MODEL_INPUT_SHAPE: Tuple[Optional[int], int, int, int] = (None, 3, 224, 224)
    TOP_K_PREDICTIONS: int = 5
    MAX_TOP_K_PREDICTIONS: int = 10
    MAX_CONCURRENT_REQUESTS: int = 256
    MAX_BATCH_SIZE: int = 64
    BATCHING_TIMEOUT_MS: int = 3
    API_RETRY: int = 5
    INFERENCE_TIMEOUT: int = 60

    @classmethod
    def from_env(cls) -> "Config":
        """Create a Config instance from environment variables and .env files."""
        load_dotenv(".env.shared")
        load_dotenv(".env")

        model_name = os.getenv("MODEL_NAME", cls.MODEL_NAME)
        model_path = os.getenv("MODEL_PATH", cls.MODEL_PATH)
        inference_device = os.getenv("INFERENCE_DEVICE", cls.INFERENCE_DEVICE)
        inference_backend = os.getenv("INFERENCE_BACKEND", cls.INFERENCE_BACKEND)
        api_host = os.getenv("API_HOST", cls.API_HOST)
        api_key = os.getenv("API_KEY", cls.API_KEY)
        api_port = int(os.getenv("API_PORT", cls.API_PORT))
        api_version = os.getenv("API_VERSION", cls.API_VERSION)
        debug = os.getenv("DEBUG", str(cls.DEBUG)).lower() in ("true", "1", "yes")
        log_level = os.getenv("LOG_LEVEL", cls.LOG_LEVEL)
        log_format = os.getenv("LOG_FORMAT", cls.LOG_FORMAT)
        max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", cls.MAX_FILE_SIZE_MB))
        max_chunk_size_mb = int(os.getenv("MAX_CHUNK_SIZE_MB", cls.MAX_CHUNK_SIZE_MB))
        max_image_dimensions = cast(
            Tuple[int, int],
            (
                tuple(
                    int(x)
                    for x in os.getenv(
                        "MAX_IMAGE_DIMENSIONS",
                        f"{cls.MAX_IMAGE_DIMENSIONS[0]},{cls.MAX_IMAGE_DIMENSIONS[1]}",
                    ).split(",")
                )
            ),
        )
        model_input_shape = cast(
            Tuple[Optional[int], int, int, int],
            (
                tuple(
                    int(x) if x != "None" else None
                    for x in os.getenv(
                        "MODEL_INPUT_SHAPE",
                        f"{cls.MODEL_INPUT_SHAPE[0]},{cls.MODEL_INPUT_SHAPE[1]},{cls.MODEL_INPUT_SHAPE[2]},{cls.MODEL_INPUT_SHAPE[3]}",
                    ).split(",")
                )
            ),
        )
        top_k_predictions = int(os.getenv("TOP_K_PREDICTIONS", cls.TOP_K_PREDICTIONS))
        max_top_k_predictions = int(
            os.getenv("MAX_TOP_K_PREDICTIONS", cls.MAX_TOP_K_PREDICTIONS)
        )
        max_concurrent_requests = int(
            os.getenv("MAX_CONCURRENT_REQUESTS", cls.MAX_CONCURRENT_REQUESTS)
        )
        max_batch_size = int(os.getenv("MAX_BATCH_SIZE", cls.MAX_BATCH_SIZE))
        batching_timeout_ms = int(
            os.getenv("BATCHING_TIMEOUT_MS", cls.BATCHING_TIMEOUT_MS)
        )
        api_retry = int(os.getenv("API_RETRY", cls.API_RETRY))
        inference_timeout = int(os.getenv("INFERENCE_TIMEOUT", cls.INFERENCE_TIMEOUT))

        config = cls(
            MODEL_NAME=model_name,
            MODEL_PATH=model_path,
            INFERENCE_DEVICE=inference_device,
            INFERENCE_BACKEND=inference_backend,
            API_HOST=api_host,
            API_KEY=api_key,
            API_PORT=api_port,
            API_VERSION=api_version,
            DEBUG=debug,
            LOG_LEVEL=log_level,
            LOG_FORMAT=log_format,
            MAX_FILE_SIZE_MB=max_file_size_mb,
            MAX_CHUNK_SIZE_MB=max_chunk_size_mb,
            MAX_IMAGE_DIMENSIONS=max_image_dimensions,
            MODEL_INPUT_SHAPE=model_input_shape,
            TOP_K_PREDICTIONS=top_k_predictions,
            MAX_TOP_K_PREDICTIONS=max_top_k_predictions,
            MAX_CONCURRENT_REQUESTS=max_concurrent_requests,
            MAX_BATCH_SIZE=max_batch_size,
            BATCHING_TIMEOUT_MS=batching_timeout_ms,
            API_RETRY=api_retry,
            INFERENCE_TIMEOUT=inference_timeout,
        )

        cls.validate(config)
        return config

    @classmethod
    def validate(cls, config: "Config") -> None:
        """Validate the configuration parameters and raise ValueError for invalid values."""
        if (
            not isinstance(config.MODEL_NAME, str)
            or not config.MODEL_NAME
            or config.MODEL_NAME
            not in (
                "microsoft/resnet-50",
                "google/mobilenet_v2_1.4_224",
                "resnet-50",
                "mobilenet_v2",
            )
        ):
            raise ValueError(
                'MODEL_NAME must be a non-empty string and one of "microsoft/resnet-50", "google/mobilenet_v2_1.4_224", "resnet-50", or "mobilenet_v2".'
            )
        if config.MODEL_PATH is not None and not isinstance(config.MODEL_PATH, str):
            raise ValueError("MODEL_PATH must be a string or None.")
        if config.INFERENCE_DEVICE not in ("cpu", "cuda"):
            raise ValueError("INFERENCE_DEVICE must be either 'cpu' or 'cuda'.")
        if config.INFERENCE_BACKEND not in ("torch", "onnx", "tensorrt"):
            raise ValueError(
                "INFERENCE_BACKEND must be either 'torch', 'onnx', or 'tensorrt'"
            )
        if not isinstance(config.API_HOST, str) or not config.API_HOST:
            raise ValueError("API_HOST must be a non-empty string.")
        if config.API_KEY is not None and not isinstance(config.API_KEY, str):
            raise ValueError("API_KEY must be a string or None.")
        if not (0 < config.API_PORT < 65536):
            raise ValueError("API_PORT must be an integer between 1 and 65535.")
        if not isinstance(config.DEBUG, bool):
            raise ValueError("DEBUG must be a boolean value.")
        if config.LOG_LEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(
                'LOG_LEVEL must be one of "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".'
            )
        if config.LOG_FORMAT not in ("json", "text"):
            raise ValueError('LOG_FORMAT must be either "json" or "text".')
        if config.MAX_FILE_SIZE_MB <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be a positive integer.")
        if not (0 < config.MAX_CHUNK_SIZE_MB <= config.MAX_FILE_SIZE_MB):
            raise ValueError(
                "MAX_CHUNK_SIZE_MB must be a positive integer less than or equal to MAX_FILE_SIZE_MB"
            )
        if len(config.MAX_IMAGE_DIMENSIONS) != 2 or any(
            d <= 0 for d in config.MAX_IMAGE_DIMENSIONS
        ):
            raise ValueError(
                "MAX_IMAGE_DIMENSIONS must be a tuple of two positive integers."
            )
        if len(config.MODEL_INPUT_SHAPE) != 4 or any(
            d is not None and d <= 0 for d in config.MODEL_INPUT_SHAPE
        ):
            raise ValueError(
                "MODEL_INPUT_SHAPE must be a tuple of four integers, where each integer is either a positive value or None."
            )
        if not (0 < config.TOP_K_PREDICTIONS <= config.MAX_TOP_K_PREDICTIONS):
            raise ValueError(
                "TOP_K_PREDICTIONS must be a positive integer less than or equal to MAX_TOP_K_PREDICTIONS."
            )
        if not (0 < config.MAX_TOP_K_PREDICTIONS <= 100):
            raise ValueError(
                "MAX_TOP_K_PREDICTIONS must be a positive integer less than or equal to 100."
            )
        if not (0 < config.MAX_CONCURRENT_REQUESTS <= 256):
            raise ValueError(
                "MAX_CONCURRENT_REQUESTS must be a positive integer less than or equal to 256."
            )
        if not (0 < config.MAX_BATCH_SIZE <= 64):
            raise ValueError(
                "MAX_BATCH_SIZE must be a positive integer less than or equal to 64."
            )
        if not (0 < config.BATCHING_TIMEOUT_MS <= 5):
            raise ValueError(
                "BATCHING_TIMEOUT_MS must be a positive integer less than or equal to 5."
            )
        if not (0 < config.API_RETRY <= 60):
            raise ValueError(
                "API_RETRY must be a positive integer less than or equal to 60."
            )
        if not (0 < config.INFERENCE_TIMEOUT <= 600):
            raise ValueError(
                "INFERENCE_TIMEOUT must be a positive integer less than or equal to 600."
            )

    def to_dict(self) -> dict:
        """Convert the configuration parameters to a dictionary for easy access and manipulation."""
        return {
            "MODEL_NAME": self.MODEL_NAME,
            "MODEL_PATH": self.MODEL_PATH,
            "INFERENCE_DEVICE": self.INFERENCE_DEVICE,
            "INFERENCE_BACKEND": self.INFERENCE_BACKEND,
            "API_HOST": self.API_HOST,
            "API_KEY": self.API_KEY,
            "API_PORT": self.API_PORT,
            "API_VERSION": self.API_VERSION,
            "DEBUG": self.DEBUG,
            "LOG_LEVEL": self.LOG_LEVEL,
            "LOG_FORMAT": self.LOG_FORMAT,
            "MAX_FILE_SIZE_MB": self.MAX_FILE_SIZE_MB,
            "MAX_CHUNK_SIZE_MB": self.MAX_CHUNK_SIZE_MB,
            "MAX_IMAGE_DIMENSIONS": self.MAX_IMAGE_DIMENSIONS,
            "MODEL_INPUT_SHAPE": self.MODEL_INPUT_SHAPE,
            "TOP_K_PREDICTIONS": self.TOP_K_PREDICTIONS,
            "MAX_TOP_K_PREDICTIONS": self.MAX_TOP_K_PREDICTIONS,
            "MAX_CONCURRENT_REQUESTS": self.MAX_CONCURRENT_REQUESTS,
            "MAX_BATCH_SIZE": self.MAX_BATCH_SIZE,
            "BATCHING_TIMEOUT_MS": self.BATCHING_TIMEOUT_MS,
            "API_RETRY": self.API_RETRY,
            "INFERENCE_TIMEOUT": self.INFERENCE_TIMEOUT,
        }

    def validate_image(self, image: Image.Image, max_dims: Tuple[int, int]) -> bool:
        return (0 < image.width <= max_dims[0]) and (0 < image.height <= max_dims[1])

    def __str__(self) -> str:
        """Return a string representation of the configuration parameters for easy logging and debugging."""
        return str(self.to_dict())
