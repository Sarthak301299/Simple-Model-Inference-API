"""
Configuration file for the model inference.
This file contains the necessary parameters and settings required to run the model inference process.
It includes paths to model name and path, inference device, API connection settings, and other relevant configurations.
Uses certain default values for parameters if not specified via the .env file.
"""

import os
from dotenv import load_dotenv
from typing import cast


class Config:
    MODEL_NAME: str = (
        "microsoft/resnet-50"  # Name of the model to be used for inference.
        # Must be resnet-50 or mobilenet-v2 for this implementation.
        # Other models may require additional modifications to the code.
    )
    MODEL_PATH: str | None = (
        None  # Models path to be used for inference. If None, the model will be downloaded from the Hugging Face Hub.
    )
    INFERENCE_DEVICE: str = (
        "cuda"  # Device to be used for inference (e.g., "cpu" or "cuda")
    )
    API_HOST: str = "0.0.0.0"  # Host URL for the API endpoint
    API_KEY: str | None = None  # API key for authentication (if required)
    API_PORT: int = 8000  # Port number for the API endpoint
    API_VERSION: str = "1.0.0"  # Version of the API
    DEBUG: bool = False  # Enable or disable debug mode for the API
    TIMEOUT: int = 30  # Default timeout for API requests in seconds
    MAX_CONTENT_LENGTH: int = (
        16 * 1024 * 1024
    )  # Default maximum content length for API requests in bytes (16 MB)
    LOG_LEVEL: str = (
        "INFO"  # Default log level for the API (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    )
    LOG_FORMAT: str = "json"  # Default log format for the API (e.g., "json", "text")
    MAX_FILE_SIZE_MB: int = 16  # Maximum file size for uploaded files in MB
    MAX_IMAGE_DIMENSIONS: tuple[int, int] = (
        4096,
        4096,
    )  # Maximum dimensions for uploaded images (width, height)
    MODEL_INPUT_SHAPE: tuple[int | None, int, int, int] = (
        None,
        3,
        224,
        224,
    )  # Default input shape for the model (batch, channels, height, width)
    TOP_K_PREDICTIONS: int = (
        5  # Number of top predictions to return from the model inference
    )
    MAX_TOP_K_PREDICTIONS: int = (
        10  # Maximum number of top predictions allowed for the model inference
    )

    def __init__(self):
        """
        Initialize the configuration by loading environment variables and reading the .env file if available.
        Validate Configuration parameters and set default values for any missing parameters.
        Log the configuration settings for debugging and monitoring purposes.
        Priority is given to environment variables over .env file values, and .env file values over default values.
        """
        load_dotenv(
            ".env.shared"
        )  # Load environment variables from .env.shared file if available
        load_dotenv(".env")  # Load environment variables from .env file if available

        # Attempt to read configuration parameters from environment variables, falling back to default values if not set
        self.MODEL_NAME = os.getenv("MODEL_NAME", self.MODEL_NAME)
        self.MODEL_PATH = os.getenv("MODEL_PATH", self.MODEL_PATH)
        self.INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", self.INFERENCE_DEVICE)
        self.API_HOST = os.getenv("API_HOST", self.API_HOST)
        self.API_KEY = os.getenv("API_KEY", self.API_KEY)
        self.API_PORT = int(os.getenv("API_PORT", self.API_PORT))
        self.API_VERSION = os.getenv("API_VERSION", self.API_VERSION)
        self.DEBUG = os.getenv("DEBUG", str(self.DEBUG)).lower() in ("true", "1", "yes")
        self.TIMEOUT = int(os.getenv("TIMEOUT", self.TIMEOUT))
        self.MAX_CONTENT_LENGTH = int(
            os.getenv("MAX_CONTENT_LENGTH", self.MAX_CONTENT_LENGTH)
        )
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", self.LOG_FORMAT)
        self.MAX_FILE_SIZE_MB = int(
            os.getenv("MAX_FILE_SIZE_MB", self.MAX_FILE_SIZE_MB)
        )
        self.MAX_IMAGE_DIMENSIONS = cast(
            tuple[int, int],
            tuple(
                int(x)
                for x in os.getenv(
                    "MAX_IMAGE_DIMENSIONS",
                    f"{self.MAX_IMAGE_DIMENSIONS[0]},{self.MAX_IMAGE_DIMENSIONS[1]}",
                ).split(",")
            ),
        )
        self.MODEL_INPUT_SHAPE = cast(
            tuple[int | None, int, int, int],
            tuple(
                int(x) if x != "None" else None
                for x in os.getenv(
                    "MODEL_INPUT_SHAPE",
                    f"{self.MODEL_INPUT_SHAPE[0]},{self.MODEL_INPUT_SHAPE[1]},{self.MODEL_INPUT_SHAPE[2]},{self.MODEL_INPUT_SHAPE[3]}",
                ).split(",")
            ),
        )
        self.TOP_K_PREDICTIONS = int(
            os.getenv("TOP_K_PREDICTIONS", self.TOP_K_PREDICTIONS)
        )
        self.MAX_TOP_K_PREDICTIONS = int(
            os.getenv("MAX_TOP_K_PREDICTIONS", self.MAX_TOP_K_PREDICTIONS)
        )

    @classmethod
    def validate(cls, config: "Config") -> None:
        """
        Validate the configuration parameters to ensure they meet the required criteria.
        Raise ValueError if any parameter is invalid or out of acceptable range.
        """
        if (
            not isinstance(config.MODEL_NAME, str)
            or not config.MODEL_NAME
            or config.MODEL_NAME
            not in ("microsoft/resnet-50", "google/mobilenet_v2_1.4_224")
        ):
            raise ValueError(
                'MODEL_NAME must be a non-empty string and one of "microsoft/resnet-50" or "google/mobilenet_v2_1.4_224".'
            )
        if config.MODEL_PATH is not None and not isinstance(config.MODEL_PATH, str):
            raise ValueError("MODEL_PATH must be a string or None.")
        if config.INFERENCE_DEVICE not in ("cpu", "cuda"):
            raise ValueError('INFERENCE_DEVICE must be either "cpu" or "cuda".')
        if not isinstance(config.API_HOST, str) or not config.API_HOST:
            raise ValueError("API_HOST must be a non-empty string.")
        if config.API_KEY is not None and not isinstance(config.API_KEY, str):
            raise ValueError("API_KEY must be a string or None.")
        if not (0 < config.API_PORT < 65536):
            raise ValueError("API_PORT must be an integer between 1 and 65535.")
        if not isinstance(config.DEBUG, bool):
            raise ValueError("DEBUG must be a boolean value.")
        if config.TIMEOUT <= 0:
            raise ValueError("TIMEOUT must be a positive integer.")
        if config.MAX_CONTENT_LENGTH <= 0:
            raise ValueError("MAX_CONTENT_LENGTH must be a positive integer.")
        if config.LOG_LEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(
                'LOG_LEVEL must be one of "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".'
            )
        if config.LOG_FORMAT not in ("json", "text"):
            raise ValueError('LOG_FORMAT must be either "json" or "text".')
        if config.MAX_FILE_SIZE_MB <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be a positive integer.")
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

    @classmethod
    def to_dict(cls, config: "Config") -> dict:
        """
        Convert the configuration parameters to a dictionary for easy access and manipulation.
        Returns a dictionary representation of the configuration parameters.
        """
        return {
            "MODEL_NAME": config.MODEL_NAME,
            "MODEL_PATH": config.MODEL_PATH,
            "INFERENCE_DEVICE": config.INFERENCE_DEVICE,
            "API_HOST": config.API_HOST,
            "API_KEY": config.API_KEY,
            "API_PORT": config.API_PORT,
            "API_VERSION": config.API_VERSION,
            "DEBUG": config.DEBUG,
            "TIMEOUT": config.TIMEOUT,
            "MAX_CONTENT_LENGTH": config.MAX_CONTENT_LENGTH,
            "LOG_LEVEL": config.LOG_LEVEL,
            "LOG_FORMAT": config.LOG_FORMAT,
            "MAX_FILE_SIZE_MB": config.MAX_FILE_SIZE_MB,
            "MAX_IMAGE_DIMENSIONS": config.MAX_IMAGE_DIMENSIONS,
            "MODEL_INPUT_SHAPE": config.MODEL_INPUT_SHAPE,
            "TOP_K_PREDICTIONS": config.TOP_K_PREDICTIONS,
            "MAX_TOP_K_PREDICTIONS": config.MAX_TOP_K_PREDICTIONS,
        }

    def __str__(self) -> str:
        """
        Return a string representation of the configuration parameters for easy logging and debugging.
        """
        return str(self.to_dict(self))