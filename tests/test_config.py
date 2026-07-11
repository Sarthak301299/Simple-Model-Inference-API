import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make src visible as a top-level import path so tests can use `from config import Config`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import Config


def test_default_config_values():
    """Verify default Config values are loaded correctly when no environment variables are set."""
    with patch("config.load_dotenv") as mocked_load_dotenv:
        mocked_load_dotenv.return_value = True
        with patch.dict(os.environ, {}, clear=True):
            config = Config()

    assert config.MODEL_NAME == "microsoft/resnet-50"
    assert config.MODEL_PATH is None
    assert config.INFERENCE_DEVICE == "cuda"
    assert config.API_HOST == "0.0.0.0"
    assert config.API_KEY is None
    assert config.API_PORT == 8000
    assert config.API_VERSION == "1.0.0"
    assert config.DEBUG is False
    assert config.TIMEOUT == 30
    assert config.MAX_CONTENT_LENGTH == 16 * 1024 * 1024
    assert config.LOG_LEVEL == "INFO"
    assert config.LOG_FORMAT == "json"
    assert config.MAX_FILE_SIZE_MB == 16
    assert config.MAX_IMAGE_DIMENSIONS == (4096, 4096)
    assert config.MODEL_INPUT_SHAPE == (None, 3, 224, 224)
    assert config.TOP_K_PREDICTIONS == 5
    assert config.MAX_TOP_K_PREDICTIONS == 10
    assert config.to_dict(config)["MODEL_NAME"] == config.MODEL_NAME
    assert str(config) == str(config.to_dict(config))


def test_env_overrides_and_parsing():
    """Verify Config reads and parses environment variables correctly, including type conversions."""
    with patch("config.load_dotenv") as mocked_load_dotenv:
        mocked_load_dotenv.return_value = True
        env = {
            "MODEL_NAME": "google/mobilenet_v2_1.4_224",
            "MODEL_PATH": "/tmp/model",
            "INFERENCE_DEVICE": "cpu",
            "API_HOST": "127.0.0.1",
            "API_KEY": "secret",
            "API_PORT": "12345",
            "API_VERSION": "2.0.0",
            "DEBUG": "True",
            "TIMEOUT": "60",
            "MAX_CONTENT_LENGTH": "1024",
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "text",
            "MAX_FILE_SIZE_MB": "32",
            "MAX_IMAGE_DIMENSIONS": "1024,768",
            "MODEL_INPUT_SHAPE": "None,3,128,128",
            "TOP_K_PREDICTIONS": "3",
            "MAX_TOP_K_PREDICTIONS": "8",
        }

        with patch.dict(os.environ, env, clear=True):
            config = Config()

    assert config.MODEL_NAME == env["MODEL_NAME"]
    assert config.MODEL_PATH == env["MODEL_PATH"]
    assert config.INFERENCE_DEVICE == env["INFERENCE_DEVICE"]
    assert config.API_HOST == env["API_HOST"]
    assert config.API_KEY == env["API_KEY"]
    assert config.API_PORT == 12345
    assert config.API_VERSION == env["API_VERSION"]
    assert config.DEBUG is True
    assert config.TIMEOUT == 60
    assert config.MAX_CONTENT_LENGTH == 1024
    assert config.LOG_LEVEL == env["LOG_LEVEL"]
    assert config.LOG_FORMAT == env["LOG_FORMAT"]
    assert config.MAX_FILE_SIZE_MB == 32
    assert config.MAX_IMAGE_DIMENSIONS == (1024, 768)
    assert config.MODEL_INPUT_SHAPE == (None, 3, 128, 128)
    assert config.TOP_K_PREDICTIONS == 3
    assert config.MAX_TOP_K_PREDICTIONS == 8


def make_config(**overrides):
    """Create a minimal Config instance for validation tests with optional overrides."""
    config = Config.__new__(Config)
    config.MODEL_NAME = overrides.get("MODEL_NAME", "microsoft/resnet-50")
    config.MODEL_PATH = overrides.get("MODEL_PATH", None)
    config.INFERENCE_DEVICE = overrides.get("INFERENCE_DEVICE", "cpu")
    config.API_HOST = overrides.get("API_HOST", "127.0.0.1")
    config.API_KEY = overrides.get("API_KEY", None)
    config.API_PORT = overrides.get("API_PORT", 8000)
    config.API_VERSION = overrides.get("API_VERSION", "1.0.0")
    config.DEBUG = overrides.get("DEBUG", False)
    config.TIMEOUT = overrides.get("TIMEOUT", 30)
    config.MAX_CONTENT_LENGTH = overrides.get("MAX_CONTENT_LENGTH", 1024)
    config.LOG_LEVEL = overrides.get("LOG_LEVEL", "INFO")
    config.LOG_FORMAT = overrides.get("LOG_FORMAT", "json")
    config.MAX_FILE_SIZE_MB = overrides.get("MAX_FILE_SIZE_MB", 16)
    config.MAX_IMAGE_DIMENSIONS = overrides.get("MAX_IMAGE_DIMENSIONS", (100, 100))
    config.MODEL_INPUT_SHAPE = overrides.get("MODEL_INPUT_SHAPE", (1, 3, 224, 224))
    config.TOP_K_PREDICTIONS = overrides.get("TOP_K_PREDICTIONS", 5)
    config.MAX_TOP_K_PREDICTIONS = overrides.get("MAX_TOP_K_PREDICTIONS", 10)
    return config


def test_validate_accepts_valid_config():
    """Ensure Config.validate accepts a valid configuration without raising an exception."""
    config = make_config(
        MODEL_NAME="google/mobilenet_v2_1.4_224",
        DEBUG=True,
        LOG_LEVEL="ERROR",
        LOG_FORMAT="text",
        MAX_IMAGE_DIMENSIONS=(100, 100),
        MODEL_INPUT_SHAPE=(1, 3, 224, 224),
    )

    Config.validate(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"MODEL_NAME": "invalid-model"},
        {"MODEL_PATH": 123},
        {"INFERENCE_DEVICE": "tpu"},
        {"API_HOST": ""},
        {"API_KEY": 12345},
        {"API_PORT": 0},
        {"DEBUG": "yes"},
        {"TIMEOUT": 0},
        {"MAX_CONTENT_LENGTH": 0},
        {"LOG_LEVEL": "VERBOSE"},
        {"LOG_FORMAT": "xml"},
        {"MAX_FILE_SIZE_MB": 0},
        {"MAX_IMAGE_DIMENSIONS": (0, 100)},
        {"MODEL_INPUT_SHAPE": (1, 3, 0, 224)},
        {"TOP_K_PREDICTIONS": 0, "MAX_TOP_K_PREDICTIONS": 10},
        {"TOP_K_PREDICTIONS": 11, "MAX_TOP_K_PREDICTIONS": 10},
        {"MAX_TOP_K_PREDICTIONS": 101},
    ],
)
def test_validate_rejects_invalid_configurations(kwargs):
    """Ensure Config.validate raises ValueError for invalid configurations."""
    config = make_config(**kwargs)

    with pytest.raises(ValueError):
        Config.validate(config)
