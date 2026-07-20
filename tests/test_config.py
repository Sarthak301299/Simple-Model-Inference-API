import pytest
from PIL import Image
from src.config import Config


def make_config(**overrides):
    """Create a minimal Config instance for validation tests with optional overrides."""
    defaults = {
        "MODEL_NAME": "microsoft/resnet-50",
        "MODEL_PATH": None,
        "INFERENCE_DEVICE": "cpu",
        "INFERENCE_BACKEND": "torch",
        "API_HOST": "127.0.0.1",
        "API_KEY": None,
        "API_PORT": 8000,
        "API_VERSION": "1.0.0",
        "DEBUG": False,
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "json",
        "MAX_FILE_SIZE_MB": 16,
        "MAX_CHUNK_SIZE_MB": 1,
        "MAX_IMAGE_DIMENSIONS": (100, 100),
        "MODEL_INPUT_SHAPE": (1, 3, 224, 224),
        "TOP_K_PREDICTIONS": 5,
        "MAX_TOP_K_PREDICTIONS": 10,
        "MAX_CONCURRENT_REQUESTS": 256,
        "MAX_BATCH_SIZE": 32,
        "BATCHING_TIMEOUT_MS": 3,
        "API_RETRY": 5,
        "INFERENCE_TIMEOUT": 60,
    }
    return Config(**{**defaults, **overrides})


@pytest.mark.parametrize(
    "env_key,env_value",
    [
        ("API_PORT", "not-an-int"),
        ("MAX_FILE_SIZE_MB", "not-an-int"),
        ("MAX_IMAGE_DIMENSIONS", "1024,abc"),
        ("MODEL_INPUT_SHAPE", "None,3,abc,224"),
    ],
)
def test_from_env_rejects_malformed_numeric_values(monkeypatch, env_key, env_value):
    monkeypatch.setattr("src.config.load_dotenv", lambda path: True)
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    monkeypatch.setenv(env_key, env_value)

    with pytest.raises(ValueError):
        Config.from_env()


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ],
)
def test_from_env_debug_boolean_parsing(monkeypatch, env_value, expected):
    monkeypatch.setattr("src.config.load_dotenv", lambda path: True)
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    monkeypatch.setenv("INFERENCE_BACKEND", "torch")
    monkeypatch.setenv("DEBUG", env_value)

    assert Config.from_env().DEBUG is expected


def test_from_env_parses_remaining_runtime_controls(monkeypatch):
    monkeypatch.setattr("src.config.load_dotenv", lambda path: True)
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    monkeypatch.setenv("INFERENCE_BACKEND", "torch")
    monkeypatch.setenv("MAX_CHUNK_SIZE_MB", "2")
    monkeypatch.setenv("BATCHING_TIMEOUT_MS", "4")
    monkeypatch.setenv("API_RETRY", "30")
    monkeypatch.setenv("INFERENCE_TIMEOUT", "120")

    config = Config.from_env()

    assert config.INFERENCE_DEVICE == "cpu"
    assert config.INFERENCE_BACKEND == "torch"
    assert config.MAX_CHUNK_SIZE_MB == 2
    assert config.BATCHING_TIMEOUT_MS == 4
    assert config.API_RETRY == 30
    assert config.INFERENCE_TIMEOUT == 120


@pytest.mark.parametrize(
    "kwargs",
    [
        {"API_PORT": 1},
        {"API_PORT": 65535},
        {"MAX_TOP_K_PREDICTIONS": 100, "TOP_K_PREDICTIONS": 100},
        {"MAX_CONCURRENT_REQUESTS": 256},
        {"MAX_BATCH_SIZE": 64},
        {"BATCHING_TIMEOUT_MS": 5},
        {"API_RETRY": 60},
        {"INFERENCE_TIMEOUT": 600},
    ],
)
def test_validate_accepts_boundary_values(kwargs):
    Config.validate(make_config(**kwargs))


def test_validate_image_boundaries():
    config = make_config(MAX_IMAGE_DIMENSIONS=(10, 20))

    assert config.validate_image(Image.new("RGB", (10, 20)), (10, 20)) is True
    assert config.validate_image(Image.new("RGB", (11, 20)), (10, 20)) is False
    assert config.validate_image(Image.new("RGB", (10, 21)), (10, 20)) is False


def test_default_config_values(monkeypatch):
    """Verify default Config values are loaded correctly when no environment variables are set."""
    monkeypatch.setattr("src.config.load_dotenv", lambda path: True)
    for env_var in [
        "MODEL_NAME",
        "MODEL_PATH",
        "INFERENCE_DEVICE",
        "INFERENCE_BACKEND",
        "API_HOST",
        "API_KEY",
        "API_PORT",
        "API_VERSION",
        "DEBUG",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "MAX_FILE_SIZE_MB",
        "MAX_CHUNK_SIZE_MB",
        "MAX_IMAGE_DIMENSIONS",
        "MODEL_INPUT_SHAPE",
        "TOP_K_PREDICTIONS",
        "MAX_TOP_K_PREDICTIONS",
        "MAX_CONCURRENT_REQUESTS",
        "MAX_BATCH_SIZE",
        "BATCHING_TIMEOUT_MS",
        "API_RETRY",
        "INFERENCE_TIMEOUT",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    config = Config.from_env()

    assert config.MODEL_NAME == "microsoft/resnet-50"
    assert config.MODEL_PATH is None
    assert config.INFERENCE_DEVICE == "cuda"
    assert config.INFERENCE_BACKEND == "torch"
    assert config.API_HOST == "0.0.0.0"
    assert config.API_KEY is None
    assert config.API_PORT == 8000
    assert config.DEBUG is False
    assert config.LOG_LEVEL == "INFO"
    assert config.LOG_FORMAT == "json"
    assert config.MAX_FILE_SIZE_MB == 16
    assert config.MAX_CHUNK_SIZE_MB == 1
    assert config.MAX_IMAGE_DIMENSIONS == (4096, 4096)
    assert config.MODEL_INPUT_SHAPE == (None, 3, 224, 224)
    assert config.TOP_K_PREDICTIONS == 5
    assert config.MAX_TOP_K_PREDICTIONS == 10
    assert config.MAX_CONCURRENT_REQUESTS == 256
    assert config.MAX_BATCH_SIZE == 64
    assert config.BATCHING_TIMEOUT_MS == 3
    assert config.API_RETRY == 5
    assert config.INFERENCE_TIMEOUT == 60
    assert config.to_dict()["MODEL_NAME"] == config.MODEL_NAME
    assert str(config) == str(config.to_dict())


def test_env_overrides_and_parsing(monkeypatch):
    """Verify Config reads and parses environment variables correctly, including type conversions."""
    monkeypatch.setattr("src.config.load_dotenv", lambda path: True)
    env = {
        "MODEL_NAME": "google/mobilenet_v2_1.4_224",
        "MODEL_PATH": "/tmp/model",
        "INFERENCE_DEVICE": "cpu",
        "INFERENCE_BACKEND": "onnx",
        "API_HOST": "127.0.0.1",
        "API_KEY": "secret",
        "API_PORT": "12345",
        "API_VERSION": "2.0.0",
        "DEBUG": "True",
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "text",
        "MAX_FILE_SIZE_MB": "32",
        "MAX_IMAGE_DIMENSIONS": "1024,768",
        "MODEL_INPUT_SHAPE": "None,3,128,128",
        "TOP_K_PREDICTIONS": "3",
        "MAX_TOP_K_PREDICTIONS": "8",
        "MAX_CONCURRENT_REQUESTS": "32",
        "MAX_BATCH_SIZE": "8",
    }

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    config = Config.from_env()

    assert config.MODEL_NAME == env["MODEL_NAME"]
    assert config.MODEL_PATH == env["MODEL_PATH"]
    assert config.INFERENCE_DEVICE == env["INFERENCE_DEVICE"]
    assert config.INFERENCE_BACKEND == env["INFERENCE_BACKEND"]
    assert config.API_HOST == env["API_HOST"]
    assert config.API_KEY == env["API_KEY"]
    assert config.API_PORT == 12345
    assert config.API_VERSION == env["API_VERSION"]
    assert config.DEBUG is True
    assert config.LOG_LEVEL == env["LOG_LEVEL"]
    assert config.LOG_FORMAT == env["LOG_FORMAT"]
    assert config.MAX_FILE_SIZE_MB == 32
    assert config.MAX_IMAGE_DIMENSIONS == (1024, 768)
    assert config.MODEL_INPUT_SHAPE == (None, 3, 128, 128)
    assert config.TOP_K_PREDICTIONS == 3
    assert config.MAX_TOP_K_PREDICTIONS == 8
    assert config.MAX_CONCURRENT_REQUESTS == 32
    assert config.MAX_BATCH_SIZE == 8


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
        {"INFERENCE_BACKEND": "backend"},
        {"API_HOST": ""},
        {"API_KEY": 12345},
        {"API_PORT": 0},
        {"DEBUG": "yes"},
        {"LOG_LEVEL": "VERBOSE"},
        {"LOG_FORMAT": "xml"},
        {"MAX_FILE_SIZE_MB": 0},
        {"MAX_CHUNK_SIZE_MB": 32},
        {"MAX_IMAGE_DIMENSIONS": (0, 100)},
        {"MAX_IMAGE_DIMENSIONS": (100, "large")},
        {"MODEL_INPUT_SHAPE": (1, 3, 0, 224)},
        {"MODEL_INPUT_SHAPE": (None, 3, "wide", 224)},
        {"TOP_K_PREDICTIONS": 0, "MAX_TOP_K_PREDICTIONS": 10},
        {"TOP_K_PREDICTIONS": 11, "MAX_TOP_K_PREDICTIONS": 10},
        {"MAX_TOP_K_PREDICTIONS": 101},
        {"MAX_CONCURRENT_REQUESTS": 0},
        {"MAX_CONCURRENT_REQUESTS": 512},
        {"MAX_BATCH_SIZE": 0},
        {"MAX_BATCH_SIZE": 128},
        {"BATCHING_TIMEOUT_MS": 0},
        {"BATCHING_TIMEOUT_MS": 10},
        {"API_RETRY": 0},
        {"API_RETRY": 61},
        {"INFERENCE_TIMEOUT": 0},
        {"INFERENCE_TIMEOUT": 601},
    ],
)
def test_validate_rejects_invalid_configurations(kwargs):
    """Config.validate should raise for any invalid config, whether the
    problem is a wrong type or an out-of-range value."""
    config = make_config(**kwargs)
    with pytest.raises((TypeError, ValueError)):
        Config.validate(config)
