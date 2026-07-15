import queue
from types import SimpleNamespace

from prometheus_client import CollectorRegistry, generate_latest

from src.metrics import QueueDepthCollector


def test_queue_depth_collector_reports_zero_without_engine():
    collector = QueueDepthCollector(lambda: None)
    metrics = list(collector.collect())

    assert metrics[0].samples[0].value == 0
    assert metrics[1].samples[0].value == 0


def test_queue_depth_collector_reports_queue_depth_and_capacity():
    q = queue.Queue(maxsize=3)
    q.put("a")
    q.put("b")
    engine = SimpleNamespace(inference_queue=q)
    collector = QueueDepthCollector(lambda: engine)
    metrics = list(collector.collect())

    assert metrics[0].name == "inference_api_queue_depth"
    assert metrics[0].samples[0].value == 2
    assert metrics[1].name == "inference_api_queue_capacity"
    assert metrics[1].samples[0].value == 3


def test_queue_depth_collector_exports_expected_metric_names():
    registry = CollectorRegistry()
    registry.register(QueueDepthCollector(lambda: None))

    output = generate_latest(registry).decode()

    assert "inference_api_queue_depth" in output
    assert "inference_api_queue_capacity" in output
