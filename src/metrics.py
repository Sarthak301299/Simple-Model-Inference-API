"""Measuring metrics via prometheus_client"""

from prometheus_client import Counter, Histogram, Gauge
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

REQUEST_COUNT = Counter(
    "inference_api_requests_total", "Total HTTP requests", ["endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "inference_api_request_duration_seconds", "HTTP request latency", ["endpoint"]
)
INFERENCE_LATENCY = Histogram(
    "inference_api_inference_duration_seconds",
    "Server-reported model inference time",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
BATCH_SIZE = Histogram(
    "inference_api_batch_size",
    "Size of batch processed by the inference worker",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
REJECTED_COUNT = Counter(
    "inference_api_rejected_total", "Requests rejected due to queue saturation"
)
MODEL_READY = Gauge(
    "inference_api_model_ready", "1 if the model is loaded and ready, else 0"
)


class QueueDepthCollector(Collector):
    def __init__(self, engine_getter):
        self.engine_getter = engine_getter

    def collect(self):
        engine = self.engine_getter()
        depth = engine.inference_queue.qsize() if engine else 0
        capacity = engine.inference_queue.maxsize if engine else 0
        depth_metric = GaugeMetricFamily(
            "inference_api_queue_depth", "Current inference queue depth"
        )
        depth_metric.add_metric([], depth)
        yield depth_metric

        capacity_metric = GaugeMetricFamily(
            "inference_api_queue_capacity", "Current max queue capacity"
        )
        capacity_metric.add_metric([], capacity)
        yield capacity_metric
