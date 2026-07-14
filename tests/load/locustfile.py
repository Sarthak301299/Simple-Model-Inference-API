import os
import random
import time
from pathlib import Path
from locust import HttpUser, task, between, events

DATA_DIR = Path(__file__).parent / "data"
ALL_IMAGE_FILES = (
    list(DATA_DIR.glob("*.JPEG"))
    + list(DATA_DIR.glob("*.png"))
    + list(DATA_DIR.glob("*.jpg"))
)
BAD_IMAGE_FILES = [f for f in ALL_IMAGE_FILES if "BAD" in f.stem]
IMAGE_FILES = [f for f in ALL_IMAGE_FILES if f not in BAD_IMAGE_FILES]
API_KEY = os.getenv("LOAD_TEST_API_KEY")


class InferenceUser(HttpUser):
    wait_time = between(0.05, 0.3)
    # wait_time = between(0,0)

    def on_start(self) -> None:
        self.headers = {"X-API-Key": API_KEY} if API_KEY else {}

    @task(5)
    def predict(self) -> None:
        img_path = random.choice(IMAGE_FILES)
        with open(img_path, "rb") as file:
            files = {"file": (img_path.name, file, "image/jpeg")}
            start = time.perf_counter()
            with self.client.post(
                "/predict", files=files, headers=self.headers, catch_response=True
            ) as response:
                total_ms = (time.perf_counter() - start) * 1000
                if response.status_code == 200:
                    server_ms = response.json().get("inference_time_ms")
                    events.request.fire(
                        request_type="INFER",
                        name="inference_time_ms",
                        response_time=server_ms,
                        response_length=0,
                    )
                    events.request.fire(
                        request_type="WALL",
                        name="predict_wall_time_ms",
                        response_time=total_ms,
                        response_length=0,
                    )
                    response.success()
                elif response.status_code == 429:
                    # Overload condition: Marked as success as it's a temporary error and the user should retry
                    response.success()
                else:
                    response.failure(
                        f"Unexpected Status {response.status_code}: {response.text}"
                    )

    @task(1)
    def predict_invalid(self):
        img_path = random.choice(BAD_IMAGE_FILES)
        with open(img_path, "rb") as file:
            files = {"file": (img_path.name, file, "image/jpeg")}
            with self.client.post(
                "/predict", files=files, headers=self.headers, catch_response=True
            ) as response:
                # A full inference queue can result in a 429 error. We count it as an valid response here.
                if response.status_code in (400, 429):
                    response.success()
                else:
                    response.failure(
                        f"Unexpected status for invalid image: {response.status_code}: {response.text}"
                    )
