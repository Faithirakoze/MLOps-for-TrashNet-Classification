"""Locust scenario for the TrashNet image prediction endpoint."""

import os
import mimetypes
from pathlib import Path

from locust import HttpUser, between, task


IMAGE_PATH = Path(
    os.environ.get("LOCUST_IMAGE", "load_test_images/sample.jpg")
)


class TrashNetPredictionUser(HttpUser):
    """Repeatedly submit a valid image to the prediction API."""

    wait_time = between(0.1, 0.5)

    def on_start(self):
        if not IMAGE_PATH.is_file():
            raise FileNotFoundError(
                f"Locust image not found: {IMAGE_PATH}. Set LOCUST_IMAGE "
                "or place a JPEG at load_test_images/sample.jpg."
            )

        self.image_bytes = IMAGE_PATH.read_bytes()
        self.image_name = IMAGE_PATH.name
        self.image_type = (
            mimetypes.guess_type(self.image_name)[0]
            or "application/octet-stream"
        )

    @task
    def predict_image(self):
        with self.client.post(
            "/predict",
            files={
                "file": (
                    self.image_name,
                    self.image_bytes,
                    self.image_type
                )
            },
            name="POST /predict",
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Prediction returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return

            try:
                result = response.json()
            except ValueError:
                response.failure("Prediction response was not valid JSON.")
                return

            if not all(
                key in result
                for key in ("predicted_class", "confidence", "probabilities")
            ):
                response.failure("Prediction response is missing expected fields.")
