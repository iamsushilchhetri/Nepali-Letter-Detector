from __future__ import annotations

import base64
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app as flask_app


class DummyPredictor:
    class_count = 46
    weights_path = PROJECT_ROOT / "models" / "model.pt"

    def predict(self, image, top_k=5):
        return [
            {"class": "character_1_ka", "label": "\u0915 (ka)", "confidence": 0.99},
            {"class": "digit_0", "label": "\u0966 (0)", "confidence": 0.01},
        ][:top_k]


def sample_data_url() -> str:
    image = Image.new("L", (280, 280), color=255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class FlaskAppTests(unittest.TestCase):
    def setUp(self) -> None:
        flask_app.config.update(TESTING=True)
        self.client = flask_app.test_client()

    def test_index_page_loads(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    @patch("app.main.get_predictor", return_value=DummyPredictor())
    def test_model_status_endpoint(self, mocked_predictor) -> None:
        response = self.client.get("/api/model")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ready"])
        self.assertEqual(body["class_count"], 46)
        self.assertEqual(body["checkpoint"], "models/model.pt")
        mocked_predictor.assert_called_once()

    @patch("app.main.get_predictor", return_value=DummyPredictor())
    def test_predict_endpoint_returns_predictions(self, mocked_predictor) -> None:
        response = self.client.post("/api/predict", json={"image": sample_data_url()})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["predictions"][0]["class"], "character_1_ka")
        mocked_predictor.assert_called_once()

    def test_predict_endpoint_requires_input(self) -> None:
        response = self.client.post("/api/predict", json={})
        self.assertEqual(response.status_code, 400)

    def test_predict_endpoint_rejects_invalid_image(self) -> None:
        response = self.client.post(
            "/api/predict",
            json={"image": "data:image/png;base64,not-real-image"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
