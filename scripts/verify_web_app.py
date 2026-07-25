from __future__ import annotations

import base64
import io
from pathlib import Path
import sys

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from src.nepali_letter_detector.inference import Predictor


def create_sample_data_url() -> str:
    image = Image.new("L", (280, 280), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((90, 200, 140, 70), fill=0, width=18)
    draw.line((140, 70, 190, 200), fill=0, width=18)
    draw.line((110, 150, 170, 150), fill=0, width=18)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def main() -> None:
    predictor = Predictor()
    print(f"checkpoint: {predictor.weights_path}")
    print(f"class_count: {predictor.class_count}")

    app.config.update(TESTING=True)
    sample_data_url = create_sample_data_url()

    with app.test_client() as client:
        health_response = client.get("/api/health")
        assert health_response.status_code == 200
        assert health_response.get_json()["status"] == "ok"

        model_response = client.get("/api/model")
        assert model_response.status_code == 200
        model_body = model_response.get_json()
        assert model_body["ready"] is True
        assert model_body["class_count"] == predictor.class_count

        prediction_response = client.post("/api/predict", json={"image": sample_data_url})
        assert prediction_response.status_code == 200
        prediction_body = prediction_response.get_json()
        assert "predictions" in prediction_body
        assert len(prediction_body["predictions"]) == 5
        assert all("confidence" in item for item in prediction_body["predictions"])

        index_response = client.get("/")
        assert index_response.status_code == 200

    print("Web app verification passed.")


if __name__ == "__main__":
    main()
