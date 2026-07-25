from __future__ import annotations

import base64
import io
from pathlib import Path
import sys

from flask import Flask, jsonify, render_template, request
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.nepali_letter_detector.inference import Predictor


app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def decode_data_url_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1) if "," in data_url else (None, data_url)
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes))


def read_request_image() -> Image.Image:
    uploaded_file = request.files.get("file")
    if uploaded_file and uploaded_file.filename:
        return Image.open(uploaded_file.stream)

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        raise ValueError("Send an uploaded file or an 'image' data URL.")
    return decode_data_url_image(image_data)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/model")
def model_status():
    try:
        predictor = get_predictor()
    except FileNotFoundError as error:
        return jsonify(ready=False, error=str(error)), 503

    return jsonify(
        ready=True,
        class_count=predictor.class_count,
        checkpoint=predictor.weights_path.relative_to(ROOT_DIR).as_posix(),
    )


@app.post("/api/predict")
def predict():
    try:
        image = read_request_image()
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except Exception:
        return jsonify(error="Could not decode the submitted image."), 400

    try:
        predictor = get_predictor()
    except FileNotFoundError as error:
        return jsonify(error=str(error)), 503

    predictions = predictor.predict(image, top_k=5)
    return jsonify(predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
