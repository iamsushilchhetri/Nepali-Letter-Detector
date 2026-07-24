import base64
import io

from flask import Flask, jsonify, render_template, request
from PIL import Image

from app.inference import Predictor

app = Flask(__name__)
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url:
        return jsonify(error="Missing 'image' field"), 400

    try:
        _, encoded = data_url.split(",", 1) if "," in data_url else (None, data_url)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return jsonify(error="Could not decode image"), 400

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503

    predictions = predictor.predict(image, top_k=5)
    return jsonify(predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
