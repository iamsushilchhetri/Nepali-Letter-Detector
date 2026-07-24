import base64
import io

import pytest
from PIL import Image

from app.main import app as flask_app


class DummyPredictor:
    def predict(self, image, top_k=5):
        return [{"class": "character_1_ka", "label": "क (ka)", "confidence": 0.99}]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.main.get_predictor", lambda: DummyPredictor())
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as c:
        yield c


def _sample_data_url():
    img = Image.new("L", (280, 280), color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def test_index_page_loads(client):
    res = client.get("/")
    assert res.status_code == 200


def test_healthz(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}


def test_predict_returns_predictions(client):
    res = client.post("/predict", json={"image": _sample_data_url()})
    assert res.status_code == 200
    body = res.get_json()
    assert body["predictions"][0]["class"] == "character_1_ka"


def test_predict_missing_image_field(client):
    res = client.post("/predict", json={})
    assert res.status_code == 400


def test_predict_invalid_image_data(client):
    res = client.post("/predict", json={"image": "data:image/png;base64,not-valid-base64!!"})
    assert res.status_code == 400
