import os

# Must be set before importing the app: src/model.py reads MODEL_URI at import time.
os.environ["MODEL_URI"] = "artifacts/model"

import pytest
from fastapi.testclient import TestClient
from app.main import app

VALID_REQUEST = {
    "PULocationID": 142,
    "DOLocationID": 236,
    "trip_distance": 3.2,
    "tpep_pickup_datetime": "2025-01-10T08:00:00",
}


@pytest.fixture(scope="module")
def client():
    # The "with" block runs the lifespan, which loads the model.
    with TestClient(app) as client:
        yield client


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200


def test_predict_valid_request(client):
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_duration_minutes" in body
    prediction = body["predicted_duration_minutes"]
    assert isinstance(prediction, float)
    assert 1 <= prediction <= 60


@pytest.mark.parametrize(
    "invalid_request",
    [
        {**VALID_REQUEST, "PULocationID": 0},
        {**VALID_REQUEST, "PULocationID": 300},
        {**VALID_REQUEST, "DOLocationID": 0},
        {**VALID_REQUEST, "trip_distance": 0},
        {**VALID_REQUEST, "trip_distance": -1.5},
        {**VALID_REQUEST, "trip_distance": 150},
        {**VALID_REQUEST, "tpep_pickup_datetime": "kein Datum"},
        {k: v for k, v in VALID_REQUEST.items() if k != "trip_distance"},
    ],
)
def test_predict_invalid_request(client, invalid_request):
    response = client.post("/predict", json=invalid_request)
    assert response.status_code == 422
