import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


valid_payload = {
    "Trip_Distance_km":      5.0,
    "Time_of_Day":           "Morning",
    "Day_of_Week":           "Weekday",
    "Passenger_Count":       1.0,
    "Traffic_Conditions":    "Low",
    "Weather":               "Clear",
    "Base_Fare":             2.5,
    "Per_Km_Rate":           1.5,
    "Per_Minute_Rate":       0.5,
    "Trip_Duration_Minutes": 20.0,
}


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_returns_expected_keys(client):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_fare" in body
    assert "shap_values" in body


def test_bad_input_returns_422(client):
    bad_payload = valid_payload.copy()
    bad_payload["Time_of_Day"] = "Midnight"  # not a valid Literal value
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
