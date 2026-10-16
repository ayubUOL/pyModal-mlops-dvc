# API tests: do the API's "doors" work?

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_returns_a_price():
    response = client.post("/predict", json={"rooms": 3, "size_sqft": 1350})
    assert response.status_code == 200
    assert response.json()["price_lakh"] > 0

def test_bad_input_is_rejected():
    response = client.post("/predict", json={"rooms": "three", "size_sqft": 1350})
    assert response.status_code == 422
