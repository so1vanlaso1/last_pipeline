from fastapi.testclient import TestClient
from exact_fama.api import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_physics():
    client = TestClient(app)
    r = client.post("/predict", json={"type":"physics", "question":"Calculate the current when V = 12 V and R = 4 ohms."})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "3"
    assert data["explanation"]
