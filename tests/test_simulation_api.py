
import pytest
from fastapi.testclient import TestClient

from services.twin_api.main import app


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_reefs(client):
    response = client.get("/reefs")
    assert response.status_code == 200
    assert "reefs" in response.json()


def test_get_reef_state_not_found(client):
    response = client.get("/reefs/nonexistent_reef/state")
    assert response.status_code == 404


def test_simulate_not_found(client):
    response = client.post(
        "/simulate",
        json={"reef_id": "nonexistent_reef", "temperature_delta_c": 1.5},
    )
    assert response.status_code == 404


def test_simulate_validation_rejects_bad_duration(client):
    response = client.post(
        "/simulate",
        json={"reef_id": "gbr_heron_reef", "duration_days": 0},
    )
    assert response.status_code == 422


def test_simulate_validation_rejects_excessive_duration(client):
    response = client.post(
        "/simulate",
        json={"reef_id": "gbr_heron_reef", "duration_days": 999},
    )
    assert response.status_code == 422


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_request_latency_seconds" in response.text
