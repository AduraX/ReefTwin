"""Integration tests: full FTI pipeline + API contract tests."""

import pytest
from fastapi.testclient import TestClient


# --- Full FTI Integration Test ---

def test_full_pipeline_end_to_end(tmp_path):
    """Full Feature → Training → Inference pipeline in one test."""
    from infrastructure.fti import run_full_fti
    results = run_full_fti()
    assert len(results) == 3
    assert all(r.status == "success" for r in results)
    # Verify state was actually written
    from infrastructure.db.factory import get_state_store
    states = get_state_store().load_states()
    assert len(states) > 0
    assert all("bleaching_risk_score" in s for s in states)


# --- API Contract Tests ---

@pytest.fixture()
def client():
    from services.twin_api.main import app
    return TestClient(app)


def test_api_health_contract(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_api_reefs_contract(client):
    r = client.get("/reefs")
    assert r.status_code == 200
    data = r.json()
    assert "reefs" in data
    assert isinstance(data["reefs"], list)


def test_api_simulate_valid(client):
    r = client.post("/simulate", json={
        "reef_id": "gbr_heron_reef",
        "temperature_delta_c": 1.5,
        "duration_days": 21,
        "turbidity_delta_pct": 10,
        "ph_delta": -0.1,
    })
    # May be 404 if no state data, or 200 if data exists
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        data = r.json()
        assert "projected_bleaching_risk" in data
        assert 0 <= data["projected_bleaching_risk"] <= 1


def test_api_simulate_invalid_duration(client):
    r = client.post("/simulate", json={"reef_id": "x", "duration_days": -1})
    assert r.status_code == 422


def test_api_rag_contract(client):
    r = client.post("/rag", json={"question": "What is DHW?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_api_rag_invalid_k(client):
    r = client.post("/rag", json={"question": "test", "k": 0})
    assert r.status_code == 422


def test_api_agent_contract(client):
    r = client.post("/agent", json={"query": "check reef status"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "tool_calls" in data


def test_api_query_contract(client):
    r = client.post("/query", json={"query": "list reefs"})
    assert r.status_code == 200
    data = r.json()
    assert "routing" in data
    assert "handler" in data["routing"]


def test_api_metrics_has_all_counters(client):
    r = client.get("/metrics")
    text = r.text
    for metric in ["api_request_latency_seconds", "simulation_requests_total",
                    "rag_queries_total", "agent_queries_total", "drift_alerts_total"]:
        assert metric in text, f"Missing metric: {metric}"
