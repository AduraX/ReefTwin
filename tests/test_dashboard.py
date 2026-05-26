"""Tests for dashboard and expanded Prometheus metrics."""



def test_dashboard_module_imports():
    """Dashboard app module should be importable."""
    # We can't fully run Streamlit in tests, but we can verify imports
    import services.dashboard.app  # noqa: F401 - just check import


def test_prometheus_metrics_defined():
    """Verify all 8 Prometheus metrics are registered."""
    from services.twin_api.main import (
        REQUEST_LATENCY,
        SIMULATION_REQUESTS,
        PREDICTION_LATENCY,
        RAG_QUERIES,
        AGENT_QUERIES,
        AGENT_TOOL_CALLS,
        REEF_STATE_UPDATES,
        DRIFT_ALERTS,
    )
    # All should be prometheus_client metric objects
    assert REQUEST_LATENCY is not None
    assert SIMULATION_REQUESTS is not None
    assert PREDICTION_LATENCY is not None
    assert RAG_QUERIES is not None
    assert AGENT_QUERIES is not None
    assert AGENT_TOOL_CALLS is not None
    assert REEF_STATE_UPDATES is not None
    assert DRIFT_ALERTS is not None


def test_metrics_endpoint_has_new_metrics():
    from fastapi.testclient import TestClient
    from services.twin_api.main import app
    client = TestClient(app)
    response = client.get("/metrics")
    text = response.text
    assert "api_request_latency_seconds" in text
    assert "simulation_requests_total" in text
    assert "rag_queries_total" in text
    assert "agent_queries_total" in text
    assert "drift_alerts_total" in text


def test_grafana_dashboard_json_exists():
    from pathlib import Path
    dashboard_path = Path("infra/grafana/provisioning/dashboards/json/reeftwin-overview.json")
    assert dashboard_path.exists()
    import json
    data = json.loads(dashboard_path.read_text())
    assert data["title"] == "ReefTwin Overview"
    assert len(data["panels"]) >= 6
