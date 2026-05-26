"""Tests for security risk treatments (R-001 through R-009)."""

import pytest


# --- R-001: API Auth ---

def test_api_key_auth_rejects_invalid(monkeypatch):
    monkeypatch.setenv("REEFTWIN_API_KEYS", "valid-key-123")
    monkeypatch.setenv("REEFTWIN_AUTH_MODE", "apikey")
    # Reset cached keys
    import infrastructure.security as sec
    sec._VALID_KEYS = None

    from fastapi.testclient import TestClient
    from services.twin_api.main import app
    client = TestClient(app)
    r = client.post("/rag", json={"question": "test"}, headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401

    sec._VALID_KEYS = None  # cleanup


def test_api_key_auth_accepts_valid(monkeypatch):
    monkeypatch.setenv("REEFTWIN_API_KEYS", "valid-key-123")
    monkeypatch.setenv("REEFTWIN_AUTH_MODE", "apikey")
    import infrastructure.security as sec
    sec._VALID_KEYS = None

    from fastapi.testclient import TestClient
    from services.twin_api.main import app
    client = TestClient(app)
    r = client.post("/rag", json={"question": "test"}, headers={"X-API-Key": "valid-key-123"})
    assert r.status_code == 200

    sec._VALID_KEYS = None


def test_auth_mode_none_allows_all(monkeypatch):
    monkeypatch.setenv("REEFTWIN_AUTH_MODE", "none")
    import infrastructure.security as sec
    sec._VALID_KEYS = None

    from fastapi.testclient import TestClient
    from services.twin_api.main import app
    client = TestClient(app)
    r = client.post("/rag", json={"question": "test"})
    assert r.status_code == 200

    sec._VALID_KEYS = None


# --- R-002: Model integrity ---

def test_model_hash_verification(tmp_path):
    from models.bleaching_risk.inference import verify_model_integrity
    import joblib
    model_path = tmp_path / "test.joblib"
    joblib.dump({"model": "test"}, model_path)
    digest = verify_model_integrity(model_path)
    assert len(digest) == 64  # SHA256 hex
    # Same file = same hash
    assert verify_model_integrity(model_path) == digest


# --- R-004: Input validation ---

def test_reef_id_validation_valid():
    from infrastructure.security import validate_reef_id
    assert validate_reef_id("gbr_heron_reef") == "gbr_heron_reef"


def test_reef_id_validation_rejects_injection():
    from infrastructure.security import validate_reef_id
    with pytest.raises(Exception):  # HTTPException
        validate_reef_id("../../../etc/passwd")


def test_reef_id_validation_rejects_spaces():
    from infrastructure.security import validate_reef_id
    with pytest.raises(Exception):
        validate_reef_id("reef with spaces")


def test_query_length_validation():
    from infrastructure.security import validate_query_length
    assert validate_query_length("short query") == "short query"


def test_query_length_rejects_long():
    from infrastructure.security import validate_query_length
    with pytest.raises(Exception):
        validate_query_length("x" * 3000)


# --- R-006: Rate limiting ---

def test_rate_limiter():
    from infrastructure.security import RateLimiter
    limiter = RateLimiter(requests_per_minute=3)
    limiter.check("127.0.0.1")
    limiter.check("127.0.0.1")
    limiter.check("127.0.0.1")
    with pytest.raises(Exception):  # HTTPException 429
        limiter.check("127.0.0.1")


def test_rate_limiter_different_ips():
    from infrastructure.security import RateLimiter
    limiter = RateLimiter(requests_per_minute=2)
    limiter.check("1.1.1.1")
    limiter.check("1.1.1.1")
    limiter.check("2.2.2.2")  # different IP — should succeed


# --- R-009: AI Fairness ---

def test_fairness_report(tmp_path):
    from pipelines.simulate_iot_stream import generate_readings
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.build_features import build_features
    from models.bleaching_risk.train import train_model
    from infrastructure.mlops.fairness import compute_group_parity

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    generate_readings(500).to_csv(iot_path, index=False)
    generate_noaa_sample(10).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(tmp_path / "features.csv", index=False)
    train_model(str(tmp_path / "features.csv"), str(tmp_path / "model.joblib"))

    report = compute_group_parity(str(tmp_path / "model.joblib"), str(tmp_path / "features.csv"))
    assert report.overall_accuracy > 0
    assert len(report.group_metrics) > 0
    assert len(report.feature_importance) > 0
    assert len(report.findings) > 0
    assert report.parity_gap >= 0


def test_feature_importance(tmp_path):
    from pipelines.simulate_iot_stream import generate_readings
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.build_features import build_features
    from models.bleaching_risk.train import train_model
    from infrastructure.mlops.fairness import compute_feature_importance

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    generate_readings(500).to_csv(iot_path, index=False)
    generate_noaa_sample(10).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(tmp_path / "features.csv", index=False)
    train_model(str(tmp_path / "features.csv"), str(tmp_path / "model.joblib"))

    importance = compute_feature_importance(str(tmp_path / "model.joblib"), str(tmp_path / "features.csv"))
    assert len(importance) == 9  # 9 features
    assert all(isinstance(v, float) for v in importance.values())
