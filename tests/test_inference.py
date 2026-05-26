import pytest

from models.bleaching_risk.inference import predict_risk, risk_category


def test_risk_category_thresholds():
    assert risk_category(0.90) == "alert"
    assert risk_category(0.85) == "alert"
    assert risk_category(0.75) == "warning"
    assert risk_category(0.70) == "warning"
    assert risk_category(0.55) == "watch"
    assert risk_category(0.50) == "watch"
    assert risk_category(0.25) == "normal"
    assert risk_category(0.0) == "normal"


def test_risk_category_boundary_values():
    assert risk_category(0.849) == "warning"
    assert risk_category(0.699) == "watch"
    assert risk_category(0.499) == "normal"


def test_predict_risk_missing_model():
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        predict_risk("/nonexistent/model.joblib", {"water_temperature_c": 30.0})


def test_predict_risk_with_trained_model(tmp_path):
    """Integration test: train a model then predict with it."""
    from pipelines.build_features import build_features
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.simulate_iot_stream import generate_readings
    from models.bleaching_risk.train import train_model

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    model_path = tmp_path / "model.joblib"

    generate_readings(500).to_csv(iot_path, index=False)
    generate_noaa_sample(10).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(tmp_path / "features.csv", index=False)

    train_model(str(tmp_path / "features.csv"), str(model_path))

    row = features.iloc[0].to_dict()
    result = predict_risk(str(model_path), row)

    assert "bleaching_risk_score" in result
    assert "risk_category" in result
    assert 0.0 <= result["bleaching_risk_score"] <= 1.0
    assert result["risk_category"] in ("normal", "watch", "warning", "alert")
