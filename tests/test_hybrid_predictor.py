import pytest

from models.reef_dynamics.hybrid_predictor import (
    compute_physics_prior,
    train_hybrid_model,
    predict_hybrid,
)


def test_physics_prior_cool_water():
    prior = compute_physics_prior({"water_temperature_c": 26.0, "degree_heating_weeks": 0.0})
    assert prior["physics_risk"] < 0.3
    assert prior["physics_stress"] < 0.1


def test_physics_prior_hot_water():
    prior = compute_physics_prior({"water_temperature_c": 31.5, "degree_heating_weeks": 6.0})
    assert prior["physics_risk"] > 0.2
    assert prior["physics_dhw_ratio"] > 0.5


def test_train_and_predict_hybrid(tmp_path):
    from pipelines.build_features import build_features
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.simulate_iot_stream import generate_readings

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    model_path = tmp_path / "hybrid.joblib"
    features_path = tmp_path / "features.csv"

    generate_readings(500).to_csv(iot_path, index=False)
    generate_noaa_sample(10).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(features_path, index=False)

    metrics = train_hybrid_model(str(features_path), str(model_path))
    assert "mse" in metrics
    assert "mae" in metrics

    row = features.iloc[0].to_dict()
    result = predict_hybrid(str(model_path), row)
    assert 0.0 <= result["bleaching_risk_score"] <= 1.0
    assert result["risk_category"] in ("normal", "watch", "warning", "alert")
    assert "physics_prior" in result


def test_predict_hybrid_missing_model():
    with pytest.raises(FileNotFoundError):
        predict_hybrid("/nonexistent/hybrid.joblib", {"water_temperature_c": 29.0})
