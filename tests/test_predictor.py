import pytest

from models.predictor import (
    get_predictor,
    PredictionResult,
)


def test_get_predictor_random_forest(tmp_path):
    """Train an RF model then use the strategy pattern to predict."""
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

    predictor = get_predictor("random_forest", rf_model_path=model_path)
    assert predictor.strategy_name == "random_forest"

    row = features.iloc[0].to_dict()
    result = predictor.predict(row)
    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.bleaching_risk_score <= 1.0
    assert result.model_strategy == "random_forest"


def test_get_predictor_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_predictor("nonexistent_strategy")
