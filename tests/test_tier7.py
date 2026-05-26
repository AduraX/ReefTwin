"""Tests for Tier 7: anomaly detection, forecasting, edge, vision, FTI, fallback chains."""

import numpy as np


# --- Anomaly Detection ---

def test_train_anomaly_detector(tmp_path):
    from pipelines.simulate_iot_stream import generate_readings
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.build_features import build_features
    from models.anomaly_detection.detector import train_anomaly_detector

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    generate_readings(500).to_csv(iot_path, index=False)
    generate_noaa_sample(10).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(tmp_path / "features.csv", index=False)

    metrics = train_anomaly_detector(str(tmp_path / "features.csv"), str(tmp_path / "model.joblib"))
    assert metrics["n_samples"] > 0
    assert 0 <= metrics["anomaly_rate"] <= 1


def test_detect_anomaly(tmp_path):
    from pipelines.simulate_iot_stream import generate_readings
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.build_features import build_features
    from models.anomaly_detection.detector import train_anomaly_detector, detect_anomaly

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    generate_readings(500).to_csv(iot_path, index=False)
    generate_noaa_sample(10).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(tmp_path / "features.csv", index=False)
    train_anomaly_detector(str(tmp_path / "features.csv"), str(tmp_path / "model.joblib"))

    normal = {"water_temperature_c": 28.3, "ph": 8.05, "salinity_psu": 35.1, "turbidity_ntu": 0.8, "dissolved_oxygen_mg_l": 6.5}
    result = detect_anomaly(str(tmp_path / "model.joblib"), normal)
    assert not result.is_anomaly  # normal reading should not be anomalous

    extreme = {"water_temperature_c": 45.0, "ph": 5.0, "salinity_psu": 60.0, "turbidity_ntu": 50.0, "dissolved_oxygen_mg_l": 0.1}
    result = detect_anomaly(str(tmp_path / "model.joblib"), extreme)
    assert result.is_anomaly  # extreme values should be anomalous


# --- Forecasting ---

def test_forecast_parameter():
    from models.forecasting.forecaster import forecast_parameter
    series = np.array([28.0, 28.2, 28.5, 28.7, 29.0, 29.3, 29.5])
    result = forecast_parameter(series, horizon=3)
    assert len(result["forecast"]) == 3
    assert len(result["lower"]) == 3
    assert len(result["upper"]) == 3
    assert result["trend"] in ("rising", "falling", "stable")


def test_forecast_short_series():
    from models.forecasting.forecaster import forecast_parameter
    series = np.array([28.0, 29.0])
    result = forecast_parameter(series, horizon=3)
    assert len(result["forecast"]) == 3


def test_forecast_reef(tmp_path):
    from pipelines.simulate_iot_stream import generate_readings
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.build_features import build_features
    from models.forecasting.forecaster import forecast_reef

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    generate_readings(500).to_csv(iot_path, index=False)
    generate_noaa_sample(10).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(tmp_path / "features.csv", index=False)

    results = forecast_reef(str(tmp_path / "features.csv"), "gbr_heron_reef", horizon_days=5)
    assert len(results) > 0
    assert results[0].horizon_days == 5
    assert len(results[0].forecast_values) == 5


# --- Edge / Lightweight Predictor ---

def test_lightweight_predictor(tmp_path):
    from pipelines.simulate_iot_stream import generate_readings
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.build_features import build_features
    from models.bleaching_risk.train import train_model
    from models.edge.exporter import LightweightPredictor

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    model_path = tmp_path / "model.joblib"
    generate_readings(500).to_csv(iot_path, index=False)
    generate_noaa_sample(10).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(tmp_path / "features.csv", index=False)
    train_model(str(tmp_path / "features.csv"), str(model_path))

    predictor = LightweightPredictor(model_path)
    row = features.iloc[0].to_dict()
    result = predictor.predict(row)
    assert 0.0 <= result["bleaching_risk_score"] <= 1.0
    assert result["risk_category"] in ("normal", "watch", "warning", "alert")
    assert result["inference_mode"] == "edge_lightweight"


# --- Coral Vision ---

def test_extract_image_features():
    from models.coral_vision.classifier import extract_image_features
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    features = extract_image_features(img)
    assert features.shape[0] > 10  # should have multiple features


def test_train_and_classify():
    from models.coral_vision.classifier import train_vision_model, classify_image
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        metrics = train_vision_model(f"{tmp}/model.joblib", n_per_class=30)
        assert metrics["accuracy"] > 0.5  # should do better than random on synthetic

        healthy_img = np.full((32, 32, 3), [180, 100, 60], dtype=np.uint8)
        result = classify_image(f"{tmp}/model.joblib", healthy_img)
        assert result.predicted_class in ("healthy", "bleached", "dead")
        assert 0 <= result.confidence <= 1


# --- FTI Architecture ---

def test_feature_pipeline():
    from infrastructure.fti import run_feature_pipeline
    result = run_feature_pipeline(iot_rows=200, noaa_days=10)
    assert result.status == "success"
    assert result.outputs["feature_rows"] > 0


def test_full_fti():
    from infrastructure.fti import run_full_fti
    results = run_full_fti()
    assert len(results) == 3
    assert all(r.status == "success" for r in results)


# --- Fallback Chains ---

def test_fallback_chain_primary_succeeds():
    from models.fallback import FallbackChain, HeuristicPredictor
    chain = FallbackChain([HeuristicPredictor()])
    result = chain.predict({"degree_heating_weeks": 6, "water_temperature_c": 30})
    assert not result.was_fallback
    assert result.model_used == "heuristic"
    assert result.prediction.bleaching_risk_score > 0


def test_fallback_chain_falls_back():
    from models.fallback import FallbackChain, HeuristicPredictor
    from models.predictor import BleachingPredictor

    class FailingPredictor(BleachingPredictor):
        @property
        def strategy_name(self): return "failing"
        def predict(self, row): raise RuntimeError("Model crashed")

    chain = FallbackChain([FailingPredictor(), HeuristicPredictor()])
    result = chain.predict({"degree_heating_weeks": 4})
    assert result.was_fallback
    assert result.model_used == "heuristic"
    assert result.attempts == 2


def test_fallback_chain_all_fail():
    from models.fallback import FallbackChain
    from models.predictor import BleachingPredictor

    class FailingPredictor(BleachingPredictor):
        @property
        def strategy_name(self): return "failing"
        def predict(self, row): raise RuntimeError("fail")

    chain = FallbackChain([FailingPredictor()])
    result = chain.predict({})
    assert result.was_fallback
    assert result.model_used == "none"
    assert result.prediction.risk_category == "unknown"


def test_heuristic_predictor():
    from models.fallback import HeuristicPredictor
    p = HeuristicPredictor()
    # DHW >= 8 → alert
    r = p.predict({"degree_heating_weeks": 10})
    assert r.risk_category == "alert"
    assert r.bleaching_risk_score >= 0.85
    # Normal conditions
    r = p.predict({"degree_heating_weeks": 0, "water_temperature_c": 27})
    assert r.risk_category == "normal"
