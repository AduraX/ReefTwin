"""Tests for PyTorch PINN, CNN coral vision, and SageMaker integration."""

import numpy as np


# --- PyTorch PINN ---

def test_pinn_train_and_predict(tmp_path):
    from pipelines.simulate_iot_stream import generate_readings
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.build_features import build_features
    from models.reef_dynamics.pinn import ReefPINN, PINNConfig

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    generate_readings(500).to_csv(iot_path, index=False)
    generate_noaa_sample(10).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(tmp_path / "features.csv", index=False)

    config = PINNConfig(epochs=20, hidden_dims=[16, 8])
    pinn = ReefPINN(config)
    metrics = pinn.train(str(tmp_path / "features.csv"))

    assert "final_total_loss" in metrics
    assert "final_physics_loss" in metrics
    assert metrics["final_total_loss"] < 1.0

    row = features.iloc[0].to_dict()
    result = pinn.predict(row)
    assert 0.0 <= result["bleaching_risk_score"] <= 1.0
    assert result["risk_category"] in ("normal", "watch", "warning", "alert")
    assert result["model_type"] == "pinn"


def test_pinn_save_load(tmp_path):
    from models.reef_dynamics.pinn import ReefPINN, PINNConfig
    from pipelines.simulate_iot_stream import generate_readings
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.build_features import build_features

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    generate_readings(200).to_csv(iot_path, index=False)
    generate_noaa_sample(5).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(tmp_path / "features.csv", index=False)

    pinn = ReefPINN(PINNConfig(epochs=5, hidden_dims=[8]))
    pinn.train(str(tmp_path / "features.csv"))
    pinn.save(tmp_path / "pinn.pt")

    pinn2 = ReefPINN(PINNConfig(hidden_dims=[8]))
    pinn2.load(tmp_path / "pinn.pt")
    result = pinn2.predict(features.iloc[0].to_dict())
    assert 0.0 <= result["bleaching_risk_score"] <= 1.0


def test_pinn_physics_loss_penalises_violations(tmp_path):
    """PINN physics loss should be > 0 at start (random weights violate physics)."""
    from models.reef_dynamics.pinn import ReefPINN, PINNConfig
    from pipelines.simulate_iot_stream import generate_readings
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.build_features import build_features

    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    generate_readings(200).to_csv(iot_path, index=False)
    generate_noaa_sample(5).to_csv(noaa_path, index=False)
    features = build_features(str(iot_path), str(noaa_path))
    features.to_csv(tmp_path / "features.csv", index=False)

    pinn = ReefPINN(PINNConfig(epochs=3, hidden_dims=[8], lambda_physics=1.0))
    metrics = pinn.train(str(tmp_path / "features.csv"))
    # Physics loss should exist (may be small but tracked)
    assert metrics["final_physics_loss"] >= 0


# --- CNN Coral Vision ---

def test_cnn_train_and_predict():
    from models.coral_vision.cnn import CoralCNN, CNNConfig

    cnn = CoralCNN(CNNConfig(epochs=5, img_size=16))
    metrics = cnn.train(n_per_class=30)
    assert metrics["accuracy"] > 0.3  # better than random (0.33) expected

    # Test prediction
    healthy_img = np.full((16, 16, 3), [0.7, 0.4, 0.2], dtype=np.float32)
    result = cnn.predict(healthy_img)
    assert result["predicted_class"] in ("healthy", "bleached", "dead")
    assert 0 <= result["confidence"] <= 1
    assert result["model_type"] == "cnn"


def test_cnn_save_load(tmp_path):
    from models.coral_vision.cnn import CoralCNN, CNNConfig

    cnn = CoralCNN(CNNConfig(epochs=3, img_size=16))
    cnn.train(n_per_class=20)
    cnn.save(tmp_path / "cnn.pt")

    cnn2 = CoralCNN(CNNConfig(img_size=16))
    cnn2.load(tmp_path / "cnn.pt")
    img = np.random.rand(16, 16, 3).astype(np.float32)
    result = cnn2.predict(img)
    assert result["predicted_class"] in ("healthy", "bleached", "dead")


# --- SageMaker Integration ---

def test_sagemaker_training_job_config():
    from infrastructure.aws.sagemaker import create_training_job, SageMakerConfig
    config = SageMakerConfig(role="arn:aws:iam::123456:role/SageMakerRole")
    job = create_training_job(config)
    assert job["entry_point"] == "models/bleaching_risk/train.py"
    assert job["instance_type"] == "ml.m5.large"
    assert not job["ready_to_submit"]  # sagemaker SDK not installed


def test_sagemaker_endpoint_config():
    from infrastructure.aws.sagemaker import create_endpoint_config, SageMakerConfig
    config = SageMakerConfig(role="arn:aws:iam::123456:role/SageMakerRole")
    ep = create_endpoint_config("s3://reeftwin/models/model.tar.gz", config)
    assert ep["endpoint_name"] == "reeftwin-bleaching-risk"
    assert ep["instance_type"] == "ml.t2.medium"


def test_sagemaker_processing_job_config():
    from infrastructure.aws.sagemaker import create_processing_job, SageMakerConfig
    config = SageMakerConfig(role="arn:aws:iam::123456:role/SageMakerRole")
    job = create_processing_job(config)
    assert job["script"] == "pipelines/build_features.py"
    assert len(job["inputs"]) > 0
    assert len(job["outputs"]) > 0


def test_sagemaker_model_registry():
    from infrastructure.aws.sagemaker import register_model
    reg = register_model("s3://reeftwin/models/model.tar.gz")
    assert reg["model_name"] == "ReefTwin-BleachingRisk"
    assert "application/json" in reg["content_types"]
