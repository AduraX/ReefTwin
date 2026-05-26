"""AWS SageMaker integration for ReefTwin.

Provides wrappers for:
    - SageMaker Training Jobs (bleaching risk + PIML hybrid)
    - SageMaker Endpoints (real-time inference)
    - SageMaker Model Registry (version management)
    - SageMaker Processing Jobs (feature engineering)

Requires: pip install 'sagemaker>=2.200'
All functions are no-op importable without sagemaker installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infrastructure.logging import get_logger

logger = get_logger("aws.sagemaker")


@dataclass
class SageMakerConfig:
    role: str = ""  # IAM role ARN
    region: str = "ap-southeast-2"  # Sydney — closest to GBR
    instance_type_training: str = "ml.m5.large"
    instance_type_inference: str = "ml.t2.medium"
    s3_bucket: str = "reeftwin"
    s3_prefix: str = "sagemaker"
    framework_version: str = "1.5.0"  # scikit-learn version on SageMaker


def create_training_job(
    config: SageMakerConfig | None = None,
    entry_point: str = "models/bleaching_risk/train.py",
    hyperparameters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create a SageMaker Training Job for model training.

    Uses the SageMaker SKLearn estimator for scikit-learn models.
    Returns job configuration (submittable with sagemaker SDK).
    """
    config = config or SageMakerConfig()

    job_config = {
        "entry_point": entry_point,
        "role": config.role,
        "instance_type": config.instance_type_training,
        "instance_count": 1,
        "framework_version": config.framework_version,
        "py_version": "py3",
        "output_path": f"s3://{config.s3_bucket}/{config.s3_prefix}/models",
        "hyperparameters": hyperparameters or {},
        "tags": [
            {"Key": "Project", "Value": "ReefTwin"},
            {"Key": "Component", "Value": "training"},
        ],
    }

    logger.info("SageMaker training job config: %s → %s", entry_point, config.instance_type_training)

    try:
        from sagemaker.sklearn import SKLearn
        estimator = SKLearn(
            entry_point=job_config["entry_point"],
            role=job_config["role"],
            instance_type=job_config["instance_type"],
            instance_count=1,
            framework_version=job_config["framework_version"],
            py_version="py3",
            output_path=job_config["output_path"],
            hyperparameters=job_config["hyperparameters"],
            tags=job_config["tags"],
        )
        job_config["estimator"] = estimator
        job_config["ready_to_submit"] = True
    except ImportError:
        logger.info("sagemaker SDK not installed — returning config only (dry run)")
        job_config["ready_to_submit"] = False

    return job_config


def create_endpoint_config(
    model_data: str,
    config: SageMakerConfig | None = None,
    endpoint_name: str = "reeftwin-bleaching-risk",
) -> dict[str, Any]:
    """Create a SageMaker real-time inference endpoint configuration.

    Args:
        model_data: S3 URI to the model artifact (model.tar.gz).
        config: SageMaker configuration.
        endpoint_name: Name for the endpoint.
    """
    config = config or SageMakerConfig()

    endpoint_config = {
        "endpoint_name": endpoint_name,
        "model_data": model_data,
        "role": config.role,
        "instance_type": config.instance_type_inference,
        "instance_count": 1,
        "framework_version": config.framework_version,
        "tags": [
            {"Key": "Project", "Value": "ReefTwin"},
            {"Key": "Component", "Value": "inference"},
        ],
    }

    logger.info("SageMaker endpoint config: %s (%s)", endpoint_name, config.instance_type_inference)

    try:
        from sagemaker.sklearn import SKLearnModel
        model = SKLearnModel(
            model_data=model_data,
            role=config.role,
            framework_version=config.framework_version,
            entry_point="models/bleaching_risk/inference.py",
        )
        endpoint_config["model"] = model
        endpoint_config["ready_to_deploy"] = True
    except ImportError:
        logger.info("sagemaker SDK not installed — returning config only (dry run)")
        endpoint_config["ready_to_deploy"] = False

    return endpoint_config


def create_processing_job(
    config: SageMakerConfig | None = None,
    script: str = "pipelines/build_features.py",
) -> dict[str, Any]:
    """Create a SageMaker Processing Job for feature engineering.

    Uses SKLearnProcessor for running data processing scripts.
    """
    config = config or SageMakerConfig()

    job_config = {
        "script": script,
        "role": config.role,
        "instance_type": config.instance_type_training,
        "instance_count": 1,
        "framework_version": config.framework_version,
        "inputs": [
            {"source": f"s3://{config.s3_bucket}/bronze/", "destination": "/opt/ml/processing/input/bronze"},
        ],
        "outputs": [
            {"source": "/opt/ml/processing/output/gold", "destination": f"s3://{config.s3_bucket}/gold/"},
        ],
        "tags": [
            {"Key": "Project", "Value": "ReefTwin"},
            {"Key": "Component", "Value": "processing"},
        ],
    }

    logger.info("SageMaker processing job config: %s", script)

    try:
        from sagemaker.sklearn.processing import SKLearnProcessor
        processor = SKLearnProcessor(
            framework_version=config.framework_version,
            role=config.role,
            instance_type=config.instance_type_training,
            instance_count=1,
        )
        job_config["processor"] = processor
        job_config["ready_to_submit"] = True
    except ImportError:
        logger.info("sagemaker SDK not installed — returning config only (dry run)")
        job_config["ready_to_submit"] = False

    return job_config


def register_model(
    model_data: str,
    model_name: str = "ReefTwin-BleachingRisk",
    config: SageMakerConfig | None = None,
) -> dict[str, Any]:
    """Register a model in SageMaker Model Registry."""
    config = config or SageMakerConfig()

    registry_config = {
        "model_name": model_name,
        "model_data": model_data,
        "content_types": ["application/json"],
        "response_types": ["application/json"],
        "description": "ReefTwin bleaching risk prediction model",
    }

    logger.info("SageMaker model registry: %s", model_name)
    return registry_config
