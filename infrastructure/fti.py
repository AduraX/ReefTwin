"""Feature-Training-Inference (FTI) architecture.

Structures ReefTwin pipelines as three independent scaling units:
  - Feature pipeline: CPU-intensive, horizontal scale
  - Training pipeline: GPU-intensive, vertical scale
  - Inference pipeline: Request-based, horizontal scale

Each pipeline can be run independently and scaled separately.
Pattern from llm-twin-kf4x.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from infrastructure.logging import get_logger

logger = get_logger("fti")


@dataclass
class PipelineResult:
    pipeline: str
    status: str  # "success", "failed"
    duration_ms: float
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def run_feature_pipeline(
    iot_rows: int = 5000,
    noaa_days: int = 60,
) -> PipelineResult:
    """Feature Pipeline: data generation → ingestion → feature engineering.

    CPU-intensive, horizontally scalable. No GPU needed.
    """
    start = time.perf_counter()
    try:
        from pipelines.simulate_iot_stream import generate_readings
        from pipelines.ingest_noaa_crw import generate_noaa_sample
        from pipelines.build_features import build_features
        from infrastructure.settings import settings

        iot_path = settings.iot_output
        noaa_path = settings.noaa_output
        features_path = settings.features_path

        iot_path.parent.mkdir(parents=True, exist_ok=True)
        noaa_path.parent.mkdir(parents=True, exist_ok=True)
        features_path.parent.mkdir(parents=True, exist_ok=True)

        generate_readings(iot_rows).to_csv(iot_path, index=False)
        generate_noaa_sample(noaa_days).to_csv(noaa_path, index=False)
        df = build_features(str(iot_path), str(noaa_path))
        df.to_csv(features_path, index=False)

        duration = (time.perf_counter() - start) * 1000
        return PipelineResult(
            pipeline="feature",
            status="success",
            duration_ms=round(duration, 1),
            outputs={"feature_rows": len(df), "features_path": str(features_path)},
        )
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.error("Feature pipeline failed: %s", e)
        return PipelineResult(pipeline="feature", status="failed", duration_ms=round(duration, 1), error=str(e))


def run_training_pipeline() -> PipelineResult:
    """Training Pipeline: model training + evaluation.

    GPU-intensive (for PIML/deep models), vertically scalable.
    """
    start = time.perf_counter()
    try:
        from models.bleaching_risk.train import train_model
        from models.reef_dynamics.hybrid_predictor import train_hybrid_model
        from infrastructure.settings import settings

        rf_metrics = train_model(str(settings.features_path), str(settings.model_path))
        hybrid_path = settings.model_path.parent.parent / "reef_dynamics" / "hybrid_model.joblib"
        hybrid_metrics = train_hybrid_model(str(settings.features_path), str(hybrid_path))

        duration = (time.perf_counter() - start) * 1000
        return PipelineResult(
            pipeline="training",
            status="success",
            duration_ms=round(duration, 1),
            outputs={
                "rf_roc_auc": rf_metrics.get("roc_auc"),
                "hybrid_mse": hybrid_metrics.get("mse"),
                "model_path": str(settings.model_path),
            },
        )
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.error("Training pipeline failed: %s", e)
        return PipelineResult(pipeline="training", status="failed", duration_ms=round(duration, 1), error=str(e))


def run_inference_pipeline() -> PipelineResult:
    """Inference Pipeline: update twin state from latest features + model.

    Request-based, horizontally scalable. Serves the API.
    """
    start = time.perf_counter()
    try:
        from pipelines.update_twin_state import update_twin_state
        from infrastructure.settings import settings

        payload = update_twin_state(str(settings.features_path), str(settings.model_path))
        n_reefs = len(payload.get("states", []))

        duration = (time.perf_counter() - start) * 1000
        return PipelineResult(
            pipeline="inference",
            status="success",
            duration_ms=round(duration, 1),
            outputs={"reefs_updated": n_reefs},
        )
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.error("Inference pipeline failed: %s", e)
        return PipelineResult(pipeline="inference", status="failed", duration_ms=round(duration, 1), error=str(e))


def run_full_fti() -> list[PipelineResult]:
    """Run the full Feature → Training → Inference pipeline sequence."""
    results = []
    for runner in [run_feature_pipeline, run_training_pipeline, run_inference_pipeline]:
        result = runner()
        logger.info("FTI %s: %s (%.1fms)", result.pipeline, result.status, result.duration_ms)
        results.append(result)
        if result.status == "failed":
            logger.error("FTI halted at %s: %s", result.pipeline, result.error)
            break
    return results
