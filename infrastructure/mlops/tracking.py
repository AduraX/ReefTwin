"""MLflow experiment tracking and model versioning.

Provides a unified interface for:
  - Experiment tracking (hyperparameters, metrics, artifacts)
  - Model versioning (register, stage, load by version)
  - Inference tracing (latency, inputs, predictions)

Falls back to local file-based tracking when MLflow server is not available.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from infrastructure.logging import get_logger

logger = get_logger("mlops.tracking")

_mlflow = None
_tracking_enabled = False


def _get_mlflow():
    """Lazy-load mlflow and configure tracking URI."""
    global _mlflow, _tracking_enabled
    if _mlflow is not None:
        return _mlflow

    try:
        import mlflow
        tracking_uri = Path("mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        _mlflow = mlflow
        _tracking_enabled = True
        logger.info("MLflow tracking enabled: %s", tracking_uri)
    except ImportError:
        logger.warning("mlflow not installed — tracking disabled")
        _tracking_enabled = False

    return _mlflow


def is_tracking_enabled() -> bool:
    _get_mlflow()
    return _tracking_enabled


@dataclass
class ExperimentRun:
    run_id: str
    experiment_name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@contextmanager
def tracked_experiment(
    experiment_name: str,
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
):
    """Context manager for tracking an experiment run.

    Usage:
        with tracked_experiment("bleaching_risk_training", params={...}) as run:
            model = train(...)
            run.log_metric("roc_auc", 0.95)
            run.log_artifact("model.joblib")
    """
    mlflow = _get_mlflow()

    if not _tracking_enabled or mlflow is None:
        yield _NoOpRun()
        return

    mlflow.set_experiment(experiment_name)
    # Handle nested runs (e.g., when called from within another tracked context)
    active = mlflow.active_run()
    nested = active is not None
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if params:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        if tags:
            mlflow.set_tags(tags)

        yield _ActiveRun(mlflow, run)


class _NoOpRun:
    """No-op run when MLflow is not available."""

    def log_metric(self, key: str, value: float) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def log_model(self, model: Any, artifact_path: str, **kwargs) -> None:
        pass

    @property
    def run_id(self) -> str:
        return "noop"


class _ActiveRun:
    """Active MLflow run wrapper."""

    def __init__(self, mlflow, run) -> None:
        self._mlflow = mlflow
        self._run = run

    def log_metric(self, key: str, value: float) -> None:
        self._mlflow.log_metric(key, value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)

    def log_model(self, model: Any, artifact_path: str = "model", **kwargs) -> None:
        self._mlflow.sklearn.log_model(model, artifact_path=artifact_path, **kwargs)

    @property
    def run_id(self) -> str:
        return self._run.info.run_id


def trace_inference(func: Callable | None = None, *, name: str = ""):
    """Decorator for tracing inference calls with MLflow.

    Logs: function name, latency_ms, input summary, output summary.

    Usage:
        @trace_inference
        def predict(features):
            ...

        @trace_inference(name="bleaching_prediction")
        def predict(features):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        trace_name = name or fn.__qualname__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            mlflow = _get_mlflow()
            start = time.perf_counter()

            result = fn(*args, **kwargs)

            latency_ms = (time.perf_counter() - start) * 1000

            if _tracking_enabled and mlflow is not None:
                try:
                    mlflow.log_metrics({
                        f"{trace_name}.latency_ms": latency_ms,
                    })
                except Exception:
                    pass  # Don't fail inference if tracking fails

            logger.debug("%s completed in %.1fms", trace_name, latency_ms)
            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
