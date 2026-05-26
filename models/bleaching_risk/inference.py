from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from infrastructure.logging import get_logger
from infrastructure.mlops.tracking import trace_inference
from infrastructure.settings import settings

logger = get_logger("models.bleaching_risk.inference")


def risk_category(score: float) -> str:
    t = settings.risk_thresholds
    if score >= t.alert:
        return "alert"
    if score >= t.warning:
        return "warning"
    if score >= t.watch:
        return "watch"
    return "normal"


def verify_model_integrity(model_path: Path) -> str:
    """Compute SHA256 hash of model file for audit trail."""
    import hashlib
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@trace_inference(name="bleaching_risk.predict")
def predict_risk(model_path: str | Path, row: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]
    X = pd.DataFrame([{k: row.get(k, 0) for k in features}])
    prob_matrix = model.predict_proba(X)
    classes = list(model.named_steps["clf"].classes_)

    if 1 in classes:
        score = float(prob_matrix[:, classes.index(1)][0])
    else:
        logger.warning(
            "Positive class (1) not found in model classes %s. "
            "Returning score 0.0 — model may have been trained on single-class data.",
            classes,
        )
        score = 0.0

    result = {"bleaching_risk_score": round(score, 4), "risk_category": risk_category(score)}
    logger.debug("Prediction: %s", result)
    return result
