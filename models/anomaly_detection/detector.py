"""Anomaly detection for reef sensor data using Isolation Forest.

Detects anomalous sensor readings that may indicate:
  - Sensor malfunction (sudden spikes/drops)
  - Genuine environmental events (heat waves, pollution plumes)
  - Data quality issues (stuck sensors, transmission errors)

Produces an anomaly score per reading + binary anomaly flag.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from infrastructure.logging import get_logger
from infrastructure.settings import settings

logger = get_logger("models.anomaly_detection")

ANOMALY_FEATURES = [
    "water_temperature_c", "ph", "salinity_psu",
    "turbidity_ntu", "dissolved_oxygen_mg_l",
]


@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_score: float
    contributing_features: list[str]


def train_anomaly_detector(
    features_path: str,
    model_out: str = "models/anomaly_detection/isolation_forest.joblib",
    contamination: float = 0.05,
) -> dict[str, Any]:
    """Train an Isolation Forest on reef feature data."""
    from infrastructure.settings import read_df
    df = read_df(features_path)
    available = [f for f in ANOMALY_FEATURES if f in df.columns]
    X = df[available].ffill().fillna(df[available].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_scaled)

    scores = model.decision_function(X_scaled)
    n_anomalies = int((model.predict(X_scaled) == -1).sum())

    out = Path(model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "features": available,
        "contamination": contamination,
    }, out)

    metrics = {
        "n_samples": len(X),
        "n_anomalies": n_anomalies,
        "anomaly_rate": round(n_anomalies / len(X), 4),
        "score_mean": round(float(np.mean(scores)), 4),
        "score_std": round(float(np.std(scores)), 4),
    }
    logger.info("Anomaly detector trained: %d samples, %d anomalies (%.1f%%)", len(X), n_anomalies, n_anomalies / len(X) * 100)
    return metrics


def detect_anomaly(
    model_path: str | Path,
    reading: dict[str, float],
) -> AnomalyResult:
    """Detect if a single sensor reading is anomalous."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Anomaly model not found: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["features"]

    X = np.array([[reading.get(f, 0) for f in features]])
    X_scaled = scaler.transform(X)

    score = float(model.decision_function(X_scaled)[0])
    prediction = int(model.predict(X_scaled)[0])
    is_anomaly = prediction == -1

    # Identify contributing features (those furthest from mean)
    z_scores = np.abs(X_scaled[0])
    top_indices = np.argsort(z_scores)[::-1][:3]
    contributing = [features[i] for i in top_indices if z_scores[i] > 1.5]

    return AnomalyResult(
        is_anomaly=is_anomaly,
        anomaly_score=round(score, 4),
        contributing_features=contributing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomaly detector")
    parser.add_argument("--features", default=str(settings.features_path))
    parser.add_argument("--model-out", default="models/anomaly_detection/isolation_forest.joblib")
    args = parser.parse_args()
    train_anomaly_detector(args.features, args.model_out)


if __name__ == "__main__":
    main()
