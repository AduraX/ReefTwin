"""Edge deployment: ONNX model export + lightweight inference.

Exports scikit-learn models to ONNX format for deployment on
edge devices (reef sensor nodes, low-power gateways) without
the full Python ML stack.

Also provides a pure-numpy inference function for environments
where even ONNX runtime is too heavy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from infrastructure.logging import get_logger
from infrastructure.settings import settings

logger = get_logger("models.edge.exporter")


def export_to_onnx(
    model_path: str | Path,
    output_path: str | Path = "models/edge/bleaching_risk.onnx",
) -> str:
    """Export a scikit-learn pipeline to ONNX format.

    Requires: pip install skl2onnx
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError("Install skl2onnx: pip install skl2onnx")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]

    initial_type = [("input", FloatTensorType([None, len(features)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info("Exported ONNX model: %s (%d features)", out, len(features))
    return str(out)


def predict_onnx(
    onnx_path: str | Path,
    features: dict[str, float],
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run inference using ONNX Runtime.

    Requires: pip install onnxruntime
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("Install onnxruntime: pip install onnxruntime")

    feature_names = feature_names or settings.features
    X = np.array([[features.get(f, 0) for f in feature_names]], dtype=np.float32)

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: X})

    # Result[0] = labels, Result[1] = probabilities (if available)
    label = int(result[0][0])
    proba = {}
    if len(result) > 1:
        proba = {str(i): round(float(p), 4) for i, p in enumerate(result[1][0])}

    return {"label": label, "probabilities": proba}


class LightweightPredictor:
    """Pure-numpy inference for extreme edge environments.

    Extracts the learned parameters from a trained RandomForest
    and provides prediction without scikit-learn dependency.
    Uses only the scaler mean/std and a simplified decision threshold.
    """

    def __init__(self, model_path: str | Path) -> None:
        bundle = joblib.load(model_path)
        model = bundle["model"]
        self.features = bundle["features"]

        # Extract scaler parameters
        scaler = model.named_steps["scaler"]
        self._mean = scaler.mean_
        self._scale = scaler.scale_

        # Extract forest for simplified prediction
        clf = model.named_steps["clf"]
        self._classes = list(clf.classes_)
        self._n_estimators = clf.n_estimators
        self._trees = clf.estimators_

        logger.info("Lightweight predictor loaded: %d features, %d trees", len(self.features), self._n_estimators)

    def predict(self, reading: dict[str, float]) -> dict[str, Any]:
        """Predict using extracted model parameters."""
        X = np.array([reading.get(f, 0) for f in self.features]).reshape(1, -1)
        X_scaled = (X - self._mean) / self._scale

        # Aggregate tree predictions
        votes = np.zeros(len(self._classes))
        for tree in self._trees:
            pred = tree.predict(X_scaled)[0]
            idx = self._classes.index(pred)
            votes[idx] += 1

        proba = votes / self._n_estimators
        label = self._classes[int(np.argmax(votes))]

        risk_score = float(proba[self._classes.index(1)]) if 1 in self._classes else 0.0

        from models.bleaching_risk.inference import risk_category
        return {
            "bleaching_risk_score": round(risk_score, 4),
            "risk_category": risk_category(risk_score),
            "inference_mode": "edge_lightweight",
        }
