"""Unified prediction interface with pluggable model strategies.

Inspired by GPURoute's strategy pattern — models are interchangeable behind
a common interface. Selection is via settings or runtime parameter.

Strategies:
    - "random_forest": Original scikit-learn RandomForest classifier
    - "physics_hybrid": PIML model (physics ODE + GBR residual correction)
    - "ensemble": Weighted average of RF + physics hybrid

Each strategy returns a PredictionResult with optional uncertainty intervals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from infrastructure.logging import get_logger
from infrastructure.settings import settings
from models.uncertainty import PredictionWithUncertainty

logger = get_logger("models.predictor")


@dataclass
class PredictionResult:
    bleaching_risk_score: float
    risk_category: str
    model_strategy: str
    physics_prior: float | None = None
    uncertainty: PredictionWithUncertainty | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BleachingPredictor(ABC):
    """Abstract interface for bleaching risk prediction strategies."""

    @property
    @abstractmethod
    def strategy_name(self) -> str: ...

    @abstractmethod
    def predict(self, row: dict[str, Any]) -> PredictionResult: ...


class RandomForestPredictor(BleachingPredictor):
    """Original RandomForest classifier strategy."""

    def __init__(self, model_path: Path | str) -> None:
        from models.bleaching_risk.inference import predict_risk
        self._model_path = Path(model_path)
        self._predict_fn = predict_risk

    @property
    def strategy_name(self) -> str:
        return "random_forest"

    def predict(self, row: dict[str, Any]) -> PredictionResult:
        result = self._predict_fn(self._model_path, row)
        return PredictionResult(
            bleaching_risk_score=result["bleaching_risk_score"],
            risk_category=result["risk_category"],
            model_strategy=self.strategy_name,
        )


class PhysicsHybridPredictor(BleachingPredictor):
    """PIML hybrid model: physics ODE + ML residual correction."""

    def __init__(self, model_path: Path | str) -> None:
        from models.reef_dynamics.hybrid_predictor import predict_hybrid
        self._model_path = Path(model_path)
        self._predict_fn = predict_hybrid

    @property
    def strategy_name(self) -> str:
        return "physics_hybrid"

    def predict(self, row: dict[str, Any]) -> PredictionResult:
        result = self._predict_fn(self._model_path, row)
        return PredictionResult(
            bleaching_risk_score=result["bleaching_risk_score"],
            risk_category=result["risk_category"],
            model_strategy=self.strategy_name,
            physics_prior=result.get("physics_prior"),
            metadata={"physics_stress": result.get("physics_stress")},
        )


class EnsemblePredictor(BleachingPredictor):
    """Weighted ensemble of multiple predictors."""

    def __init__(self, predictors: list[tuple[BleachingPredictor, float]]) -> None:
        total = sum(w for _, w in predictors)
        self._predictors = [(p, w / total) for p, w in predictors]

    @property
    def strategy_name(self) -> str:
        names = [p.strategy_name for p, _ in self._predictors]
        return f"ensemble({'+'.join(names)})"

    def predict(self, row: dict[str, Any]) -> PredictionResult:
        weighted_score = 0.0
        physics_prior = None
        for predictor, weight in self._predictors:
            result = predictor.predict(row)
            weighted_score += result.bleaching_risk_score * weight
            if result.physics_prior is not None:
                physics_prior = result.physics_prior

        score = round(float(np.clip(weighted_score, 0.0, 1.0)), 4)
        t = settings.risk_thresholds
        if score >= t.alert:
            category = "alert"
        elif score >= t.warning:
            category = "warning"
        elif score >= t.watch:
            category = "watch"
        else:
            category = "normal"

        return PredictionResult(
            bleaching_risk_score=score,
            risk_category=category,
            model_strategy=self.strategy_name,
            physics_prior=physics_prior,
        )


def get_predictor(
    strategy: str = "random_forest",
    rf_model_path: Path | str | None = None,
    hybrid_model_path: Path | str | None = None,
) -> BleachingPredictor:
    """Factory for creating a predictor with the specified strategy."""
    rf_path = Path(rf_model_path or settings.model_path)
    hybrid_path = Path(hybrid_model_path or settings.model_path.parent.parent / "reef_dynamics" / "hybrid_model.joblib")

    if strategy == "random_forest":
        return RandomForestPredictor(rf_path)

    elif strategy == "physics_hybrid":
        return PhysicsHybridPredictor(hybrid_path)

    elif strategy == "ensemble":
        predictors: list[tuple[BleachingPredictor, float]] = []
        if rf_path.exists():
            predictors.append((RandomForestPredictor(rf_path), 0.4))
        if hybrid_path.exists():
            predictors.append((PhysicsHybridPredictor(hybrid_path), 0.6))
        if not predictors:
            raise FileNotFoundError("No model files found for ensemble")
        return EnsemblePredictor(predictors)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Options: random_forest, physics_hybrid, ensemble")
