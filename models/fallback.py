"""Resilient model invocation with fallback chains.

Tries predictors in priority order, falling back to the next
on failure. Logs fallback events for governance tracking.

Pattern from llm-twin-enhancements FallbackLLMCaller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from infrastructure.logging import get_logger
from models.predictor import BleachingPredictor, PredictionResult

logger = get_logger("models.fallback")


@dataclass
class FallbackResult:
    prediction: PredictionResult
    model_used: str
    was_fallback: bool
    attempts: int
    latency_ms: float
    errors: list[str] = field(default_factory=list)


class FallbackChain:
    """Tries multiple predictors in sequence, falls back on failure.

    Usage:
        chain = FallbackChain([rf_predictor, hybrid_predictor, heuristic])
        result = chain.predict(features)
    """

    def __init__(self, predictors: list[BleachingPredictor]) -> None:
        if not predictors:
            raise ValueError("Need at least one predictor")
        self._predictors = predictors
        self.total_calls = 0
        self.fallback_count = 0

    def predict(self, row: dict[str, Any]) -> FallbackResult:
        """Try each predictor in order until one succeeds."""
        self.total_calls += 1
        errors = []
        start = time.perf_counter()

        for i, predictor in enumerate(self._predictors):
            try:
                result = predictor.predict(row)
                latency = (time.perf_counter() - start) * 1000
                was_fallback = i > 0

                if was_fallback:
                    self.fallback_count += 1
                    logger.warning(
                        "Fallback to %s (attempt %d) after: %s",
                        predictor.strategy_name, i + 1, "; ".join(errors),
                    )

                return FallbackResult(
                    prediction=result,
                    model_used=predictor.strategy_name,
                    was_fallback=was_fallback,
                    attempts=i + 1,
                    latency_ms=round(latency, 2),
                    errors=errors,
                )
            except Exception as e:
                errors.append(f"{predictor.strategy_name}: {e}")
                logger.debug("Predictor %s failed: %s", predictor.strategy_name, e)

        # All predictors failed — return safe default
        latency = (time.perf_counter() - start) * 1000
        self.fallback_count += 1
        logger.error("All %d predictors failed: %s", len(self._predictors), errors)

        return FallbackResult(
            prediction=PredictionResult(
                bleaching_risk_score=0.0,
                risk_category="unknown",
                model_strategy="fallback_exhausted",
            ),
            model_used="none",
            was_fallback=True,
            attempts=len(self._predictors),
            latency_ms=round(latency, 2),
            errors=errors,
        )

    @property
    def fallback_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.fallback_count / self.total_calls


class HeuristicPredictor(BleachingPredictor):
    """Simple rule-based fallback when ML models are unavailable.

    Uses known NOAA thresholds directly — no trained model needed.
    """

    @property
    def strategy_name(self) -> str:
        return "heuristic"

    def predict(self, row: dict[str, Any]) -> PredictionResult:
        dhw = float(row.get("degree_heating_weeks", 0))
        temp = float(row.get("water_temperature_c", 28))
        hotspot = float(row.get("hotspot_c", 0))

        # NOAA-based heuristic
        risk = 0.0
        if dhw >= 8:
            risk = 0.9
        elif dhw >= 4:
            risk = 0.7
        elif hotspot >= 1:
            risk = 0.5
        elif temp >= 30:
            risk = 0.4

        from infrastructure.settings import settings
        t = settings.risk_thresholds
        if risk >= t.alert:
            category = "alert"
        elif risk >= t.warning:
            category = "warning"
        elif risk >= t.watch:
            category = "watch"
        else:
            category = "normal"

        return PredictionResult(
            bleaching_risk_score=round(risk, 4),
            risk_category=category,
            model_strategy="heuristic",
        )
