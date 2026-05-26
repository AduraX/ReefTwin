"""Uncertainty quantification via split conformal prediction.

Provides calibrated prediction intervals for any point predictor.
Given a desired confidence level (e.g., 90%), the intervals are
guaranteed to contain the true value at least that often on future data.

This addresses AIMS JD Responsibility 7: "enhancing predictive capability,
uncertainty quantification, and real-time responsiveness."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionWithUncertainty:
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    interval_width: float


class ConformalPredictor:
    """Split conformal prediction wrapper for regression models.

    Usage:
        cp = ConformalPredictor(confidence=0.90)
        cp.calibrate(y_cal_true, y_cal_pred)
        result = cp.predict(point_estimate=0.73)
    """

    def __init__(self, confidence: float = 0.90) -> None:
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        self.confidence = confidence
        self._quantile: float | None = None
        self._residuals: np.ndarray | None = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Calibrate using a held-out calibration set.

        Computes the nonconformity scores (absolute residuals) and finds
        the quantile corresponding to the desired confidence level.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if len(y_true) < 2:
            raise ValueError("Need at least 2 calibration samples")

        self._residuals = np.abs(y_true - y_pred)

        # Finite-sample correction: use ceil((n+1)*confidence)/n quantile
        n = len(self._residuals)
        adjusted_quantile = min(np.ceil((n + 1) * self.confidence) / n, 1.0)
        self._quantile = float(np.quantile(self._residuals, adjusted_quantile))

    @property
    def is_calibrated(self) -> bool:
        return self._quantile is not None

    @property
    def calibration_quantile(self) -> float:
        if self._quantile is None:
            raise RuntimeError("Call calibrate() first")
        return self._quantile

    def predict(self, point_estimate: float) -> PredictionWithUncertainty:
        """Wrap a point prediction with a calibrated interval."""
        if self._quantile is None:
            raise RuntimeError("Call calibrate() first")

        lower = max(0.0, point_estimate - self._quantile)
        upper = min(1.0, point_estimate + self._quantile)

        return PredictionWithUncertainty(
            point_estimate=round(point_estimate, 4),
            lower_bound=round(lower, 4),
            upper_bound=round(upper, 4),
            confidence_level=self.confidence,
            interval_width=round(upper - lower, 4),
        )

    def predict_batch(
        self, point_estimates: np.ndarray
    ) -> list[PredictionWithUncertainty]:
        return [self.predict(float(p)) for p in point_estimates]
