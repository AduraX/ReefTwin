"""Time-series forecasting for reef environmental parameters.

Pluggable backends:
    - holtwinters: Holt-Winters Exponential Smoothing (statsmodels, default)
    - sarima:      SARIMA (statsmodels, seasonal autoregressive)
    - prophet:     Prophet (Meta, additive/multiplicative seasonality)

Selection via REEFTWIN_FORECAST_BACKEND env var.
All backends produce point forecasts + prediction intervals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from infrastructure.logging import get_logger

logger = get_logger("models.forecasting")


@dataclass
class ForecastResult:
    parameter: str
    reef_id: str
    horizon_days: int
    forecast_values: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    last_observed: float
    trend: str  # "rising", "falling", "stable"
    backend: str = ""


def _detect_trend(values: np.ndarray, window: int = 7) -> str:
    if len(values) < 3:
        return "stable"
    recent = values[-min(window, len(values)):]
    slope = float(np.polyfit(range(len(recent)), recent, 1)[0])
    if slope > 0.05:
        return "rising"
    elif slope < -0.05:
        return "falling"
    return "stable"


# ---------------------------------------------------------------------------
# Forecasting Backends
# ---------------------------------------------------------------------------

class ForecastBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def forecast(self, series: np.ndarray, horizon: int, confidence: float) -> dict[str, Any]: ...


class HoltWintersBackend(ForecastBackend):
    """Holt-Winters Exponential Smoothing (statsmodels)."""

    @property
    def name(self) -> str:
        return "holtwinters"

    def forecast(self, series: np.ndarray, horizon: int, confidence: float = 0.95) -> dict[str, Any]:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        if len(series) < 4:
            last = float(series[-1]) if len(series) > 0 else 0.0
            return {"forecast": [last] * horizon, "lower": [last] * horizon, "upper": [last] * horizon}

        try:
            model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True, use_brute=False)
            fcast = fit.forecast(horizon)
            residuals = series - fit.fittedvalues
            std = float(np.std(residuals))
            from scipy.stats import norm
            z = norm.ppf(1 - (1 - confidence) / 2)
            margin = z * std * np.sqrt(np.arange(1, horizon + 1))
            return {"forecast": list(fcast), "lower": list(fcast - margin), "upper": list(fcast + margin)}
        except Exception:
            return self._linear_fallback(series, horizon)

    @staticmethod
    def _linear_fallback(series: np.ndarray, horizon: int) -> dict[str, Any]:
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        future_x = np.arange(len(series), len(series) + horizon)
        fcast = slope * future_x + intercept
        std = float(np.std(series - (slope * x + intercept)))
        return {"forecast": list(fcast), "lower": list(fcast - 1.96 * std), "upper": list(fcast + 1.96 * std)}


class SARIMABackend(ForecastBackend):
    """SARIMA — Seasonal ARIMA (statsmodels).

    Uses auto order selection with fallback to (1,1,1) if optimization fails.
    """

    @property
    def name(self) -> str:
        return "sarima"

    def forecast(self, series: np.ndarray, horizon: int, confidence: float = 0.95) -> dict[str, Any]:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if len(series) < 6:
            last = float(series[-1]) if len(series) > 0 else 0.0
            return {"forecast": [last] * horizon, "lower": [last] * horizon, "upper": [last] * horizon}

        # Try ARIMA(1,1,1) — simple and robust
        try:
            model = SARIMAX(series, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=50)
            pred = fit.get_forecast(steps=horizon)
            fcast = pred.predicted_mean
            ci = pred.conf_int(alpha=1 - confidence)
            return {"forecast": list(fcast), "lower": list(ci.iloc[:, 0]), "upper": list(ci.iloc[:, 1])}
        except Exception as e:
            logger.warning("SARIMA failed (%s), falling back to Holt-Winters", e)
            return HoltWintersBackend().forecast(series, horizon, confidence)


class ProphetBackend(ForecastBackend):
    """Meta Prophet — additive trend + seasonality.

    Requires: pip install prophet
    """

    @property
    def name(self) -> str:
        return "prophet"

    def forecast(self, series: np.ndarray, horizon: int, confidence: float = 0.95) -> dict[str, Any]:
        try:
            from prophet import Prophet
        except ImportError:
            logger.warning("prophet not installed, falling back to Holt-Winters")
            return HoltWintersBackend().forecast(series, horizon, confidence)

        if len(series) < 4:
            last = float(series[-1]) if len(series) > 0 else 0.0
            return {"forecast": [last] * horizon, "lower": [last] * horizon, "upper": [last] * horizon}

        try:
            df = pd.DataFrame({
                "ds": pd.date_range(end=pd.Timestamp.now(), periods=len(series), freq="D"),
                "y": series,
            })

            model = Prophet(
                interval_width=confidence,
                daily_seasonality=False,
                weekly_seasonality=len(series) >= 14,
                yearly_seasonality=False,
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=horizon)
            pred = model.predict(future)
            pred_future = pred.tail(horizon)

            return {
                "forecast": pred_future["yhat"].tolist(),
                "lower": pred_future["yhat_lower"].tolist(),
                "upper": pred_future["yhat_upper"].tolist(),
            }
        except Exception as e:
            logger.warning("Prophet failed (%s), falling back to Holt-Winters", e)
            return HoltWintersBackend().forecast(series, horizon, confidence)


# ---------------------------------------------------------------------------
# Factory + Public API
# ---------------------------------------------------------------------------

def get_forecast_backend(backend: str | None = None) -> ForecastBackend:
    """Create a forecasting backend by name."""
    import os
    backend = backend or os.getenv("REEFTWIN_FORECAST_BACKEND", "holtwinters")

    if backend == "holtwinters":
        return HoltWintersBackend()
    elif backend == "sarima":
        return SARIMABackend()
    elif backend == "prophet":
        return ProphetBackend()
    else:
        raise ValueError(f"Unknown forecast backend: {backend!r}. Options: holtwinters, sarima, prophet")


def forecast_parameter(
    series: pd.Series | np.ndarray,
    horizon: int = 7,
    confidence: float = 0.95,
    backend: str | None = None,
) -> dict[str, Any]:
    """Forecast a single time series using the configured backend."""
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    fb = get_forecast_backend(backend)

    result = fb.forecast(values, horizon, confidence)
    result["trend"] = _detect_trend(values)
    result["backend"] = fb.name

    # Round all values
    for key in ("forecast", "lower", "upper"):
        result[key] = [round(float(v), 4) for v in result[key]]

    return result


def forecast_reef(
    features_path: str,
    reef_id: str,
    horizon_days: int = 7,
    parameters: list[str] | None = None,
    backend: str | None = None,
) -> list[ForecastResult]:
    """Forecast multiple parameters for a specific reef."""
    from infrastructure.settings import read_df
    df = read_df(features_path)
    reef_data = df[df["reef_id"] == reef_id].sort_values("date")

    if reef_data.empty:
        logger.warning("No data for reef %s", reef_id)
        return []

    params = parameters or ["water_temperature_c", "degree_heating_weeks", "ph"]
    results = []

    for param in params:
        if param not in reef_data.columns:
            continue
        series = reef_data[param].dropna()
        if series.empty:
            continue

        fcast = forecast_parameter(series, horizon=horizon_days, backend=backend)
        results.append(ForecastResult(
            parameter=param,
            reef_id=reef_id,
            horizon_days=horizon_days,
            forecast_values=fcast["forecast"],
            lower_bound=fcast["lower"],
            upper_bound=fcast["upper"],
            last_observed=round(float(series.iloc[-1]), 4),
            trend=fcast["trend"],
            backend=fcast.get("backend", ""),
        ))

    logger.info("Forecast %d parameters for %s (%d days, backend=%s)", len(results), reef_id, horizon_days, backend or "default")
    return results
