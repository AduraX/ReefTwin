import numpy as np
import pytest

from models.uncertainty import ConformalPredictor


def test_calibrate_and_predict():
    cp = ConformalPredictor(confidence=0.90)
    y_true = np.array([0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.2, 0.9, 0.55, 0.45])
    y_pred = np.array([0.35, 0.48, 0.65, 0.42, 0.58, 0.75, 0.25, 0.85, 0.5, 0.4])
    cp.calibrate(y_true, y_pred)

    assert cp.is_calibrated
    assert cp.calibration_quantile > 0

    result = cp.predict(0.6)
    assert result.lower_bound <= 0.6 <= result.upper_bound
    assert result.confidence_level == 0.90


def test_interval_bounds_clipped():
    cp = ConformalPredictor(confidence=0.99)
    y_true = np.array([0.0, 1.0, 0.5])
    y_pred = np.array([0.5, 0.5, 0.5])
    cp.calibrate(y_true, y_pred)

    result = cp.predict(0.05)
    assert result.lower_bound >= 0.0

    result = cp.predict(0.95)
    assert result.upper_bound <= 1.0


def test_predict_before_calibrate():
    cp = ConformalPredictor()
    with pytest.raises(RuntimeError, match="calibrate"):
        cp.predict(0.5)


def test_batch_predict():
    cp = ConformalPredictor(confidence=0.80)
    cp.calibrate(np.array([0.5, 0.6, 0.7]), np.array([0.55, 0.58, 0.72]))
    results = cp.predict_batch(np.array([0.3, 0.5, 0.8]))
    assert len(results) == 3
    assert all(r.confidence_level == 0.80 for r in results)


def test_invalid_confidence():
    with pytest.raises(ValueError):
        ConformalPredictor(confidence=1.5)
