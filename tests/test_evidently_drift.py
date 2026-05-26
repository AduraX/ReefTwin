"""Tests for Evidently AI drift monitoring integration."""

import numpy as np
import pandas as pd

from infrastructure.mlops.evidently_drift import (
    run_data_drift_report,
    run_data_quality_report,
    run_drift_test_suite,
)


def _make_reef_df(n: int, temp_mean: float = 28.3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "water_temperature_c": rng.normal(temp_mean, 0.5, n),
        "ph": rng.normal(8.1, 0.05, n),
        "salinity_psu": rng.normal(35.1, 0.3, n),
        "turbidity_ntu": rng.normal(0.8, 0.2, n),
        "dissolved_oxygen_mg_l": rng.normal(6.5, 0.4, n),
    })


def test_no_drift_detected():
    ref = _make_reef_df(200, seed=1)
    cur = _make_reef_df(200, seed=2)  # same distribution, different samples
    result = run_data_drift_report(ref, cur)
    # Same distribution should have low drift share
    assert result.drift_share < 0.5
    assert result.n_total_features == 5


def test_drift_detected_with_shift():
    ref = _make_reef_df(200, temp_mean=28.3, seed=1)
    # Significant temperature shift (heat wave)
    cur = _make_reef_df(200, temp_mean=32.0, seed=2)
    result = run_data_drift_report(ref, cur, feature_columns=["water_temperature_c"])
    assert result.is_drifted
    assert result.n_drifted_features >= 1
    assert "water_temperature_c" in result.feature_details


def test_drift_report_html_output(tmp_path):
    ref = _make_reef_df(100, seed=1)
    cur = _make_reef_df(100, temp_mean=31.0, seed=2)
    output = tmp_path / "drift_report.html"
    result = run_data_drift_report(ref, cur, output_path=output)
    assert output.exists()
    assert result.report_path == str(output)
    # HTML file should have some content
    assert len(output.read_text()) > 100


def test_data_quality_report():
    df = _make_reef_df(100)
    result = run_data_quality_report(df)
    assert "metrics" in result


def test_drift_test_suite():
    ref = _make_reef_df(200, seed=1)
    cur = _make_reef_df(200, seed=2)
    result = run_drift_test_suite(ref, cur)
    assert "overall_status" in result
    assert result["tests_total"] > 0
    assert result["overall_status"] in ("pass", "fail")


def test_drift_test_suite_with_shift():
    ref = _make_reef_df(200, temp_mean=28.0, seed=1)
    cur = _make_reef_df(200, temp_mean=35.0, seed=2)  # extreme shift
    result = run_drift_test_suite(ref, cur, feature_columns=["water_temperature_c"])
    assert result["tests_failed"] >= 1
    assert result["overall_status"] == "fail"
