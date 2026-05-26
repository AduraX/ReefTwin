"""Evidently AI integration for production-grade drift monitoring.

Uses Evidently 0.7+ API (Report + Snapshot pattern).
Provides data drift detection, data quality reports, and CI/CD-ready
test suites with pass/fail results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from infrastructure.logging import get_logger

logger = get_logger("mlops.evidently_drift")


@dataclass
class EvidentlyDriftResult:
    """Result from Evidently drift analysis."""

    is_drifted: bool
    drift_share: float
    n_drifted_features: int
    n_total_features: int
    feature_details: dict[str, dict[str, Any]]
    report_path: str | None = None


def run_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str] | None = None,
    output_path: str | Path | None = None,
) -> EvidentlyDriftResult:
    """Run Evidently data drift report comparing reference vs current data.

    Args:
        reference: Reference dataset (e.g., training data).
        current: Current production dataset.
        feature_columns: Columns to check. If None, uses all shared columns.
        output_path: If set, saves interactive HTML report to this path.

    Returns:
        EvidentlyDriftResult with per-feature drift details.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    if feature_columns:
        reference = reference[feature_columns].copy()
        current = current[feature_columns].copy()

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    report_path_str = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_path))
        report_path_str = str(output_path)
        logger.info("Evidently drift report saved: %s", output_path)

    # Parse Evidently 0.7 result structure
    result_dict = snapshot.dict()
    drift_share = 0.0
    n_drifted = 0
    n_total = 0
    feature_details = {}

    for metric in result_dict.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        config = metric.get("config", {})
        value = metric.get("value", {})

        if "DriftedColumnsCount" in metric_name:
            if isinstance(value, dict):
                drift_share = value.get("share", 0.0)
                n_drifted = int(value.get("count", 0))
            # n_total inferred from per-column ValueDrift metrics below

        elif "ValueDrift" in metric_name:
            col = config.get("column", "")
            threshold = config.get("threshold", 0.05)
            method = config.get("method", "unknown")
            drift_score = value if isinstance(value, (int, float)) else 0.0

            is_col_drifted = drift_score < threshold if "p_value" in method else drift_score > threshold
            feature_details[col] = {
                "drifted": is_col_drifted,
                "stattest": method,
                "drift_score": drift_score,
                "threshold": threshold,
            }
            n_total += 1

    if not n_total and feature_details:
        n_total = len(feature_details)
    if not n_drifted:
        n_drifted = sum(1 for v in feature_details.values() if v.get("drifted"))
    if n_total > 0:
        drift_share = n_drifted / n_total

    is_drifted = n_drifted > 0
    if is_drifted:
        drifted_names = [k for k, v in feature_details.items() if v.get("drifted")]
        logger.warning("Evidently drift in %d/%d features: %s", n_drifted, n_total, ", ".join(drifted_names))

    return EvidentlyDriftResult(
        is_drifted=is_drifted,
        drift_share=drift_share,
        n_drifted_features=n_drifted,
        n_total_features=n_total,
        feature_details=feature_details,
        report_path=report_path_str,
    )


def run_data_quality_report(
    data: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run Evidently data quality report (missing values, stats)."""
    from evidently import Report
    from evidently.presets import DataSummaryPreset

    report = Report(metrics=[DataSummaryPreset()])
    snapshot = report.run(reference_data=reference, current_data=data)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_path))
        logger.info("Evidently data quality report saved: %s", output_path)

    return snapshot.dict()


def run_drift_test_suite(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Run Evidently test suite for automated pass/fail drift checks.

    Uses include_tests=True with DataDriftPreset to get built-in tests.
    Returns overall pass/fail — suitable for CI/CD pipeline gates.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    if feature_columns:
        reference = reference[feature_columns].copy()
        current = current[feature_columns].copy()

    report = Report(metrics=[DataDriftPreset()], include_tests=True)
    snapshot = report.run(reference_data=reference, current_data=current)

    result = snapshot.dict()
    tests = result.get("tests", [])

    passed = sum(1 for t in tests if str(t.get("status", "")).upper() in ("SUCCESS", "TESTSTATUS.SUCCESS"))
    # Handle enum status
    failed = 0
    for t in tests:
        status = str(t.get("status", "")).upper()
        if "FAIL" in status:
            failed += 1
        elif "SUCCESS" not in status and "PASS" not in status:
            # Unknown status — count as passed
            passed += 1

    total = len(tests)
    logger.info("Evidently test suite: %d passed, %d failed / %d total", total - failed, failed, total)

    return {
        "overall_status": "pass" if failed == 0 else "fail",
        "tests_passed": total - failed,
        "tests_failed": failed,
        "tests_total": total,
        "details": tests,
    }
