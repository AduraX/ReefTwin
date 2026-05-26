"""Tests for experiment runner."""

from infrastructure.mlops.experiments import (
    run_experiment_1_latency,
    run_experiment_2_cost,
    run_experiment_3_reliability,
    generate_report,
)


def test_experiment_1_latency():
    result = run_experiment_1_latency()
    assert result.name == "Experiment 1: Pipeline Latency"
    assert "p50_ms" in result.baseline
    assert "p50_ms" in result.optimized
    assert result.summary  # non-empty


def test_experiment_2_cost():
    result = run_experiment_2_cost()
    assert result.name == "Experiment 2: Inference Cost"
    assert result.optimized["invocations"] < result.baseline["invocations"]
    assert result.target_met  # cache should achieve >22% reduction


def test_experiment_3_reliability():
    result = run_experiment_3_reliability()
    assert result.name == "Experiment 3: Pipeline Reliability"
    assert result.optimized["rejected_to_dlq"] > 0
    assert result.target_met  # validation catches bad records


def test_generate_report():
    results = [run_experiment_2_cost(), run_experiment_3_reliability()]
    report = generate_report(results)
    assert "EXPERIMENT REPORT" in report
    assert "OVERALL" in report
