"""Tests for MLOps components: tracking, benchmarks, drift, governance."""

import json
import numpy as np
import pytest

from infrastructure.mlops.tracking import (
    tracked_experiment, trace_inference,
)
from infrastructure.mlops.benchmark import (
    BenchmarkResult, compare, measure_latency, run_pipeline_benchmark,
)
from infrastructure.mlops.drift import DriftMonitor, _compute_psi
from infrastructure.mlops.governance import (
    ModelCard, AuditTrail, AuditEntry, create_bleaching_model_card, build_reeftwin_lineage,
)


# --- Tracking ---

def test_tracked_experiment_context():
    with tracked_experiment("test_experiment", params={"lr": 0.01}) as run:
        run.log_metric("accuracy", 0.95)
        assert run.run_id  # Should have a run ID (real or noop)


def test_trace_inference_decorator():
    @trace_inference(name="test_fn")
    def dummy_fn(x):
        return x * 2

    result = dummy_fn(5)
    assert result == 10


def test_trace_inference_no_args():
    @trace_inference
    def dummy_fn(x):
        return x + 1

    assert dummy_fn(3) == 4


# --- Benchmarks ---

def test_benchmark_result_save_load(tmp_path):
    result = BenchmarkResult(
        name="test_bench",
        metrics={"latency_ms": 42.5, "throughput": 100},
    )
    path = tmp_path / "bench.json"
    result.save(path)

    loaded = BenchmarkResult.load(path)
    assert loaded.name == "test_bench"
    assert loaded.metrics["latency_ms"] == 42.5


def test_benchmark_compare():
    baseline = BenchmarkResult(name="baseline", metrics={"latency_ms": 100, "accuracy": 0.90})
    optimized = BenchmarkResult(name="optimized", metrics={"latency_ms": 65, "accuracy": 0.93})
    result = compare(baseline, optimized)

    assert result.deltas["latency_ms"]["delta"] == -35
    assert result.deltas["latency_ms"]["delta_pct"] == -35.0
    assert result.deltas["accuracy"]["delta"] == pytest.approx(0.03, abs=0.001)
    assert "→" in result.summary


def test_measure_latency():
    def fast_fn():
        return sum(range(100))

    stats = measure_latency(fast_fn, iterations=5, warmup=1)
    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert "mean_ms" in stats
    assert stats["iterations"] == 5
    assert stats["p50_ms"] >= 0


def test_run_pipeline_benchmark():
    result = run_pipeline_benchmark("test_pipeline")
    assert result.name == "test_pipeline"
    assert "data_gen_ms" in result.metrics
    assert "feature_eng_ms" in result.metrics
    assert "training_ms" in result.metrics
    assert "inference_p50_ms" in result.metrics
    assert "total_pipeline_ms" in result.metrics


# --- Drift ---

def test_psi_identical_distributions():
    a = np.random.normal(0, 1, 1000)
    psi = _compute_psi(a, a)
    assert psi < 0.05


def test_psi_shifted_distribution():
    ref = np.random.normal(0, 1, 1000)
    shifted = np.random.normal(2, 1, 1000)
    psi = _compute_psi(ref, shifted)
    assert psi > 0.1  # Should detect the shift


def test_drift_monitor_no_drift():
    rng = np.random.default_rng(42)
    monitor = DriftMonitor()
    ref_data = rng.normal(28, 0.5, 500)
    monitor.set_reference({"temp": ref_data})

    # Use same distribution (different samples) — should not drift
    current = {"temp": rng.normal(28, 0.5, 500)}
    summary = monitor.check(current)
    assert summary.overall_status == "healthy"


def test_drift_monitor_with_drift():
    monitor = DriftMonitor(psi_warning=0.1)
    ref = {"temp": np.random.normal(28, 0.5, 500)}
    monitor.set_reference(ref)

    # Significant shift
    current = {"temp": np.random.normal(32, 1.5, 500)}
    summary = monitor.check(current)
    assert summary.drifted_features > 0
    assert summary.overall_status in ("warning", "critical")


def test_drift_monitor_multiple_features():
    monitor = DriftMonitor()
    ref = {
        "temp": np.random.normal(28, 0.5, 200),
        "ph": np.random.normal(8.1, 0.05, 200),
    }
    monitor.set_reference(ref)
    summary = monitor.check(ref)  # Same data — no drift
    assert summary.total_features == 2


# --- Governance: Model Card ---

def test_model_card_save_load(tmp_path):
    card = ModelCard(
        model_name="test_model",
        version="1.0",
        model_type="RandomForest",
        description="Test model",
        metrics={"accuracy": 0.95},
    )
    path = tmp_path / "card.json"
    card.save(path)

    loaded = ModelCard.load(path)
    assert loaded.model_name == "test_model"
    assert loaded.metrics["accuracy"] == 0.95


def test_model_card_to_markdown():
    card = create_bleaching_model_card({"roc_auc": 0.92})
    md = card.to_markdown()
    assert "Bleaching Risk Model" in md
    assert "roc_auc" in md
    assert "Limitations" in md
    assert "Bias Risks" in md


# --- Governance: Audit Trail ---

def test_audit_trail(tmp_path):
    trail = AuditTrail(path=tmp_path / "audit.jsonl")
    entry = AuditEntry(
        timestamp="2026-05-07T00:00:00Z",
        reef_id="gbr_heron_reef",
        model_name="bleaching_risk",
        model_version="0.1.0",
        prediction={"risk_score": 0.75, "category": "warning"},
        input_features={"temp": 30.5, "dhw": 6.2},
        latency_ms=12.3,
    )
    trail.log(entry)
    trail.log(entry)

    assert trail.count == 2
    entries = trail.read_all()
    assert len(entries) == 2
    assert entries[0].reef_id == "gbr_heron_reef"


# --- Governance: Data Lineage ---

def test_data_lineage():
    lineage = build_reeftwin_lineage()
    ancestors = lineage.get_lineage("predictions")
    assert "bleaching_model" in ancestors
    assert "reef_features" in ancestors
    assert "iot_sensors" in ancestors


def test_lineage_save(tmp_path):
    lineage = build_reeftwin_lineage()
    path = tmp_path / "lineage.json"
    lineage.save(path)
    data = json.loads(path.read_text())
    assert "iot_sensors" in data
    assert "predictions" in data
