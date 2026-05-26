"""Before/after benchmark comparison tool.

Measures pipeline latency, model inference time, and reliability,
then compares results to quantify improvements for experiments 1-3.

Inspired by InferForge bench/compare.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from infrastructure.logging import get_logger

logger = get_logger("mlops.benchmark")


@dataclass
class BenchmarkResult:
    name: str
    timestamp: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        logger.info("Benchmark saved: %s → %s", self.name, path)

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkResult:
        data = json.loads(Path(path).read_text())
        return cls(**data)


@dataclass
class ComparisonResult:
    baseline_name: str
    optimized_name: str
    deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    summary: str = ""


def compare(baseline: BenchmarkResult, optimized: BenchmarkResult) -> ComparisonResult:
    """Compare two benchmark results and compute deltas."""
    deltas = {}
    all_keys = set(baseline.metrics) | set(optimized.metrics)

    for key in sorted(all_keys):
        base_val = baseline.metrics.get(key, 0.0)
        opt_val = optimized.metrics.get(key, 0.0)
        abs_delta = opt_val - base_val
        pct_delta = (abs_delta / base_val * 100) if base_val != 0 else 0.0

        deltas[key] = {
            "baseline": round(base_val, 4),
            "optimized": round(opt_val, 4),
            "delta": round(abs_delta, 4),
            "delta_pct": round(pct_delta, 2),
        }

    # Build summary
    lines = [f"Comparison: {baseline.name} → {optimized.name}", ""]
    for key, d in deltas.items():
        direction = "↓" if d["delta"] < 0 else "↑" if d["delta"] > 0 else "="
        lines.append(f"  {key}: {d['baseline']:.4f} → {d['optimized']:.4f} ({direction} {abs(d['delta_pct']):.1f}%)")

    return ComparisonResult(
        baseline_name=baseline.name,
        optimized_name=optimized.name,
        deltas=deltas,
        summary="\n".join(lines),
    )


def measure_latency(func: Callable, iterations: int = 10, warmup: int = 2, **kwargs) -> dict[str, float]:
    """Measure function latency over multiple iterations.

    Returns p50, p95, p99, mean, min, max in milliseconds.
    """
    import numpy as np

    # Warmup
    for _ in range(warmup):
        func(**kwargs)

    # Measure
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(**kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)

    arr = np.array(timings)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "iterations": iterations,
    }


def run_pipeline_benchmark(name: str = "pipeline_baseline") -> BenchmarkResult:
    """Run a full pipeline benchmark: data gen → features → train → predict."""
    from pipelines.simulate_iot_stream import generate_readings
    from pipelines.ingest_noaa_crw import generate_noaa_sample
    from pipelines.build_features import build_features
    from models.bleaching_risk.train import train_model
    from models.bleaching_risk.inference import predict_risk
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        metrics = {}

        # Data generation
        start = time.perf_counter()
        generate_readings(1000).to_csv(tmp / "iot.csv", index=False)
        generate_noaa_sample(30).to_csv(tmp / "noaa.csv", index=False)
        metrics["data_gen_ms"] = (time.perf_counter() - start) * 1000

        # Feature engineering
        start = time.perf_counter()
        features = build_features(str(tmp / "iot.csv"), str(tmp / "noaa.csv"))
        features.to_csv(tmp / "features.csv", index=False)
        metrics["feature_eng_ms"] = (time.perf_counter() - start) * 1000

        # Model training
        start = time.perf_counter()
        train_metrics = train_model(str(tmp / "features.csv"), str(tmp / "model.joblib"))
        metrics["training_ms"] = (time.perf_counter() - start) * 1000
        if train_metrics.get("roc_auc") is not None:
            metrics["roc_auc"] = train_metrics["roc_auc"]

        # Inference latency
        row = features.iloc[0].to_dict()
        inf_stats = measure_latency(
            predict_risk, iterations=20, warmup=3,
            model_path=str(tmp / "model.joblib"), row=row,
        )
        for k, v in inf_stats.items():
            metrics[f"inference_{k}"] = v

        # Total pipeline
        metrics["total_pipeline_ms"] = (
            metrics["data_gen_ms"] + metrics["feature_eng_ms"] + metrics["training_ms"]
        )

    return BenchmarkResult(name=name, metrics=metrics)
