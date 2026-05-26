"""Prediction and feature drift monitoring.

Detects when model predictions or input feature distributions shift
significantly from a reference baseline. Triggers alerts when drift
exceeds configurable thresholds.

Drift detection methods:
  - Population Stability Index (PSI) for distribution shifts
  - Mean/std monitoring for continuous features
  - Prediction distribution shift tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from infrastructure.logging import get_logger

logger = get_logger("mlops.drift")


@dataclass
class DriftReport:
    feature_name: str
    psi: float
    mean_shift: float
    std_shift: float
    is_drifted: bool
    severity: str  # "none", "warning", "critical"


@dataclass
class DriftSummary:
    total_features: int
    drifted_features: int
    reports: list[DriftReport] = field(default_factory=list)
    overall_status: str = "healthy"  # "healthy", "warning", "critical"


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1: no significant shift
    PSI 0.1–0.25: moderate shift (warning)
    PSI > 0.25: significant shift (critical)
    """
    # Use shared bin edges from reference
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())

    if min_val == max_val:
        return 0.0

    edges = np.linspace(min_val, max_val, bins + 1)
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)

    # Normalize to proportions, add small epsilon to avoid division by zero
    eps = 1e-6
    ref_pct = ref_hist / max(ref_hist.sum(), 1) + eps
    cur_pct = cur_hist / max(cur_hist.sum(), 1) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


class DriftMonitor:
    """Monitors feature and prediction drift against a reference baseline.

    Usage:
        monitor = DriftMonitor(psi_warning=0.1, psi_critical=0.25)
        monitor.set_reference(reference_df)
        summary = monitor.check(current_df)
    """

    def __init__(
        self,
        psi_warning: float = 0.1,
        psi_critical: float = 0.25,
        mean_shift_threshold: float = 2.0,
    ) -> None:
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.mean_shift_threshold = mean_shift_threshold
        self._reference: dict[str, np.ndarray] = {}
        self._ref_stats: dict[str, dict[str, float]] = {}

    def set_reference(self, data: dict[str, np.ndarray | list]) -> None:
        """Set the reference distribution for each feature."""
        for name, values in data.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) < 2:
                continue
            self._reference[name] = arr
            self._ref_stats[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
        logger.info("Drift reference set for %d features", len(self._reference))

    def check_feature(self, name: str, current: np.ndarray) -> DriftReport:
        """Check drift for a single feature."""
        current = np.asarray(current, dtype=float)
        current = current[np.isfinite(current)]

        if name not in self._reference or len(current) < 2:
            return DriftReport(
                feature_name=name, psi=0.0, mean_shift=0.0, std_shift=0.0,
                is_drifted=False, severity="none",
            )

        ref = self._reference[name]
        ref_stats = self._ref_stats[name]

        psi = _compute_psi(ref, current)
        cur_mean = float(np.mean(current))
        cur_std = float(np.std(current))
        ref_std = ref_stats["std"] if ref_stats["std"] > 0 else 1.0

        mean_shift = abs(cur_mean - ref_stats["mean"]) / ref_std
        std_shift = abs(cur_std - ref_stats["std"]) / ref_std if ref_stats["std"] > 0 else 0.0

        if psi >= self.psi_critical or mean_shift >= self.mean_shift_threshold * 2:
            severity = "critical"
        elif psi >= self.psi_warning or mean_shift >= self.mean_shift_threshold:
            severity = "warning"
        else:
            severity = "none"

        return DriftReport(
            feature_name=name,
            psi=round(psi, 4),
            mean_shift=round(mean_shift, 4),
            std_shift=round(std_shift, 4),
            is_drifted=severity != "none",
            severity=severity,
        )

    def check(self, data: dict[str, np.ndarray | list]) -> DriftSummary:
        """Check drift for all features against reference."""
        reports = []
        for name, values in data.items():
            report = self.check_feature(name, np.asarray(values, dtype=float))
            reports.append(report)

        drifted = [r for r in reports if r.is_drifted]
        critical = any(r.severity == "critical" for r in reports)

        overall = "critical" if critical else ("warning" if drifted else "healthy")

        if drifted:
            logger.warning(
                "Drift detected in %d/%d features (status=%s): %s",
                len(drifted), len(reports), overall,
                ", ".join(r.feature_name for r in drifted),
            )

        return DriftSummary(
            total_features=len(reports),
            drifted_features=len(drifted),
            reports=reports,
            overall_status=overall,
        )
