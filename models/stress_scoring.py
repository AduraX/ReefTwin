"""Multi-objective reef stress scoring model.

Inspired by GPURoute's weighted cost model — multiple stress dimensions
are normalized and combined via configurable weights. This replaces the
hardcoded simulation coefficients in the API.

Stress dimensions:
    - thermal: SST anomaly + DHW accumulation
    - water_quality: turbidity + pH deviation
    - biological: dissolved oxygen deficit
    - cumulative: degree heating weeks relative to critical threshold
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from infrastructure.logging import get_logger

logger = get_logger("models.stress_scoring")


@dataclass
class StressWeights:
    thermal: float = 1.5
    water_quality: float = 1.0
    biological: float = 0.8
    cumulative: float = 2.0


@dataclass
class StressBreakdown:
    total_score: float
    thermal_score: float
    water_quality_score: float
    biological_score: float
    cumulative_score: float
    weights_used: dict[str, float]
    dominant_stressor: str


def _sigmoid(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


class ReefStressModel:
    """Weighted multi-objective stress scoring for reef state assessment."""

    def __init__(self, weights: StressWeights | None = None) -> None:
        self.weights = weights or StressWeights()

    def score(self, features: dict[str, Any]) -> StressBreakdown:
        """Compute composite stress score from environmental features.

        All sub-scores are normalized to [0, 1] before weighting.
        """
        # Thermal stress: SST anomaly mapped through sigmoid
        sst_anomaly = float(features.get("sst_anomaly_c", 0.0))
        hotspot = float(features.get("hotspot_c", 0.0))
        thermal = _sigmoid(sst_anomaly, midpoint=1.0, steepness=2.0) * 0.5 + \
                  _sigmoid(hotspot, midpoint=1.0, steepness=3.0) * 0.5

        # Water quality: turbidity and pH deviation from optimal
        turbidity = float(features.get("turbidity_ntu", 0.0))
        ph = float(features.get("ph", 8.1))
        ph_deviation = abs(ph - 8.1)
        wq = _sigmoid(turbidity, midpoint=1.5, steepness=2.0) * 0.6 + \
             _sigmoid(ph_deviation, midpoint=0.15, steepness=10.0) * 0.4

        # Biological stress: dissolved oxygen deficit
        do = float(features.get("dissolved_oxygen_mg_l", 6.5))
        do_deficit = max(0.0, 5.0 - do)  # stress below 5 mg/L
        bio = _sigmoid(do_deficit, midpoint=1.0, steepness=3.0)

        # Cumulative thermal stress: DHW relative to critical thresholds
        dhw = float(features.get("degree_heating_weeks", 0.0))
        cumulative = _sigmoid(dhw, midpoint=4.0, steepness=0.5)

        # Weighted combination
        w = self.weights
        total_weight = w.thermal + w.water_quality + w.biological + w.cumulative
        total = (
            w.thermal * thermal
            + w.water_quality * wq
            + w.biological * bio
            + w.cumulative * cumulative
        ) / total_weight

        total = float(np.clip(total, 0.0, 1.0))

        # Identify dominant stressor
        scores = {
            "thermal": thermal,
            "water_quality": wq,
            "biological": bio,
            "cumulative": cumulative,
        }
        dominant = max(scores, key=scores.get)

        return StressBreakdown(
            total_score=round(total, 4),
            thermal_score=round(thermal, 4),
            water_quality_score=round(wq, 4),
            biological_score=round(bio, 4),
            cumulative_score=round(cumulative, 4),
            weights_used={
                "thermal": w.thermal,
                "water_quality": w.water_quality,
                "biological": w.biological,
                "cumulative": w.cumulative,
            },
            dominant_stressor=dominant,
        )
