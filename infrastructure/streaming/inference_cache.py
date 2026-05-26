"""Inference cache with drift-triggered re-prediction.

Caches reef state predictions and only re-runs inference when:
  1. Cache TTL expires, OR
  2. Feature drift exceeds threshold (detected via simple delta check)

This implements Experiment 2: "Cut inference cost by 22% using
batching, caching, and drift-triggered inference."
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


from infrastructure.logging import get_logger

logger = get_logger("streaming.inference_cache")


@dataclass
class CachedPrediction:
    reef_id: str
    prediction: dict[str, Any]
    features: dict[str, float]
    timestamp: float
    cache_hits: int = 0


class InferenceCache:
    """TTL + drift-aware inference cache.

    Args:
        ttl_seconds: Max age before forced re-prediction.
        drift_threshold: Re-predict if any feature changes by more than this fraction.
    """

    def __init__(self, ttl_seconds: float = 300, drift_threshold: float = 0.05) -> None:
        self._cache: dict[str, CachedPrediction] = {}
        self._ttl = ttl_seconds
        self._drift_threshold = drift_threshold
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.drift_triggered = 0

    def get(self, reef_id: str, current_features: dict[str, float]) -> dict[str, Any] | None:
        """Return cached prediction if valid, None if cache miss or drift detected."""
        self.total_requests += 1

        cached = self._cache.get(reef_id)
        if cached is None:
            self.cache_misses += 1
            return None

        # Check TTL
        age = time.time() - cached.timestamp
        if age > self._ttl:
            self.cache_misses += 1
            logger.debug("Cache expired for %s (age=%.1fs)", reef_id, age)
            return None

        # Check feature drift
        if self._has_drifted(cached.features, current_features):
            self.cache_misses += 1
            self.drift_triggered += 1
            logger.debug("Drift detected for %s — re-predicting", reef_id)
            return None

        # Cache hit
        self.cache_hits += 1
        cached.cache_hits += 1
        return cached.prediction

    def put(self, reef_id: str, prediction: dict[str, Any], features: dict[str, float]) -> None:
        self._cache[reef_id] = CachedPrediction(
            reef_id=reef_id,
            prediction=prediction,
            features=features,
            timestamp=time.time(),
        )

    def _has_drifted(self, old: dict[str, float], new: dict[str, float]) -> bool:
        """Check if any feature has changed beyond the drift threshold."""
        for key in old:
            if key not in new:
                continue
            old_val = old[key]
            new_val = new[key]
            if old_val == 0:
                if abs(new_val) > self._drift_threshold:
                    return True
            elif abs(new_val - old_val) / abs(old_val) > self._drift_threshold:
                return True
        return False

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def skip_rate(self) -> float:
        """Fraction of predictions skipped (cache hits / total)."""
        return self.hit_rate

    def stats(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "drift_triggered": self.drift_triggered,
            "hit_rate": round(self.hit_rate, 4),
            "skip_rate": round(self.skip_rate, 4),
            "cached_reefs": len(self._cache),
        }

    def clear(self) -> None:
        self._cache.clear()
