"""Schema validation, dead-letter queue, and retry logic.

Implements Experiment 3: "Improve pipeline reliability from 97% to 99.9%
through validation, retries, dead-letter queues, and observability."
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from infrastructure.logging import get_logger

logger = get_logger("streaming.validation")


# ---------------------------------------------------------------------------
# Schema Definitions (Pydantic validation)
# ---------------------------------------------------------------------------

class IoTReadingSchema(BaseModel):
    """Validated IoT sensor reading."""
    reef_id: str
    timestamp: str
    water_temperature_c: float = Field(ge=-5, le=50)
    ph: float = Field(ge=6.0, le=9.5)
    salinity_psu: float = Field(ge=0, le=60)
    turbidity_ntu: float = Field(ge=0, le=100)
    dissolved_oxygen_mg_l: float = Field(ge=0, le=20)


class NOAARecordSchema(BaseModel):
    """Validated NOAA heat-stress record."""
    reef_id: str
    date: str
    sst_celsius: float = Field(ge=10, le=45)
    sst_anomaly_c: float = Field(ge=-10, le=15)
    hotspot_c: float = Field(ge=0, le=15)
    degree_heating_weeks: float = Field(ge=0, le=30)
    bleaching_alert_area: str


def validate_iot_reading(data: dict[str, Any]) -> tuple[IoTReadingSchema | None, str | None]:
    """Validate an IoT reading. Returns (validated, None) or (None, error_msg)."""
    try:
        return IoTReadingSchema(**data), None
    except ValidationError as e:
        return None, str(e)


def validate_noaa_record(data: dict[str, Any]) -> tuple[NOAARecordSchema | None, str | None]:
    """Validate a NOAA record. Returns (validated, None) or (None, error_msg)."""
    try:
        return NOAARecordSchema(**data), None
    except ValidationError as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Dead-Letter Queue
# ---------------------------------------------------------------------------

@dataclass
class DLQEntry:
    timestamp: str
    source: str
    error: str
    record: dict[str, Any]
    attempt: int = 1


class DeadLetterQueue:
    """Append-only dead-letter queue for rejected/failed records."""

    def __init__(self, path: str | Path = "data/dlq/rejected.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def push(self, entry: DLQEntry) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        self._count += 1
        logger.warning("DLQ: %s — %s", entry.source, entry.error[:100])

    @property
    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text().strip().split("\n") if line)

    def read_all(self) -> list[DLQEntry]:
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text().strip().split("\n"):
            if line:
                entries.append(DLQEntry(**json.loads(line)))
        return entries

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self._count = 0


# ---------------------------------------------------------------------------
# Retry Logic
# ---------------------------------------------------------------------------

def with_retries(
    func: Callable,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    dlq: DeadLetterQueue | None = None,
    source: str = "unknown",
    record: dict[str, Any] | None = None,
) -> Any:
    """Execute a function with exponential backoff retries.

    On final failure, pushes to dead-letter queue if provided.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = backoff_base * (2 ** (attempt - 1))
                logger.warning("Retry %d/%d for %s: %s (wait %.1fs)", attempt, max_retries, source, e, wait)
                time.sleep(wait)

    # All retries exhausted
    logger.error("All %d retries failed for %s: %s", max_retries, source, last_error)
    if dlq and record:
        dlq.push(DLQEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            error=str(last_error),
            record=record,
            attempt=max_retries,
        ))
    return None


# ---------------------------------------------------------------------------
# Validated Pipeline Runner
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    retried_records: int = 0
    dlq_records: int = 0
    success_rate: float = 0.0


def run_validated_pipeline(
    records: list[dict[str, Any]],
    schema: str = "iot",
    dlq: DeadLetterQueue | None = None,
) -> tuple[list[dict[str, Any]], PipelineStats]:
    """Run records through schema validation with DLQ for rejects.

    Returns (valid_records, stats).
    """
    validate_fn = validate_iot_reading if schema == "iot" else validate_noaa_record
    dlq = dlq or DeadLetterQueue()
    valid = []
    stats = PipelineStats()

    for record in records:
        stats.total_records += 1
        validated, error = validate_fn(record)
        if validated:
            stats.valid_records += 1
            valid.append(validated.model_dump())
        else:
            stats.invalid_records += 1
            stats.dlq_records += 1
            dlq.push(DLQEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                source=f"validation:{schema}",
                error=error or "Unknown validation error",
                record=record,
            ))

    stats.success_rate = stats.valid_records / max(stats.total_records, 1)
    logger.info(
        "Validation: %d/%d valid (%.1f%%), %d rejected → DLQ",
        stats.valid_records, stats.total_records, stats.success_rate * 100, stats.dlq_records,
    )
    return valid, stats
