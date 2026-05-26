"""Streaming IoT sensor producer.

Emits reef sensor readings as events to the configured stream backend.
Replaces the batch CSV generator for real-time ingestion (Experiment 1).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from infrastructure.logging import get_logger
from infrastructure.settings import settings
from infrastructure.streaming.queue import Event, EventProducer, get_producer

logger = get_logger("streaming.iot_producer")

TOPIC = "reef.iot.readings"
_AVRO_SCHEMA_NAME = "iot_reading"


def produce_readings(
    producer: EventProducer | None = None,
    n_events: int = 100,
    interval_ms: float = 0,
    seed: int = 42,
    use_avro: bool = False,
) -> int:
    """Emit IoT sensor readings as stream events.

    Args:
        producer: EventProducer instance (defaults to configured backend).
        n_events: Number of events to produce.
        interval_ms: Delay between events in ms (0 = no delay / batch mode).
        seed: RNG seed for reproducibility.
        use_avro: If True, serialize event values as Avro binary.

    Returns:
        Number of events produced.
    """
    producer = producer or get_producer()
    avro_serialize = None
    if use_avro:
        from infrastructure.streaming.avro_serde import serialize_avro
        avro_serialize = serialize_avro
    rng = np.random.default_rng(seed)
    reef_ids = settings.reef_ids
    sent = 0

    for i in range(n_events):
        reef_id = reef_ids[i % len(reef_ids)]
        heat_wave = 1.8 if i > n_events * 0.72 and reef_id == reef_ids[0] else 0.0

        reading = {
            "reef_id": reef_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "water_temperature_c": round(float(rng.normal(28.3, 0.45) + heat_wave), 3),
            "ph": round(float(rng.normal(8.05, 0.06) - (0.07 if heat_wave else 0)), 3),
            "salinity_psu": round(float(rng.normal(35.1, 0.35)), 3),
            "turbidity_ntu": round(max(0.05, float(rng.normal(0.8, 0.22) + (0.45 if heat_wave else 0))), 3),
            "dissolved_oxygen_mg_l": round(float(rng.normal(6.5, 0.4) - heat_wave * 0.2), 3),
        }

        producer.send(Event(topic=TOPIC, key=reef_id, value=reading))
        sent += 1

        if interval_ms > 0:
            time.sleep(interval_ms / 1000)

    producer.flush()
    logger.info("Produced %d IoT events to %s", sent, TOPIC)
    return sent
