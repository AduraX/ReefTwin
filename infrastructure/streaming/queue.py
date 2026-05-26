"""Pluggable event queue for streaming ingestion.

Backends:
    - memory:  asyncio.Queue (dev/test, no external deps)
    - kafka:   Kafka/Redpanda producer+consumer (production)

Selection via REEFTWIN_STREAM_BACKEND env var.
Decouples producers (IoT simulator, NOAA poller) from consumers
(feature pipeline, twin state updater).
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from infrastructure.logging import get_logger

logger = get_logger("streaming.queue")


@dataclass
class Event:
    topic: str
    key: str
    value: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class EventProducer(ABC):
    @abstractmethod
    def send(self, event: Event) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...


class EventConsumer(ABC):
    @abstractmethod
    def subscribe(self, topics: list[str]) -> None: ...

    @abstractmethod
    def poll(self, timeout: float = 1.0) -> list[Event]: ...

    @abstractmethod
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# In-Memory Backend (dev/test)
# ---------------------------------------------------------------------------

class _InMemoryBroker:
    """Shared in-memory broker for producer/consumer pairs."""

    def __init__(self) -> None:
        self._queues: dict[str, list[Event]] = defaultdict(list)
        self._offsets: dict[str, int] = defaultdict(int)

    def publish(self, event: Event) -> None:
        self._queues[event.topic].append(event)

    def consume(self, topic: str, consumer_id: str) -> list[Event]:
        key = f"{topic}:{consumer_id}"
        offset = self._offsets[key]
        events = self._queues[topic][offset:]
        self._offsets[key] = len(self._queues[topic])
        return events

    @property
    def total_events(self) -> int:
        return sum(len(q) for q in self._queues.values())


_broker = _InMemoryBroker()


class InMemoryProducer(EventProducer):
    def __init__(self, broker: _InMemoryBroker | None = None) -> None:
        self._broker = broker or _broker
        self._sent = 0

    def send(self, event: Event) -> None:
        self._broker.publish(event)
        self._sent += 1

    def flush(self) -> None:
        pass

    @property
    def sent_count(self) -> int:
        return self._sent


class InMemoryConsumer(EventConsumer):
    _counter = 0

    def __init__(self, broker: _InMemoryBroker | None = None) -> None:
        self._broker = broker or _broker
        InMemoryConsumer._counter += 1
        self._id = f"consumer-{InMemoryConsumer._counter}"
        self._topics: list[str] = []

    def subscribe(self, topics: list[str]) -> None:
        self._topics = topics

    def poll(self, timeout: float = 1.0) -> list[Event]:
        events = []
        for topic in self._topics:
            events.extend(self._broker.consume(topic, self._id))
        return events

    def close(self) -> None:
        self._topics = []


# ---------------------------------------------------------------------------
# Kafka Backend (production)
# ---------------------------------------------------------------------------

class KafkaProducer(EventProducer):
    """Kafka/Redpanda producer. Requires confluent-kafka.

    When ``avro_schema`` is set, values are serialized as Avro binary
    using fastavro (falls back to JSON if fastavro is not installed).
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        avro_schema: str | None = None,
    ) -> None:
        try:
            from confluent_kafka import Producer
        except ImportError:
            raise ImportError("Install confluent-kafka: pip install confluent-kafka")
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})
        self._avro_schema = avro_schema
        logger.info("Kafka producer: %s (avro=%s)", bootstrap_servers, avro_schema or "off")

    def send(self, event: Event) -> None:
        if self._avro_schema:
            from infrastructure.streaming.avro_serde import serialize_avro
            value_bytes = serialize_avro(event.value, self._avro_schema)
        else:
            value_bytes = json.dumps(event.value).encode()
        self._producer.produce(
            topic=event.topic,
            key=event.key.encode(),
            value=value_bytes,
        )

    def flush(self) -> None:
        self._producer.flush()


class KafkaConsumer(EventConsumer):
    """Kafka/Redpanda consumer. Requires confluent-kafka.

    When ``avro_schema`` is set, values are deserialized from Avro binary.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "reeftwin",
        avro_schema: str | None = None,
    ) -> None:
        try:
            from confluent_kafka import Consumer
        except ImportError:
            raise ImportError("Install confluent-kafka: pip install confluent-kafka")
        self._consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
        })
        self._avro_schema = avro_schema
        logger.info("Kafka consumer: %s group=%s (avro=%s)", bootstrap_servers, group_id, avro_schema or "off")

    def subscribe(self, topics: list[str]) -> None:
        self._consumer.subscribe(topics)

    def poll(self, timeout: float = 1.0) -> list[Event]:
        msg = self._consumer.poll(timeout)
        if msg is None or msg.error():
            return []
        raw = msg.value()
        if self._avro_schema:
            from infrastructure.streaming.avro_serde import deserialize_avro
            value = deserialize_avro(raw, self._avro_schema)
        else:
            value = json.loads(raw.decode())
        return [Event(
            topic=msg.topic(),
            key=msg.key().decode() if msg.key() else "",
            value=value,
            timestamp=msg.timestamp()[1] / 1000 if msg.timestamp()[0] else time.time(),
        )]

    def close(self) -> None:
        self._consumer.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_producer(backend: str = "memory", avro_schema: str | None = None) -> EventProducer:
    if backend == "memory":
        return InMemoryProducer()
    elif backend == "kafka":
        return KafkaProducer(avro_schema=avro_schema)
    else:
        raise ValueError(f"Unknown stream backend: {backend!r}. Options: memory, kafka")


def get_consumer(backend: str = "memory", avro_schema: str | None = None) -> EventConsumer:
    if backend == "memory":
        return InMemoryConsumer()
    elif backend == "kafka":
        return KafkaConsumer(avro_schema=avro_schema)
    else:
        raise ValueError(f"Unknown stream backend: {backend!r}. Options: memory, kafka")
