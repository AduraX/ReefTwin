"""Tests for streaming infrastructure, inference cache, and validation."""


from infrastructure.streaming.queue import (
    Event, InMemoryProducer, InMemoryConsumer, _InMemoryBroker,
    get_producer, get_consumer,
)
from infrastructure.streaming.iot_producer import produce_readings
from infrastructure.streaming.inference_cache import InferenceCache
from infrastructure.streaming.validation import (
    validate_iot_reading, DeadLetterQueue, with_retries, run_validated_pipeline,
)


# --- Event Queue ---

def test_inmemory_producer_consumer():
    broker = _InMemoryBroker()
    producer = InMemoryProducer(broker)
    consumer = InMemoryConsumer(broker)
    consumer.subscribe(["test.topic"])

    producer.send(Event(topic="test.topic", key="k1", value={"x": 1}))
    producer.send(Event(topic="test.topic", key="k2", value={"x": 2}))
    producer.flush()

    events = consumer.poll()
    assert len(events) == 2
    assert events[0].value["x"] == 1


def test_consumer_only_gets_subscribed_topics():
    broker = _InMemoryBroker()
    producer = InMemoryProducer(broker)
    consumer = InMemoryConsumer(broker)
    consumer.subscribe(["topic.a"])

    producer.send(Event(topic="topic.a", key="k", value={"a": 1}))
    producer.send(Event(topic="topic.b", key="k", value={"b": 2}))

    events = consumer.poll()
    assert len(events) == 1
    assert events[0].topic == "topic.a"


def test_factory_memory_backend():
    producer = get_producer("memory")
    consumer = get_consumer("memory")
    assert isinstance(producer, InMemoryProducer)
    assert isinstance(consumer, InMemoryConsumer)


# --- IoT Producer ---

def test_iot_producer():
    broker = _InMemoryBroker()
    producer = InMemoryProducer(broker)
    sent = produce_readings(producer=producer, n_events=50)
    assert sent == 50
    assert producer.sent_count == 50


# --- Inference Cache ---

def test_cache_hit():
    cache = InferenceCache(ttl_seconds=60, drift_threshold=0.05)
    features = {"temp": 28.3, "ph": 8.05}
    cache.put("reef_a", {"risk": 0.5}, features)

    result = cache.get("reef_a", {"temp": 28.31, "ph": 8.05})  # small change
    assert result is not None
    assert result["risk"] == 0.5
    assert cache.cache_hits == 1


def test_cache_miss_on_drift():
    cache = InferenceCache(ttl_seconds=60, drift_threshold=0.05)
    features = {"temp": 28.3, "ph": 8.05}
    cache.put("reef_a", {"risk": 0.5}, features)

    result = cache.get("reef_a", {"temp": 31.0, "ph": 7.8})  # big change
    assert result is None
    assert cache.drift_triggered == 1


def test_cache_miss_on_new_reef():
    cache = InferenceCache()
    result = cache.get("unknown_reef", {"temp": 28})
    assert result is None
    assert cache.cache_misses == 1


def test_cache_stats():
    cache = InferenceCache(ttl_seconds=60, drift_threshold=0.1)
    features = {"temp": 28.0}
    cache.put("r1", {"risk": 0.3}, features)
    cache.get("r1", {"temp": 28.01})
    cache.get("r1", {"temp": 28.01})
    cache.get("r2", {"temp": 29.0})

    stats = cache.stats()
    assert stats["cache_hits"] == 2
    assert stats["cache_misses"] == 1
    assert stats["hit_rate"] > 0


# --- Schema Validation ---

def test_valid_iot_reading():
    data = {
        "reef_id": "gbr_heron_reef",
        "timestamp": "2026-05-07T00:00:00Z",
        "water_temperature_c": 28.5,
        "ph": 8.1,
        "salinity_psu": 35.0,
        "turbidity_ntu": 0.8,
        "dissolved_oxygen_mg_l": 6.5,
    }
    validated, error = validate_iot_reading(data)
    assert validated is not None
    assert error is None


def test_invalid_iot_reading_out_of_range():
    data = {
        "reef_id": "test",
        "timestamp": "2026-05-07T00:00:00Z",
        "water_temperature_c": 999,  # out of range
        "ph": 8.1,
        "salinity_psu": 35.0,
        "turbidity_ntu": 0.8,
        "dissolved_oxygen_mg_l": 6.5,
    }
    validated, error = validate_iot_reading(data)
    assert validated is None
    assert error is not None


def test_invalid_iot_reading_missing_field():
    data = {"reef_id": "test"}
    validated, error = validate_iot_reading(data)
    assert validated is None


# --- Dead-Letter Queue ---

def test_dlq_push_and_read(tmp_path):
    dlq = DeadLetterQueue(path=tmp_path / "dlq.jsonl")
    from infrastructure.streaming.validation import DLQEntry
    dlq.push(DLQEntry(timestamp="t1", source="test", error="bad data", record={"x": 1}))
    dlq.push(DLQEntry(timestamp="t2", source="test", error="bad data", record={"x": 2}))

    assert dlq.count == 2
    entries = dlq.read_all()
    assert len(entries) == 2


# --- Retries ---

def test_with_retries_succeeds():
    counter = {"n": 0}
    def flaky():
        counter["n"] += 1
        if counter["n"] < 3:
            raise ValueError("not yet")
        return "ok"

    result = with_retries(flaky, max_retries=3, backoff_base=0.01)
    assert result == "ok"


def test_with_retries_exhausted(tmp_path):
    dlq = DeadLetterQueue(path=tmp_path / "dlq.jsonl")
    def always_fails():
        raise RuntimeError("fail")

    result = with_retries(always_fails, max_retries=2, backoff_base=0.01, dlq=dlq, source="test", record={"x": 1})
    assert result is None
    assert dlq.count == 1


# --- Validated Pipeline ---

def test_validated_pipeline(tmp_path):
    records = [
        {"reef_id": "r1", "timestamp": "t", "water_temperature_c": 28, "ph": 8.1,
         "salinity_psu": 35, "turbidity_ntu": 0.8, "dissolved_oxygen_mg_l": 6.5},
        {"reef_id": "bad"},  # invalid
        {"reef_id": "r2", "timestamp": "t", "water_temperature_c": 29, "ph": 8.0,
         "salinity_psu": 35, "turbidity_ntu": 0.5, "dissolved_oxygen_mg_l": 7.0},
    ]
    dlq = DeadLetterQueue(path=tmp_path / "dlq.jsonl")
    valid, stats = run_validated_pipeline(records, schema="iot", dlq=dlq)

    assert len(valid) == 2
    assert stats.valid_records == 2
    assert stats.invalid_records == 1
    assert stats.dlq_records == 1
    assert dlq.count == 1
