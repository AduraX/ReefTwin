"""Avro serialization/deserialization for streaming events.

Uses fastavro for fast Avro encoding/decoding. When a Confluent Schema Registry
is available (Redpanda built-in at :8081), schemas are registered automatically.

Falls back to JSON serialization if fastavro is not installed.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from infrastructure.logging import get_logger

logger = get_logger("streaming.avro_serde")

_SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "avro"


def _load_avro_schema(name: str) -> dict[str, Any]:
    path = _SCHEMA_DIR / f"{name}.avsc"
    if not path.exists():
        raise FileNotFoundError(f"Avro schema not found: {path}")
    with open(path) as f:
        return json.load(f)


# Parsed schema cache
_parsed_schemas: dict[str, Any] = {}


def _get_parsed_schema(name: str) -> Any:
    if name not in _parsed_schemas:
        import fastavro
        raw = _load_avro_schema(name)
        _parsed_schemas[name] = fastavro.parse_schema(raw)
    return _parsed_schemas[name]


def serialize_avro(record: dict[str, Any], schema_name: str = "iot_reading") -> bytes:
    """Serialize a dict to Avro binary using the named schema.

    Falls back to JSON bytes if fastavro is not installed.
    """
    try:
        import fastavro
    except ImportError:
        logger.debug("fastavro not installed — falling back to JSON serialization")
        return json.dumps(record).encode()

    schema = _get_parsed_schema(schema_name)
    buf = io.BytesIO()
    fastavro.schemaless_writer(buf, schema, record)
    return buf.getvalue()


def deserialize_avro(data: bytes, schema_name: str = "iot_reading") -> dict[str, Any]:
    """Deserialize Avro binary to a dict.

    Falls back to JSON decoding if fastavro is not installed.
    """
    try:
        import fastavro
    except ImportError:
        return json.loads(data.decode())

    schema = _get_parsed_schema(schema_name)
    buf = io.BytesIO(data)
    return fastavro.schemaless_reader(buf, schema)


class SchemaRegistryClient:
    """Lightweight client for Confluent/Redpanda Schema Registry.

    Registers Avro schemas and retrieves schema IDs.
    """

    def __init__(self, url: str = "http://localhost:8081") -> None:
        self._url = url.rstrip("/")

    def register(self, subject: str, schema_name: str = "iot_reading") -> int | None:
        """Register an Avro schema. Returns the schema ID, or None on failure."""
        import urllib.request

        raw_schema = _load_avro_schema(schema_name)
        payload = json.dumps({"schemaType": "AVRO", "schema": json.dumps(raw_schema)}).encode()

        try:
            req = urllib.request.Request(
                f"{self._url}/subjects/{subject}/versions",
                data=payload,
                headers={"Content-Type": "application/vnd.schemaregistry.v1+json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read())
                schema_id = result.get("id")
                logger.info("Registered schema '%s' as subject '%s' (id=%s)", schema_name, subject, schema_id)
                return schema_id
        except Exception as e:
            logger.warning("Schema registry registration failed: %s", e)
            return None

    def get_latest(self, subject: str) -> dict[str, Any] | None:
        """Get the latest schema version for a subject."""
        import urllib.request

        try:
            req = urllib.request.Request(
                f"{self._url}/subjects/{subject}/versions/latest",
                headers={"Accept": "application/vnd.schemaregistry.v1+json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.warning("Schema registry lookup failed: %s", e)
            return None
