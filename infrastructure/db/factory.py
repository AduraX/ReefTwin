from __future__ import annotations

from infrastructure.db.base import JsonStateStore, StateStore
from infrastructure.settings import settings

_instance: StateStore | None = None


def get_state_store() -> StateStore:
    global _instance
    if _instance is not None:
        return _instance

    backend = settings.state_store_backend

    if backend == "json":
        _instance = JsonStateStore(settings.state_path)
    elif backend == "s3":
        from infrastructure.db.s3_store import S3StateStore
        _instance = S3StateStore()
    else:
        raise ValueError(
            f"Unknown state store backend: {backend!r}. "
            f"Supported: 'json', 's3'. (PostgreSQL/TimescaleDB coming soon.)"
        )

    return _instance
