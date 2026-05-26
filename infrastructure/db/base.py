from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class StateStore(ABC):
    """Abstract interface for reef twin state persistence."""

    @abstractmethod
    def save(self, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def load(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_states(self) -> list[dict[str, Any]]: ...


class JsonStateStore(StateStore):
    """File-based JSON state store (development default)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"generated_at": None, "states": []}
        return json.loads(self.path.read_text())

    def load_states(self) -> list[dict[str, Any]]:
        return self.load().get("states", [])
