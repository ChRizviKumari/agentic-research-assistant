"""Structured JSON trace logging (one JSON object per event, optionally JSONL on disk)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any


class TraceLogger:
    """Collects events in memory and optionally appends them to a JSONL file.

    Every event has ``run_id``, ``seq`` (monotonic per run), ``ts``, ``type`` and a ``data`` dict.

    Args:
        path: Optional JSONL file to append to.
        run_id: Identifier shared by all events of a run (random if omitted).
        clock: Time source; injectable so tests can be deterministic.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        run_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path) if path else None
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._clock = clock
        self.events: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **data: Any) -> dict[str, Any]:
        """Record an event and return it."""
        event = {
            "run_id": self.run_id,
            "seq": len(self.events),
            "ts": self._clock(),
            "type": event_type,
            "data": data,
        }
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        return event

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        """Return all events of one type."""
        return [e for e in self.events if e["type"] == event_type]
