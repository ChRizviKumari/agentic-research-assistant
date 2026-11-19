"""Short-term (conversation window) and long-term (JSON file) memory."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from research_agent.llm import Message

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of in on at to is are was were be and or for with what who when where how why "
    "do does did it this that as by from".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens with stop words removed."""
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


class ShortTermMemory:
    """Bounded conversation buffer.

    The *pinned* messages (system prompt and task) are never evicted. Older turns are dropped from
    the front of the recent window; eviction always leaves an ``assistant`` message first so that
    user/assistant roles keep alternating (required by chat APIs).
    """

    def __init__(self, pinned: Sequence[Message], max_recent: int = 20) -> None:
        self.pinned = list(pinned)
        self.max_recent = max_recent
        self._recent: list[Message] = []

    def append(self, message: Message) -> None:
        """Add a message, evicting the oldest turns if the window is full."""
        self._recent.append(message)
        while len(self._recent) > self.max_recent or (
            self._recent and self._recent[0].role != "assistant" and len(self._recent) > 1
        ):
            self._recent.pop(0)

    def messages(self) -> list[Message]:
        """Pinned messages followed by the recent window."""
        return [*self.pinned, *self._recent]

    def __len__(self) -> int:
        return len(self.pinned) + len(self._recent)


@dataclass
class MemoryRecord:
    """A persisted fact."""

    id: int
    text: str
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0


class LongTermMemory:
    """JSON-file backed fact store with keyword-overlap recall.

    Writes are atomic (temp file + ``os.replace``). A corrupt file is moved aside to
    ``<name>.corrupt`` and the store starts empty instead of crashing the agent.
    """

    def __init__(self, path: Path | str, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self._clock = clock
        self._records: list[MemoryRecord] = self._load()

    def _load(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return [MemoryRecord(**item) for item in raw]
        except (json.JSONDecodeError, TypeError, ValueError):
            self.path.replace(self.path.with_suffix(self.path.suffix + ".corrupt"))
            return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps([asdict(r) for r in self._records], indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, text: str, tags: Iterable[str] = ()) -> MemoryRecord:
        """Persist a new fact and return it."""
        record = MemoryRecord(
            id=max((r.id for r in self._records), default=0) + 1,
            text=text,
            tags=list(tags),
            created_at=self._clock(),
        )
        self._records.append(record)
        self._save()
        return record

    def recall(self, query: str, k: int = 3) -> list[MemoryRecord]:
        """Return up to ``k`` records sharing the most keywords with ``query`` (newest wins ties)."""
        q = set(tokenize(query))
        scored: list[tuple[int, float, MemoryRecord]] = []
        for rec in self._records:
            overlap = len(q & set(tokenize(rec.text + " " + " ".join(rec.tags))))
            if overlap:
                scored.append((overlap, rec.created_at, rec))
        scored.sort(key=lambda t: (t[0], t[1], t[2].id), reverse=True)
        return [rec for _, _, rec in scored[:k]]

    def all(self) -> list[MemoryRecord]:
        """All stored records."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)
