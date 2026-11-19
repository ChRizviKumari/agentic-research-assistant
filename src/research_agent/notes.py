"""File-backed notes store with path-traversal protection."""

from __future__ import annotations

import re
from pathlib import Path

from research_agent.tools import ToolError

_SLUG_BAD = re.compile(r"[^a-z0-9_-]+")


class NotesStore:
    """Stores notes as ``<slug>.md`` files inside a single directory."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, title: str) -> Path:
        slug = _SLUG_BAD.sub("-", title.strip().lower()).strip("-")[:80]
        if not slug:
            raise ToolError("note title must contain letters or digits")
        path = (self.directory / f"{slug}.md").resolve()
        if path.parent != self.directory.resolve():  # defence in depth; slug already has no '/'
            raise ToolError("invalid note title")
        return path

    def write(self, title: str, content: str, append: bool = False) -> str:
        """Create, overwrite or append to a note; returns the stored file name."""
        path = self._path(title)
        if append and path.exists():
            content = path.read_text(encoding="utf-8").rstrip("\n") + "\n" + content
        path.write_text(content.rstrip("\n") + "\n", encoding="utf-8")
        return path.name

    def read(self, title: str) -> str:
        """Return a note's content or raise :class:`ToolError` if it does not exist."""
        path = self._path(title)
        if not path.exists():
            raise ToolError(f"note '{title}' not found")
        return path.read_text(encoding="utf-8")

    def list(self) -> list[str]:
        """Sorted note names (without extension)."""
        return sorted(p.stem for p in self.directory.glob("*.md"))
