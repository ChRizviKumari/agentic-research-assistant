"""Typed tool abstraction and registry with validation, retries and uniform error reporting."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

MAX_OUTPUT_CHARS = 2_000


class ToolError(Exception):
    """A tool failed in a way the model should be told about (not retried)."""


class TransientToolError(ToolError):
    """A tool failed transiently; the registry retries it before reporting failure."""


@dataclass(frozen=True)
class Tool:
    """A callable capability exposed to the model.

    Attributes:
        name: Identifier the model uses in ``action.tool``.
        description: One-line description shown in the system prompt.
        args_model: Pydantic model describing and validating the arguments.
        func: Implementation; receives the *validated* args model and returns text.
    """

    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[Any], str]

    def signature(self) -> str:
        """Compact ``name(arg: type, ...)`` string for prompts."""
        props = self.args_model.model_json_schema().get("properties", {})
        required = set(self.args_model.model_json_schema().get("required", []))
        parts = []
        for arg, spec in props.items():
            kind = spec.get("type") or "|".join(s.get("type", "any") for s in spec.get("anyOf", []))
            parts.append(f"{arg}{'' if arg in required else '?'}: {kind}")
        return f"{self.name}({', '.join(parts)})"


@dataclass(frozen=True)
class ToolResult:
    """Outcome of executing a tool call."""

    tool: str
    ok: bool
    output: str
    attempts: int = 1


class ToolRegistry:
    """Holds tools and executes calls safely.

    ``execute`` never raises: unknown tools, schema violations and tool exceptions all come back as
    ``ToolResult(ok=False, ...)`` so the agent can show the error to the model and let it recover.
    """

    def __init__(
        self,
        max_retries: int = 2,
        backoff_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep

    def register(self, tool: Tool) -> None:
        """Add a tool; duplicate names are rejected."""
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        """Registered tool names."""
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def describe(self) -> str:
        """Multi-line tool listing for the system prompt."""
        return "\n".join(f"- {t.signature()}: {t.description}" for t in self._tools.values())

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Validate and run a tool, retrying :class:`TransientToolError` with backoff."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, False, f"unknown tool '{name}'. Available: {', '.join(self.names())}")
        try:
            parsed = tool.args_model.model_validate(args)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors()
            )
            return ToolResult(name, False, f"invalid arguments: {details}")
        attempts = 0
        while True:
            attempts += 1
            try:
                return ToolResult(name, True, _clip(tool.func(parsed)), attempts)
            except TransientToolError as exc:
                if attempts > self.max_retries:
                    return ToolResult(name, False, f"transient failure persisted: {exc}", attempts)
                self._sleep(self.backoff_seconds * 2 ** (attempts - 1))
            except ToolError as exc:
                return ToolResult(name, False, str(exc), attempts)
            except Exception as exc:  # noqa: BLE001 - tools are untrusted boundaries
                return ToolResult(name, False, f"{type(exc).__name__}: {exc}", attempts)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT_CHARS else text[:MAX_OUTPUT_CHARS] + "... [truncated]"
