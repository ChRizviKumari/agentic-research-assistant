"""Wire protocol between the agent and the LLM: one JSON object per reply."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator


class ToolCall(BaseModel):
    """A request to run a tool."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    """One ReAct turn: a thought plus *either* a tool call *or* a final answer."""

    thought: str = ""
    action: ToolCall | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> AgentStep:
        if (self.action is None) == (self.final_answer is None):
            raise ValueError("reply must contain exactly one of 'action' or 'final_answer'")
        return self


class ProtocolError(ValueError):
    """The model reply could not be parsed into an :class:`AgentStep`."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object embedded in ``text`` (tolerates prose and code fences)."""
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ProtocolError("no JSON object found in reply")


def parse_step(text: str) -> AgentStep:
    """Parse a model reply into an :class:`AgentStep` or raise :class:`ProtocolError`."""
    obj = extract_json_object(text)
    try:
        return AgentStep.model_validate(obj)
    except ValidationError as exc:
        first = exc.errors()[0]["msg"]
        raise ProtocolError(f"invalid reply structure: {first}") from exc


def parse_plan(text: str) -> list[str]:
    """Parse a planner reply of the form ``{"plan": ["step", ...]}``."""
    obj = extract_json_object(text)
    plan = obj.get("plan")
    if not isinstance(plan, list) or not all(isinstance(s, str) for s in plan):
        raise ProtocolError("planner reply needs a 'plan' list of strings")
    return plan


def act(tool: str, thought: str = "", **args: Any) -> str:
    """Build a tool-call reply (helper for scripts and tests)."""
    return json.dumps({"thought": thought, "action": {"tool": tool, "args": args}})


def final(answer: str, thought: str = "") -> str:
    """Build a final-answer reply (helper for scripts and tests)."""
    return json.dumps({"thought": thought, "final_answer": answer})


def plan_json(*steps: str) -> str:
    """Build a planner reply (helper for scripts and tests)."""
    return json.dumps({"plan": list(steps)})
