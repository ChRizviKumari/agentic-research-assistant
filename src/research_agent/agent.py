"""The planner + executor (ReAct) loop."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError  # noqa: F401  (re-exported for callers catching errors)

from research_agent.budget import Budget, BudgetExceeded
from research_agent.llm import LLMError, LLMProvider, LLMResponse, Message, TransientLLMError
from research_agent.memory import LongTermMemory, ShortTermMemory
from research_agent.protocol import AgentStep, ProtocolError, parse_plan, parse_step
from research_agent.tools import ToolRegistry
from research_agent.trace import TraceLogger

Status = Literal["success", "budget_exceeded", "failed", "stuck"]

SYSTEM_PROMPT = """You are a careful research assistant that solves tasks by using tools.

Available tools:
{tools}

Reply with exactly ONE JSON object per turn, either
  {{"thought": "...", "action": {{"tool": "<name>", "args": {{...}}}}}}
or, when you have the answer,
  {{"thought": "...", "final_answer": "..."}}
Tool results arrive as 'Observation' messages. If a tool reports an error, read it and try a
corrected call or a different tool. You have at most {max_steps} steps."""

PLANNER_PROMPT = """Break the task into a short ordered list of steps that use the tools below.
Reply with ONE JSON object: {{"plan": ["step 1", "step 2", ...]}}.

Available tools:
{tools}"""


@dataclass
class AgentConfig:
    """Tunable behaviour of :class:`Agent`.

    Attributes:
        max_steps: Step budget (ReAct iterations; the planner call does not count as a step).
        max_tokens: Token budget across all LLM calls in a run.
        use_planner: Run a planning call before the loop.
        max_parse_retries: Extra attempts when the model reply is not valid protocol JSON.
        max_llm_retries: Extra attempts after a :class:`TransientLLMError`.
        max_repeat_calls: Identical tool calls tolerated before the run is declared ``stuck``.
        backoff_seconds: Base for exponential backoff between LLM retries.
        remember_results: Store successful (task, answer) pairs in long-term memory.
        recent_messages: Size of the short-term conversation window.
    """

    max_steps: int = 12
    max_tokens: int = 20_000
    use_planner: bool = True
    max_parse_retries: int = 2
    max_llm_retries: int = 2
    max_repeat_calls: int = 2
    backoff_seconds: float = 0.0
    remember_results: bool = False
    recent_messages: int = 20


@dataclass
class AgentResult:
    """Everything a caller needs to know about a run."""

    status: Status
    answer: str
    steps: int
    tokens: int
    plan: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the run produced a final answer."""
        return self.status == "success"

    def tool_calls(self) -> list[str]:
        """Names of tools invoked, in order."""
        return [e["data"]["tool"] for e in self.trace if e["type"] == "tool_call"]

    def tool_errors(self) -> int:
        """Number of tool calls that ended in failure."""
        return sum(1 for e in self.trace if e["type"] == "tool_result" and not e["data"]["ok"])


class _Stop(Exception):
    """Internal: abort the loop with a terminal status."""

    def __init__(self, status: Status, message: str) -> None:
        super().__init__(message)
        self.status = status


class Agent:
    """ReAct agent: plan once, then alternate model reasoning and tool execution.

    Args:
        llm: Any :class:`LLMProvider`.
        tools: Tool registry the model may call.
        config: Behavioural settings.
        long_term: Optional persistent memory; relevant facts are injected into the prompt.
        trace_path: Optional JSONL file receiving structured trace events.
        sleep: Sleep function used for retry backoff (injectable for tests).
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        long_term: LongTermMemory | None = None,
        trace_path: Path | str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig()
        self.long_term = long_term
        self.trace_path = trace_path
        self._sleep = sleep

    # -- public API -----------------------------------------------------------------------
    def run(self, task: str) -> AgentResult:
        """Solve ``task`` and return an :class:`AgentResult` (never raises for model/tool faults)."""
        cfg = self.config
        trace = TraceLogger(self.trace_path)
        budget = Budget(cfg.max_steps, cfg.max_tokens)
        trace.log("run_start", task=task, max_steps=cfg.max_steps, max_tokens=cfg.max_tokens)
        plan: list[str] = []
        status: Status = "failed"
        answer = ""
        try:
            memories = self.long_term.recall(task) if self.long_term is not None else []
            if memories:
                trace.log("memory_recall", ids=[m.id for m in memories])
            if cfg.use_planner:
                plan = self._plan(task, budget, trace)
            memory = self._init_memory(task, plan, [m.text for m in memories])
            answer = self._loop(memory, budget, trace)
            status = "success"
        except _Stop as stop:
            status, answer = stop.status, str(stop)
        except BudgetExceeded as exc:
            status, answer = "budget_exceeded", f"Stopped: {exc}"
            trace.log("budget_exceeded", kind=exc.kind, message=str(exc))
        except LLMError as exc:
            status, answer = "failed", f"LLM error: {exc}"
            trace.log("llm_failure", error=str(exc))
        if status == "success" and self.long_term is not None and cfg.remember_results:
            self.long_term.add(f"Q: {task} A: {answer}", tags=["qa"])
        trace.log("run_end", status=status, steps=budget.steps_used, tokens=budget.tokens_used)
        return AgentResult(status, answer, budget.steps_used, budget.tokens_used, plan, trace.events)

    # -- stages ---------------------------------------------------------------------------
    def _plan(self, task: str, budget: Budget, trace: TraceLogger) -> list[str]:
        messages = [
            Message("system", PLANNER_PROMPT.format(tools=self.tools.describe())),
            Message("user", task),
        ]
        try:
            resp = self._call_llm(messages, budget, trace, "plan")
            plan = parse_plan(resp.text)
        except ProtocolError as exc:
            trace.log("plan_fallback", error=str(exc))
            return []
        trace.log("plan", steps=plan)
        return plan

    def _init_memory(self, task: str, plan: list[str], memories: list[str]) -> ShortTermMemory:
        system = SYSTEM_PROMPT.format(tools=self.tools.describe(), max_steps=self.config.max_steps)
        user = f"Task: {task}"
        if memories:
            user += "\n\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memories)
        if plan:
            user += "\n\nSuggested plan:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1))
        return ShortTermMemory(
            [Message("system", system), Message("user", user)], self.config.recent_messages
        )

    def _loop(self, memory: ShortTermMemory, budget: Budget, trace: TraceLogger) -> str:
        seen: dict[str, int] = {}
        while True:
            budget.start_step()
            step = self._next_step(memory, budget, trace)
            trace.log("thought", step=budget.steps_used, thought=step.thought)
            if step.final_answer is not None:
                return step.final_answer
            assert step.action is not None
            call = step.action
            signature = f"{call.tool}:{json.dumps(call.args, sort_keys=True, default=str)}"
            seen[signature] = seen.get(signature, 0) + 1
            if seen[signature] > self.config.max_repeat_calls:
                trace.log("stuck", signature=signature)
                raise _Stop("stuck", f"Stopped: repeated identical call to '{call.tool}'")
            trace.log("tool_call", step=budget.steps_used, tool=call.tool, args=call.args)
            result = self.tools.execute(call.tool, call.args)
            trace.log(
                "tool_result",
                step=budget.steps_used,
                tool=call.tool,
                ok=result.ok,
                attempts=result.attempts,
                output=result.output,
            )
            memory.append(
                Message(
                    "user",
                    f"Observation (tool={call.tool}, ok={str(result.ok).lower()}): {result.output}",
                )
            )

    def _next_step(self, memory: ShortTermMemory, budget: Budget, trace: TraceLogger) -> AgentStep:
        for attempt in range(self.config.max_parse_retries + 1):
            resp = self._call_llm(memory.messages(), budget, trace, "step")
            memory.append(Message("assistant", resp.text))
            try:
                return parse_step(resp.text)
            except ProtocolError as exc:
                trace.log("parse_error", attempt=attempt, error=str(exc), reply=resp.text[:300])
                memory.append(
                    Message(
                        "user",
                        f"Your reply was invalid ({exc}). Reply with a single JSON object "
                        "containing either 'action' or 'final_answer'.",
                    )
                )
        raise _Stop("failed", "Stopped: model never produced a valid reply")

    def _call_llm(
        self, messages: list[Message], budget: Budget, trace: TraceLogger, purpose: str
    ) -> LLMResponse:
        for attempt in range(self.config.max_llm_retries + 1):
            try:
                resp = self.llm.complete(messages)
            except TransientLLMError as exc:
                trace.log("llm_retry", purpose=purpose, attempt=attempt, error=str(exc))
                if attempt == self.config.max_llm_retries:
                    raise
                self._sleep(self.config.backoff_seconds * 2**attempt)
                continue
            trace.log(
                "llm_call",
                purpose=purpose,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            )
            budget.charge(resp.total_tokens)
            return resp
        raise LLMError("unreachable")  # pragma: no cover
