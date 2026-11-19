"""Eval harness: a small task set, deterministic scripted models, and pass/fail scoring.

IMPORTANT: every task is driven by a :class:`ScriptedLLM`, so the harness measures the *framework*
(parsing, tool execution, retries, budgets, loop guards, memory plumbing), NOT the reasoning
ability of any real model. To evaluate a real model, build the agent with ``AnthropicLLM`` and
reuse :func:`score_task` on the same ``EvalTask`` definitions (scripts are then ignored).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from research_agent.agent import Agent, AgentConfig, AgentResult
from research_agent.builtin_tools import default_registry
from research_agent.llm import LLMProvider, Message, ScriptedLLM, ScriptStep, TransientLLMError
from research_agent.protocol import act, final, plan_json


def last_observation(messages: Sequence[Message]) -> str:
    """Text of the latest tool observation in a conversation (helper for reactive scripts)."""
    for m in reversed(messages):
        if m.content.startswith("Observation"):
            return m.content.split("): ", 1)[-1].strip()
    return ""


def _answer_from_obs(template: str) -> Callable[[Sequence[Message]], str]:
    return lambda msgs: final(template.format(obs=last_observation(msgs)))


@dataclass
class EvalTask:
    """One eval case.

    Attributes:
        id: Short identifier.
        task: Prompt given to the agent.
        script: Scripted model replies (used by the offline harness).
        expect_contains: Substrings the final answer must contain (case-insensitive).
        expect_status: Required terminal status.
        require_tools: Tools that must have been called.
        min_tool_errors: Minimum number of failed tool calls (checks error recovery happened).
        config: Config overrides.
        repeat_last: Passed to the scripted model.
    """

    id: str
    task: str
    script: list[ScriptStep]
    expect_contains: list[str] = field(default_factory=list)
    expect_status: str = "success"
    require_tools: list[str] = field(default_factory=list)
    min_tool_errors: int = 0
    config: dict = field(default_factory=dict)
    repeat_last: bool = False


@dataclass
class EvalOutcome:
    """Scored result of one task."""

    id: str
    passed: bool
    reasons: list[str]
    status: str
    answer: str
    steps: int
    tokens: int
    tool_calls: list[str]


def score_task(task: EvalTask, result: AgentResult) -> EvalOutcome:
    """Apply the task's pass/fail criteria to an agent result."""
    reasons: list[str] = []
    if result.status != task.expect_status:
        reasons.append(f"status {result.status!r} != expected {task.expect_status!r}")
    for needle in task.expect_contains:
        if needle.lower() not in result.answer.lower():
            reasons.append(f"answer missing {needle!r}")
    used = result.tool_calls()
    for tool in task.require_tools:
        if tool not in used:
            reasons.append(f"tool {tool!r} never called")
    if result.tool_errors() < task.min_tool_errors:
        reasons.append(f"expected >= {task.min_tool_errors} tool errors, saw {result.tool_errors()}")
    return EvalOutcome(
        task.id, not reasons, reasons, result.status, result.answer, result.steps, result.tokens, used
    )


def default_tasks() -> list[EvalTask]:
    """The built-in task set (12 tasks)."""
    return [
        EvalTask(
            "calc_basic", "What is 17*23 + 4?",
            [plan_json("calculate"), act("calculator", expression="17*23 + 4"), _answer_from_obs("The result is {obs}.")],
            ["395"], require_tools=["calculator"],
        ),
        EvalTask(
            "search_everest", "How tall is Mount Everest?",
            [plan_json("search"), act("search_web", query="Mount Everest elevation"), _answer_from_obs("Per the corpus: {obs}")],
            ["8,849"], require_tools=["search_web"],
        ),
        EvalTask(
            "search_two_then_calc", "How many years after Python's first release was Amazon founded?",
            [
                plan_json("find python year", "find amazon year", "subtract"),
                act("search_web", query="Python first released"),
                act("search_web", query="Amazon founded"),
                act("calculator", expression="1994 - 1991"),
                _answer_from_obs("{obs} years"),
            ],
            ["3 years"], require_tools=["search_web", "calculator"],
        ),
        EvalTask(
            "python_sum_even_squares", "Sum of squares of even numbers below 10?",
            [plan_json("compute"), act("python_eval", expression="sum(x*x for x in range(10) if x % 2 == 0)"),
             _answer_from_obs("{obs}")],
            ["120"], require_tools=["python_eval"],
        ),
        EvalTask(
            "notes_roundtrip", "Save a note 'todo' saying 'buy milk' and read it back.",
            [plan_json("write", "read"), act("notes", action="write", title="todo", content="buy milk"),
             act("notes", action="read", title="todo"), _answer_from_obs("Note says: {obs}")],
            ["buy milk"], require_tools=["notes"],
        ),
        EvalTask(
            "recover_bad_expression", "Compute 17 times 3.",
            [plan_json("compute"), act("calculator", expression="17 x 3"),
             act("calculator", expression="17 * 3"), _answer_from_obs("{obs}")],
            ["51"], require_tools=["calculator"], min_tool_errors=1,
        ),
        EvalTask(
            "recover_unknown_tool", "What is sqrt(144)?",
            [plan_json("compute"), act("math_solver", expression="sqrt(144)"),
             act("calculator", expression="sqrt(144)"), _answer_from_obs("{obs}")],
            ["12"], min_tool_errors=1,
        ),
        EvalTask(
            "recover_malformed_reply", "What is 2**10?",
            [plan_json("compute"), "Sure! I will just answer: 1024 (no json)",
             act("calculator", expression="2**10"), _answer_from_obs("{obs}")],
            ["1024"],
        ),
        EvalTask(
            "recover_transient_llm_error", "What is 9*9?",
            [plan_json("compute"), TransientLLMError("rate limited"),
             act("calculator", expression="9*9"), _answer_from_obs("{obs}")],
            ["81"],
        ),
        EvalTask(
            "sandbox_blocks_escape", "Run __import__('os').getcwd() and report.",
            [plan_json("try"), act("python_eval", expression="__import__('os').getcwd()"),
             final("Refused: the sandbox does not allow imports.")],
            ["refused"], min_tool_errors=1,
        ),
        EvalTask(
            "budget_stops_runaway", "Keep searching forever.",
            [plan_json("loop")] + [act("calculator", expression=f"{i}+1") for i in range(10)],
            [], expect_status="budget_exceeded", config={"max_steps": 3},
        ),
        EvalTask(
            "stuck_detection", "Repeat yourself.",
            [plan_json("loop"), act("calculator", expression="1+1")],
            [], expect_status="stuck", repeat_last=True,
        ),
    ]


def run_task(task: EvalTask, llm: LLMProvider | None = None) -> EvalOutcome:
    """Run one task in a throwaway notes directory and score it."""
    with tempfile.TemporaryDirectory() as tmp:
        tools = default_registry(Path(tmp) / "notes")
        model = llm or ScriptedLLM(task.script, repeat_last=task.repeat_last)
        agent = Agent(model, tools, AgentConfig(**task.config))
        return score_task(task, agent.run(task.task))


def run_suite(tasks: Sequence[EvalTask] | None = None) -> list[EvalOutcome]:
    """Run every task and return outcomes."""
    return [run_task(t) for t in (tasks if tasks is not None else default_tasks())]


def format_report(outcomes: Sequence[EvalOutcome]) -> str:
    """Human-readable table plus summary line."""
    lines = [f"{'task':<30} {'result':<6} {'status':<16} {'steps':>5} {'tokens':>6}  tools"]
    for o in outcomes:
        lines.append(
            f"{o.id:<30} {'PASS' if o.passed else 'FAIL':<6} {o.status:<16} {o.steps:>5} {o.tokens:>6}  "
            + ",".join(o.tool_calls)
        )
        lines.extend(f"    - {r}" for r in o.reasons)
    passed = sum(o.passed for o in outcomes)
    lines.append(f"\n{passed}/{len(outcomes)} tasks passed (scripted MockLLM; measures the framework, not a real model)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: run the suite, print the report, optionally write JSON."""
    parser = argparse.ArgumentParser(description="Run the offline agent eval suite.")
    parser.add_argument("--report", type=Path, help="write a JSON report to this path")
    args = parser.parse_args(argv)
    outcomes = run_suite()
    print(format_report(outcomes))
    if args.report:
        args.report.write_text(json.dumps([o.__dict__ for o in outcomes], indent=2), encoding="utf-8")
    return 0 if all(o.passed for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
