"""Offline demo: a scripted MockLLM drives the full agent loop (no network, no API key).

Run from the repo root:  python examples/demo.py
The "model" replays a fixed script, so this demonstrates the framework (planning, tool calls,
error recovery, memory, tracing), not real model intelligence.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from research_agent import Agent, AgentConfig, LongTermMemory, MockLLM  # noqa: E402
from research_agent.builtin_tools import default_registry  # noqa: E402
from research_agent.protocol import act, final, plan_json  # noqa: E402

TASK = "How many years after Python's first release was Amazon founded? Save the answer as a note."


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="research_agent_demo_"))
    script = [
        plan_json("Look up Python's release year", "Look up Amazon's founding year", "Subtract", "Save a note"),
        act("search_web", "Find when Python was first released.", query="Python first released"),
        act("search_web", "Now Amazon's founding.", query="Amazon founded"),
        act("calculator", "Typo on purpose to show recovery.", expression="1994 minus 1991"),
        act("calculator", "Fix the expression.", expression="1994 - 1991"),
        act("notes", "Persist the result.", action="write", title="answer", content="Gap: 3 years"),
        final("Amazon (1994) was founded 3 years after Python's first release (1991). Saved as note 'answer'."),
    ]
    memory = LongTermMemory(out / "memory.json")
    agent = Agent(
        MockLLM(script),
        default_registry(out / "notes"),
        AgentConfig(remember_results=True),
        long_term=memory,
        trace_path=out / "trace.jsonl",
    )
    result = agent.run(TASK)

    print(f"Task:   {TASK}")
    print(f"Plan:   {result.plan}")
    for ev in result.trace:
        d = ev["data"]
        if ev["type"] == "tool_call":
            print(f"  step {d['step']}: {d['tool']}({json.dumps(d['args'])})")
        elif ev["type"] == "tool_result":
            print(f"          -> ok={d['ok']}: {d['output'][:90]}")
    print(f"Status: {result.status}  steps={result.steps}  tokens(estimated)={result.tokens}")
    print(f"Answer: {result.answer}")
    print(f"Trace:  {out / 'trace.jsonl'}  ({len(result.trace)} events)")
    print(f"Memory: {len(memory)} fact(s) stored in {out / 'memory.json'}")


if __name__ == "__main__":
    main()
