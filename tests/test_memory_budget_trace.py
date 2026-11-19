import json

import pytest

from research_agent.budget import Budget, BudgetExceeded
from research_agent.llm import Message
from research_agent.memory import LongTermMemory, ShortTermMemory
from research_agent.trace import TraceLogger


def test_budget_steps_and_tokens():
    b = Budget(max_steps=2, max_tokens=100)
    b.start_step()
    b.start_step()
    with pytest.raises(BudgetExceeded) as e:
        b.start_step()
    assert e.value.kind == "steps"
    b.charge(100)
    assert b.tokens_remaining == 0
    with pytest.raises(BudgetExceeded) as e:
        b.charge(1)
    assert e.value.kind == "tokens"


def test_short_term_memory_pins_and_evicts_keeping_alternation():
    pinned = [Message("system", "s"), Message("user", "task")]
    mem = ShortTermMemory(pinned, max_recent=4)
    for i in range(10):
        mem.append(Message("assistant", f"a{i}"))
        mem.append(Message("user", f"u{i}"))
    msgs = mem.messages()
    assert msgs[:2] == pinned
    recent = msgs[2:]
    assert len(recent) <= 4 and recent[0].role == "assistant"
    assert recent[-1].content == "u9"


def test_long_term_memory_persists_and_recalls(tmp_path):
    path = tmp_path / "m.json"
    mem = LongTermMemory(path)
    mem.add("Everest is 8849 metres tall", tags=["geo"])
    mem.add("Python was created by Guido")
    reloaded = LongTermMemory(path)
    assert len(reloaded) == 2
    assert reloaded.recall("how tall is everest")[0].tags == ["geo"]
    assert reloaded.recall("unrelated quantum") == []
    assert reloaded.add("x").id == 3


def test_long_term_memory_survives_corruption(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{not json")
    mem = LongTermMemory(path)
    assert len(mem) == 0
    assert (tmp_path / "m.json.corrupt").exists()
    mem.add("fresh")
    assert json.loads(path.read_text())[0]["text"] == "fresh"


def test_trace_logger_writes_jsonl_with_monotonic_seq(tmp_path):
    t = TraceLogger(tmp_path / "t.jsonl", run_id="r1", clock=lambda: 5.0)
    t.log("a", x=1)
    t.log("b")
    lines = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert [l["seq"] for l in lines] == [0, 1]
    assert lines[0] == {"run_id": "r1", "seq": 0, "ts": 5.0, "type": "a", "data": {"x": 1}}
    assert len(t.of_type("a")) == 1
