import json

import pytest

from research_agent import (
    Agent, AgentConfig, AnthropicLLM, LLMError, LongTermMemory, MockLLM, TransientLLMError,
)
from research_agent.protocol import ProtocolError, act, final, parse_step, plan_json


def make(registry, script, **cfg):
    llm = MockLLM(script, repeat_last=cfg.pop("repeat_last", False))
    return Agent(llm, registry, AgentConfig(**cfg)), llm


def test_happy_path_plan_then_tool_then_answer(registry):
    agent, llm = make(registry, [plan_json("calc"), act("calculator", expression="6*7"), final("42")])
    res = agent.run("6*7?")
    assert res.ok and res.answer == "42" and res.steps == 2
    assert res.plan == ["calc"] and res.tool_calls() == ["calculator"]
    # observation was fed back to the model
    assert "Observation (tool=calculator, ok=true): 42" in llm.calls[-1][-1].content
    assert res.tokens > 0


def test_planner_garbage_falls_back_to_no_plan(registry):
    agent, _ = make(registry, ["not json", final("done")])
    res = agent.run("t")
    assert res.ok and res.plan == []
    assert any(e["type"] == "plan_fallback" for e in res.trace)


def test_tool_error_is_shown_to_model_and_recovered(registry):
    agent, llm = make(
        registry, [plan_json("x"), act("calculator", expression="1 +"), act("calculator", expression="1+1"), final("2")]
    )
    res = agent.run("t")
    assert res.ok and res.tool_errors() == 1
    assert "ok=false" in llm.calls[2][-1].content


def test_malformed_reply_gets_corrective_prompt(registry):
    agent, llm = make(registry, [plan_json("x"), "hello", '{"thought": "both", "final_answer": "a", "action": {"tool": "notes"}}', final("ok")])
    res = agent.run("t")
    assert res.ok
    assert len([e for e in res.trace if e["type"] == "parse_error"]) == 2
    assert "invalid" in llm.calls[-1][-1].content


def test_parse_retries_exhausted_fails_cleanly(registry):
    agent, _ = make(registry, [plan_json("x"), "junk"], repeat_last=True, max_parse_retries=1)
    res = agent.run("t")
    assert res.status == "failed" and "valid reply" in res.answer


def test_transient_llm_error_retried_with_backoff(registry):
    sleeps = []
    llm = MockLLM([plan_json("x"), TransientLLMError("429"), TransientLLMError("429"), final("fine")])
    agent = Agent(llm, registry, AgentConfig(backoff_seconds=0.5), sleep=sleeps.append)
    res = agent.run("t")
    assert res.ok and sleeps == [0.5, 1.0]
    assert len([e for e in res.trace if e["type"] == "llm_retry"]) == 2


def test_persistent_llm_failure_reports_failed(registry):
    llm = MockLLM([TransientLLMError("down")], repeat_last=True)
    res = Agent(llm, registry, AgentConfig(max_llm_retries=1, use_planner=False), sleep=lambda s: None).run("t")
    assert res.status == "failed" and "LLM error" in res.answer


def test_fatal_llm_error_not_retried(registry):
    llm = MockLLM([LLMError("bad key")])
    res = Agent(llm, registry, AgentConfig(use_planner=False)).run("t")
    assert res.status == "failed" and len(llm.calls) == 1


def test_step_budget_enforced(registry):
    script = [act("calculator", expression=f"{i}+1") for i in range(10)]
    agent, _ = make(registry, script, max_steps=3, use_planner=False)
    res = agent.run("t")
    assert res.status == "budget_exceeded" and res.steps == 3


def test_token_budget_enforced(registry):
    agent, _ = make(registry, [act("calculator", expression="1+1"), final("x")], max_tokens=50, use_planner=False)
    res = agent.run("t")
    assert res.status == "budget_exceeded" and "token" in res.answer


def test_repeated_identical_calls_detected_as_stuck(registry):
    agent, _ = make(registry, [act("calculator", expression="1+1")], repeat_last=True, use_planner=False)
    res = agent.run("t")
    assert res.status == "stuck"


def test_trace_file_is_structured_jsonl(registry, tmp_path):
    path = tmp_path / "logs" / "trace.jsonl"
    llm = MockLLM([plan_json("x"), act("calculator", expression="2+2"), final("4")])
    res = Agent(llm, registry, trace_path=path).run("t")
    events = [json.loads(l) for l in path.read_text().splitlines()]
    assert [e["type"] for e in events][0] == "run_start" and events[-1]["type"] == "run_end"
    assert events[-1]["data"]["status"] == "success"
    assert len({e["run_id"] for e in events}) == 1 and len(events) == len(res.trace)


def test_long_term_memory_recalled_and_stored(registry, tmp_path):
    mem = LongTermMemory(tmp_path / "m.json")
    mem.add("Everest elevation is 8849 metres")
    llm = MockLLM([final("8849")])
    agent = Agent(llm, registry, AgentConfig(use_planner=False, remember_results=True), long_term=mem)
    agent.run("What is the Everest elevation?")
    assert "Relevant memories" in llm.calls[0][1].content and "8849" in llm.calls[0][1].content
    assert len(mem) == 2  # recalled one + stored the new Q/A


def test_empty_long_term_memory_still_stores_result(registry, tmp_path):
    # regression: an empty memory is falsy via __len__; must still be used
    mem = LongTermMemory(tmp_path / "m.json")
    Agent(MockLLM([final("a")]), registry, AgentConfig(use_planner=False, remember_results=True), long_term=mem).run("q")
    assert len(mem) == 1


def test_protocol_parsing():
    assert parse_step('Sure:\n```json\n{"final_answer": "x"}\n```').final_answer == "x"
    with pytest.raises(ProtocolError):
        parse_step('{"thought": "neither"}')
    with pytest.raises(ProtocolError):
        parse_step("no braces")


def test_scripted_llm_exhaustion_and_reactive_steps():
    llm = MockLLM([lambda msgs: f"n={len(msgs)}"])
    assert llm.complete([]).text == "n=0"
    with pytest.raises(LLMError):
        llm.complete([])


def test_anthropic_llm_requires_key_and_model(monkeypatch):
    # Real API calls are NOT exercised; only the guard rails that run before any network use.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM(model="m")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    with pytest.raises(LLMError, match="model"):
        AnthropicLLM()
