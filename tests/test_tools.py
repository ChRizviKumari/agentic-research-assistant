from pydantic import BaseModel

from research_agent.tools import Tool, ToolRegistry, TransientToolError


def test_registry_unknown_tool_and_bad_args(registry):
    r = registry.execute("nope", {})
    assert not r.ok and "unknown tool" in r.output
    r = registry.execute("calculator", {})
    assert not r.ok and "invalid arguments" in r.output
    r = registry.execute("search_web", {"query": "x", "top_k": 99})
    assert not r.ok


def test_calculator_tool(registry):
    assert registry.execute("calculator", {"expression": "17*23+4"}).output == "395"
    assert registry.execute("calculator", {"expression": "1/3"}).output == "0.333333333333"
    assert not registry.execute("calculator", {"expression": "'a'"}).ok


def test_python_eval_tool_blocks_escape(registry):
    r = registry.execute("python_eval", {"expression": "__import__('os')"})
    assert not r.ok


def test_search_tool_ranks_relevant_doc_first(registry):
    out = registry.execute("search_web", {"query": "who created the Python language"}).output
    assert out.startswith("[python_language]")
    assert registry.execute("search_web", {"query": "zzzxqj"}).output == "no results"


def test_notes_roundtrip_append_and_traversal(registry):
    assert registry.execute("notes", {"action": "write", "title": "a", "content": "one"}).ok
    assert registry.execute("notes", {"action": "append", "title": "a", "content": "two"}).ok
    assert registry.execute("notes", {"action": "read", "title": "a"}).output == "one\ntwo\n"
    evil = registry.execute("notes", {"action": "write", "title": "../../evil", "content": "x"})
    assert evil.ok and "/" not in evil.output  # slugified, stays inside the notes dir
    assert not registry.execute("notes", {"action": "read", "title": "missing"}).ok
    assert not registry.execute("notes", {"action": "write", "title": "t"}).ok  # content required
    assert "a" in registry.execute("notes", {"action": "list"}).output


class _Args(BaseModel):
    x: int = 0


def test_transient_errors_retried_then_succeed():
    calls = {"n": 0}
    sleeps: list[float] = []

    def flaky(_args):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientToolError("blip")
        return "ok"

    reg = ToolRegistry(max_retries=2, backoff_seconds=1.0, sleep=sleeps.append)
    reg.register(Tool("flaky", "d", _Args, flaky))
    r = reg.execute("flaky", {})
    assert r.ok and r.attempts == 3 and sleeps == [1.0, 2.0]


def test_transient_errors_give_up_and_unexpected_exceptions_contained():
    reg = ToolRegistry(max_retries=1)

    def always(_a):
        raise TransientToolError("down")

    def boom(_a):
        raise RuntimeError("kaboom")

    reg.register(Tool("always", "d", _Args, always))
    reg.register(Tool("boom", "d", _Args, boom))
    r = reg.execute("always", {})
    assert not r.ok and r.attempts == 2
    r = reg.execute("boom", {})
    assert not r.ok and "RuntimeError" in r.output


def test_duplicate_registration_rejected_and_output_clipped():
    reg = ToolRegistry()
    reg.register(Tool("big", "d", _Args, lambda a: "x" * 5000))
    try:
        reg.register(Tool("big", "d", _Args, lambda a: ""))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert reg.execute("big", {}).output.endswith("[truncated]")
    assert "big(x?: integer)" in reg.describe()
