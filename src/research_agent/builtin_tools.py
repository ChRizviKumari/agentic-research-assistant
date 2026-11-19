"""The built-in toolset: calculator, web-search stub, python sandbox and file notes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from research_agent.notes import NotesStore
from research_agent.sandbox import SandboxError, calculator_evaluator, python_evaluator
from research_agent.search import DEFAULT_CORPUS_DIR, LocalCorpus
from research_agent.tools import Tool, ToolError, ToolRegistry


class CalculatorArgs(BaseModel):
    """Arguments for ``calculator``."""

    expression: str = Field(description="arithmetic expression, e.g. '(2+3)*sqrt(16)'")


class SearchArgs(BaseModel):
    """Arguments for ``search_web``."""

    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class PythonArgs(BaseModel):
    """Arguments for ``python_eval``."""

    expression: str = Field(description="single Python expression; no imports/attributes")


class NotesArgs(BaseModel):
    """Arguments for ``notes``."""

    action: Literal["write", "append", "read", "list"]
    title: str | None = None
    content: str | None = None

    @model_validator(mode="after")
    def _check(self) -> NotesArgs:
        if self.action != "list" and not self.title:
            raise ValueError(f"'title' is required for action '{self.action}'")
        if self.action in ("write", "append") and self.content is None:
            raise ValueError(f"'content' is required for action '{self.action}'")
        return self


def _format_number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError("calculator expressions must produce a number")
    return str(value) if isinstance(value, int) else f"{value:.12g}"


def default_registry(
    notes_dir: Path | str,
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    max_retries: int = 2,
) -> ToolRegistry:
    """Build a registry with the four built-in tools.

    Args:
        notes_dir: Directory where the ``notes`` tool stores files.
        corpus_dir: Directory of ``.txt`` / ``.md`` documents searched by ``search_web``.
        max_retries: Retries for transient tool failures.
    """
    corpus = LocalCorpus(corpus_dir)
    store = NotesStore(notes_dir)
    calc, py = calculator_evaluator(), python_evaluator()

    def calculator(args: CalculatorArgs) -> str:
        try:
            return _format_number(calc.evaluate(args.expression))
        except SandboxError as exc:
            raise ToolError(str(exc)) from exc

    def search_web(args: SearchArgs) -> str:
        hits = corpus.search(args.query, args.top_k)
        if not hits:
            return "no results"
        return "\n".join(f"[{h.doc_id}] {h.title} (score {h.score}): {h.snippet}" for h in hits)

    def python_eval(args: PythonArgs) -> str:
        try:
            return repr(py.evaluate(args.expression))
        except SandboxError as exc:
            raise ToolError(str(exc)) from exc

    def notes(args: NotesArgs) -> str:
        if args.action == "list":
            names = store.list()
            return ", ".join(names) if names else "no notes"
        assert args.title is not None
        if args.action == "read":
            return store.read(args.title)
        assert args.content is not None
        name = store.write(args.title, args.content, append=args.action == "append")
        return f"saved {name}"

    registry = ToolRegistry(max_retries=max_retries)
    registry.register(Tool("calculator", "Evaluate a math expression exactly.", CalculatorArgs, calculator))
    registry.register(Tool("search_web", "Search the local knowledge corpus (offline stub).", SearchArgs, search_web))
    registry.register(Tool("python_eval", "Evaluate a sandboxed Python expression.", PythonArgs, python_eval))
    registry.register(Tool("notes", "Write/append/read/list persistent notes.", NotesArgs, notes))
    return registry
