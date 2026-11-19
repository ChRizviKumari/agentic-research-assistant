"""Step and token budgets that bound an agent run."""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(Exception):
    """Raised when a run would exceed its step or token allowance."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class Budget:
    """Tracks consumption of reasoning steps and LLM tokens.

    Args:
        max_steps: Maximum number of ReAct iterations.
        max_tokens: Maximum total (input + output) tokens across all LLM calls in the run.
    """

    max_steps: int = 12
    max_tokens: int = 20_000
    steps_used: int = 0
    tokens_used: int = 0

    def start_step(self) -> None:
        """Begin a new step, raising :class:`BudgetExceeded` if none remain."""
        if self.steps_used >= self.max_steps:
            raise BudgetExceeded("steps", f"step budget of {self.max_steps} exhausted")
        self.steps_used += 1

    def charge(self, tokens: int) -> None:
        """Record token usage, raising :class:`BudgetExceeded` once the allowance is passed."""
        self.tokens_used += tokens
        if self.tokens_used > self.max_tokens:
            raise BudgetExceeded(
                "tokens", f"token budget exceeded ({self.tokens_used}/{self.max_tokens})"
            )

    @property
    def steps_remaining(self) -> int:
        """Steps still available."""
        return max(0, self.max_steps - self.steps_used)

    @property
    def tokens_remaining(self) -> int:
        """Tokens still available."""
        return max(0, self.max_tokens - self.tokens_used)
