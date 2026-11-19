"""LLM provider interface plus a deterministic scripted provider and an optional Anthropic client.

The agent only depends on the :class:`LLMProvider` protocol, so tests and demos run fully offline
with :class:`ScriptedLLM`. :class:`AnthropicLLM` is a thin real-provider adapter that is **not
exercised by the test suite or demo** (no network / API key is available there).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable


@dataclass(frozen=True)
class Message:
    """A single chat message. ``role`` is one of ``system``, ``user`` or ``assistant``."""

    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Text returned by a provider along with token accounting."""

    text: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens


class LLMError(Exception):
    """Non-recoverable provider failure."""


class TransientLLMError(LLMError):
    """Provider failure that is worth retrying (rate limit, timeout, connection reset)."""


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token). Used by providers without usage data."""
    return max(1, (len(text) + 3) // 4)


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can turn a message list into a text completion."""

    def complete(self, messages: Sequence[Message]) -> LLMResponse:  # pragma: no cover - protocol
        """Return the model's reply for ``messages``."""
        ...


ScriptStep = Union[str, Callable[[Sequence[Message]], str], Exception]


class ScriptedLLM:
    """Deterministic provider that replays a fixed script.

    Each script item is consumed per call and may be:

    * ``str`` - returned verbatim,
    * ``Callable[[messages], str]`` - lets a script react to the conversation (e.g. read the last
      observation),
    * an ``Exception`` instance - raised, to simulate provider failures.

    Args:
        script: Items to replay in order.
        repeat_last: If true, the final item is reused forever instead of raising when the script
            runs out (handy for simulating a model that never finishes).
    """

    def __init__(self, script: Iterable[ScriptStep], repeat_last: bool = False) -> None:
        self._script: list[ScriptStep] = list(script)
        self._index = 0
        self._repeat_last = repeat_last
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message]) -> LLMResponse:
        """Return the next scripted reply, recording the prompt in :attr:`calls`."""
        self.calls.append(list(messages))
        if self._index >= len(self._script):
            if self._repeat_last and self._script:
                step = self._script[-1]
            else:
                raise LLMError("ScriptedLLM script exhausted")
        else:
            step = self._script[self._index]
            self._index += 1
        if isinstance(step, Exception):
            raise step
        text = step(messages) if callable(step) else step
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        return LLMResponse(text=text, input_tokens=prompt_tokens, output_tokens=estimate_tokens(text))


#: Alias: the deterministic scripted provider is the project's "MockLLM" for tests and the demo.
MockLLM = ScriptedLLM


class AnthropicLLM:
    """Adapter for the Anthropic Messages API.

    NOTE: Not exercised by the tests or the demo - it needs the optional ``anthropic`` package, a
    network connection and a valid key. The key is read from the ``ANTHROPIC_API_KEY`` environment
    variable (or passed explicitly); it is never stored in the repository. The model name has no
    default on purpose: pass ``model=`` or set ``ANTHROPIC_MODEL``.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        chosen = model or os.environ.get("ANTHROPIC_MODEL")
        if not chosen:
            raise LLMError("Pass model=... or set ANTHROPIC_MODEL")
        try:
            import anthropic  # imported lazily so the package works without the SDK
        except ImportError as exc:
            raise LLMError("The 'anthropic' package is required: pip install anthropic") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=key)
        self.model = chosen
        self.max_tokens = max_tokens

    def complete(self, messages: Sequence[Message]) -> LLMResponse:
        """Call the Messages API; rate-limit/connection errors become :class:`TransientLLMError`."""
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        chat = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system, messages=chat
            )
        except (self._anthropic.RateLimitError, self._anthropic.APIConnectionError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except self._anthropic.APIError as exc:
            raise LLMError(str(exc)) from exc
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, resp.usage.input_tokens, resp.usage.output_tokens)
