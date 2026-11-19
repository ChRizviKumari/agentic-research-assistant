"""research_agent: a small, dependency-light ReAct-style agent framework."""

from research_agent.agent import Agent, AgentConfig, AgentResult
from research_agent.budget import Budget, BudgetExceeded
from research_agent.llm import (
    AnthropicLLM,
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    MockLLM,
    ScriptedLLM,
    TransientLLMError,
)
from research_agent.memory import LongTermMemory, ShortTermMemory
from research_agent.tools import Tool, ToolError, ToolRegistry, ToolResult, TransientToolError
from research_agent.trace import TraceLogger

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AnthropicLLM",
    "Budget",
    "BudgetExceeded",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "LongTermMemory",
    "Message",
    "MockLLM",
    "ScriptedLLM",
    "ShortTermMemory",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "TraceLogger",
    "TransientLLMError",
    "TransientToolError",
]
