"""Core domain — pure logic and port definitions.

This package contains zero infrastructure imports. Everything here
depends only on stdlib, other core modules, and the Tool Protocol
in ``thesaurus.tools.base`` (which is itself a port definition
with zero infrastructure deps).

The public API for external consumers (e.g. ``thesaurus_web``):

    from thesaurus.core import Agent, AgentCallbacks, LLMClient
"""

from .agent import Agent, AgentCallbacks
from .ports import AgentResponse, LLMClient, ToolCall

__all__ = [
    "Agent",
    "AgentCallbacks",
    "AgentResponse",
    "LLMClient",
    "ToolCall",
]
