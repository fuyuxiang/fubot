"""Multi-agent intent routing and dispatch runtime."""

from echo_agent.agent.multi_agent.models import (
    AgentProfile,
    AgentRunResult,
    DispatchCandidate,
    DispatchPlan,
    DispatchResult,
)
from echo_agent.agent.multi_agent.registry import AgentRegistry
from echo_agent.agent.multi_agent.router import IntentRouter
from echo_agent.agent.multi_agent.runtime import MultiAgentRuntime

__all__ = [
    "AgentProfile",
    "AgentRunResult",
    "DispatchCandidate",
    "DispatchPlan",
    "DispatchResult",
    "AgentRegistry",
    "IntentRouter",
    "MultiAgentRuntime",
]
