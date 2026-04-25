"""Data structures for multi-agent dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentProfile:
    id: str
    name: str
    description: str = ""
    instructions: str = ""
    capabilities: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    tools_allow: tuple[str, ...] = ()
    tools_deny: tuple[str, ...] = ()
    model: str = ""
    provider: str = ""
    max_iterations: int = 8
    max_tokens: int = 4096
    temperature: float = 0.4
    priority: int = 0

    @property
    def is_general(self) -> bool:
        return self.id == "general"


@dataclass
class DispatchCandidate:
    agent_id: str
    score: float
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class DispatchPlan:
    query: str
    task_type: str = "chat"
    strategy: str = "single"  # single | parallel | main
    primary_agent_id: str = "general"
    candidates: list[DispatchCandidate] = field(default_factory=list)
    selected_agent_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""

    @property
    def should_dispatch(self) -> bool:
        return self.strategy in {"single", "parallel"} and bool(self.selected_agent_ids)


@dataclass
class AgentRunResult:
    agent_id: str
    success: bool
    output: str = ""
    error: str = ""
    iterations: int = 0
    model: str = ""
    provider_name: str = ""
    tool_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DispatchResult:
    plan: DispatchPlan
    results: list[AgentRunResult] = field(default_factory=list)
    final_output: str = ""
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
