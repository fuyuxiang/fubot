"""Structured records for workflow orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    """Return an audit-friendly UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentRouteDecision:
    """Auditable routing decision for one executor turn."""

    agent_id: str
    agent_name: str
    agent_role: str
    task_type: str
    model: str
    provider: str | None
    reason: str
    fallback_chain: list[str] = field(default_factory=list)
    health_score: float = 1.0
    current_load: int = 0
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRecord:
    """A unit of work scheduled by the coordinator."""

    id: str
    workflow_id: str
    title: str
    kind: str
    status: str = "pending"
    assigned_agent_id: str | None = None
    assigned_agent_name: str | None = None
    route: dict[str, Any] | None = None
    summary: str = ""
    result: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssignmentRecord:
    """Executor assignment for a task."""

    id: str
    workflow_id: str
    task_id: str
    agent_id: str
    agent_name: str
    agent_role: str
    status: str = "assigned"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionLogRecord:
    """Append-only workflow execution log."""

    workflow_id: str
    task_id: str | None
    agent_id: str
    agent_name: str
    agent_role: str
    message_kind: str
    content: str
    is_final: bool = False
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowRecord:
    """Workflow state persisted across restarts."""

    id: str
    session_key: str
    channel: str
    chat_id: str
    user_message: str
    status: str = "running"
    task_ids: list[str] = field(default_factory=list)
    assignment_ids: list[str] = field(default_factory=list)
    shared_board: dict[str, Any] = field(default_factory=dict)
    execution_logs: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
