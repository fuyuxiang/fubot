"""A2A protocol data models — Agent Card, Task, Message, Artifact."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class A2AMessage:
    role: str  # "user" or "agent"
    parts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def text(cls, role: str, content: str) -> A2AMessage:
        return cls(role=role, parts=[{"type": "text", "text": content}])

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "parts": self.parts}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2AMessage:
        return cls(role=data.get("role", "user"), parts=data.get("parts", []))

    @property
    def text_content(self) -> str:
        return " ".join(p.get("text", "") for p in self.parts if p.get("type") == "text")


@dataclass
class Artifact:
    name: str = ""
    content_type: str = "text/plain"
    data: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "contentType": self.content_type, "data": self.data}


@dataclass
class A2ATask:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    state: TaskState = TaskState.SUBMITTED
    messages: list[A2AMessage] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "state": self.state.value,
            "messages": [m.to_dict() for m in self.messages],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2ATask:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:16]),
            state=TaskState(data.get("state", "submitted")),
            messages=[A2AMessage.from_dict(m) for m in data.get("messages", [])],
            artifacts=[],
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentCard:
    name: str = "echo-agent"
    description: str = "A modular AI agent framework"
    url: str = ""
    version: str = "0.1.0"
    capabilities: list[str] = field(default_factory=lambda: ["chat", "tool_use"])
    skills: list[str] = field(default_factory=list)
    authentication: dict[str, Any] = field(default_factory=lambda: {"schemes": ["bearer"]})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "url": self.url, "version": self.version,
            "capabilities": {"streaming": True, "pushNotifications": False},
            "skills": [{"id": s, "name": s} for s in self.skills],
            "authentication": self.authentication,
        }
