"""Memory type definitions — enums, dataclasses for the four-tier memory hierarchy."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryTier(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ARCHIVAL = "archival"


class MemoryType(str, Enum):
    USER = "user"
    ENVIRONMENT = "environment"


@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: MemoryType = MemoryType.USER
    tier: MemoryTier = MemoryTier.SEMANTIC
    key: str = ""
    content: str = ""
    tags: list[str] = field(default_factory=list)
    source_session: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5
    access_count: int = 0
    last_accessed: str = ""
    embedding_id: str = ""
    episode_id: str = ""
    version: int = 1
    superseded_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "tier": self.tier.value,
            "key": self.key,
            "content": self.content,
            "tags": self.tags,
            "source_session": self.source_session,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "embedding_id": self.embedding_id,
            "episode_id": self.episode_id,
            "version": self.version,
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            type=MemoryType(data.get("type", "user")),
            tier=MemoryTier(data.get("tier", "semantic")),
            key=data.get("key", ""),
            content=data.get("content", ""),
            tags=list(data.get("tags", [])),
            source_session=data.get("source_session", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", ""),
            embedding_id=data.get("embedding_id", ""),
            episode_id=data.get("episode_id", ""),
            version=data.get("version", 1),
            superseded_by=data.get("superseded_by", ""),
        )

    def effective_importance(self, decay_half_life_days: float = 30.0) -> float:
        if not self.last_accessed or decay_half_life_days <= 0:
            return self.importance
        try:
            last = datetime.fromisoformat(self.last_accessed)
            days = (datetime.now() - last).total_seconds() / 86400
            half_life = decay_half_life_days * (1 + math.log2(1 + self.access_count))
            decay = math.pow(0.5, days / half_life)
            return self.importance * decay
        except (ValueError, OverflowError):
            return self.importance

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()

    @property
    def is_superseded(self) -> bool:
        return bool(self.superseded_by)


@dataclass
class Episode:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_key: str = ""
    summary: str = ""
    message_range_start: int = 0
    message_range_end: int = 0
    entity_ids: list[str] = field(default_factory=list)
    importance: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_key": self.session_key,
            "summary": self.summary,
            "message_range_start": self.message_range_start,
            "message_range_end": self.message_range_end,
            "entity_ids": self.entity_ids,
            "importance": self.importance,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            session_key=data.get("session_key", ""),
            summary=data.get("summary", ""),
            message_range_start=data.get("message_range_start", 0),
            message_range_end=data.get("message_range_end", 0),
            entity_ids=list(data.get("entity_ids", [])),
            importance=data.get("importance", 0.5),
            created_at=data.get("created_at", ""),
        )

@dataclass
class Contradiction:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    memory_id_a: str = ""
    memory_id_b: str = ""
    description: str = ""
    resolution: str | None = None
    resolved_at: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "memory_id_a": self.memory_id_a,
            "memory_id_b": self.memory_id_b, "description": self.description,
            "resolution": self.resolution, "resolved_at": self.resolved_at,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Contradiction:
        return cls(
            id=data.get("id", ""), memory_id_a=data.get("memory_id_a", ""),
            memory_id_b=data.get("memory_id_b", ""),
            description=data.get("description", ""),
            resolution=data.get("resolution"), resolved_at=data.get("resolved_at"),
            created_at=data.get("created_at", ""),
        )
