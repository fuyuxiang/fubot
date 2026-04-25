from __future__ import annotations
import json
import uuid
from datetime import datetime
from typing import Any, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from echo_agent.storage.backend import StorageBackend

from echo_agent.memory.types import MemoryEntry, MemoryTier, MemoryType, Episode


class WorkingMemory:
    """In-process buffer for current conversation context. Not persisted."""

    def __init__(self, max_entries: int = 20):
        self._max = max_entries
        self._entries: list[MemoryEntry] = []

    def load(self, entries: list[MemoryEntry]) -> None:
        """Replace working memory with given entries."""
        self._entries = entries[:self._max]

    def add(self, entry: MemoryEntry) -> None:
        """Add entry, evicting oldest if at capacity."""
        if len(self._entries) >= self._max:
            self._entries.pop(0)
        entry.tier = MemoryTier.WORKING
        self._entries.append(entry)

    def get_context(self, max_chars: int = 2000) -> str:
        """Render working memory as markdown for prompt injection."""
        if not self._entries:
            return ""
        lines = []
        used = 0
        for entry in reversed(self._entries):
            line = f"- {entry.key}: {entry.content}" if entry.key else f"- {entry.content}"
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(reversed(lines))

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    @property
    def count(self) -> int:
        return len(self._entries)


class EpisodicManager:
    """Manages conversation episodes with temporal indexing."""

    def __init__(self, storage: StorageBackend):
        self._storage = storage

    async def create_episode(
        self,
        session_key: str,
        messages: list[dict[str, Any]],
        summary: str,
        entity_ids: list[str] | None = None,
        importance: float = 0.5,
        message_range: tuple[int, int] = (0, 0),
    ) -> Episode:
        episode = Episode(
            id=uuid.uuid4().hex[:12],
            session_key=session_key,
            summary=summary,
            message_range_start=message_range[0],
            message_range_end=message_range[1],
            entity_ids=entity_ids or [],
            importance=importance,
            created_at=datetime.now().isoformat(),
        )
        await self._storage.execute_sql(
            "INSERT OR REPLACE INTO memory_episodes (id, session_key, summary, "
            "message_range_start, message_range_end, entities, importance, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode.id, episode.session_key, episode.summary,
                episode.message_range_start, episode.message_range_end,
                json.dumps(episode.entity_ids), episode.importance, episode.created_at,
            ),
        )
        logger.debug("Created episode {} for session {}", episode.id, session_key)
        return episode

    async def search_episodes(
        self, query: str, session_key: str | None = None, limit: int = 5,
    ) -> list[Episode]:
        if session_key:
            rows = await self._storage.fetch_sql(
                "SELECT * FROM memory_episodes WHERE session_key = ? "
                "AND summary LIKE ? ORDER BY created_at DESC LIMIT ?",
                (session_key, f"%{query}%", limit),
            )
        else:
            rows = await self._storage.fetch_sql(
                "SELECT * FROM memory_episodes WHERE summary LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit),
            )
        return [self._row_to_episode(r) for r in rows]

    async def get_session_episodes(
        self, session_key: str, limit: int = 50,
    ) -> list[Episode]:
        rows = await self._storage.fetch_sql(
            "SELECT * FROM memory_episodes WHERE session_key = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_key, limit),
        )
        return [self._row_to_episode(r) for r in rows]

    async def get_recent(self, limit: int = 20) -> list[Episode]:
        rows = await self._storage.fetch_sql(
            "SELECT * FROM memory_episodes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_episode(r) for r in rows]

    @staticmethod
    def _row_to_episode(row: dict[str, Any]) -> Episode:
        entities = row.get("entities", "[]")
        if isinstance(entities, str):
            entities = json.loads(entities)
        return Episode(
            id=row["id"],
            session_key=row.get("session_key", ""),
            summary=row.get("summary", ""),
            message_range_start=row.get("message_range_start", 0),
            message_range_end=row.get("message_range_end", 0),
            entity_ids=entities,
            importance=row.get("importance", 0.5),
            created_at=row.get("created_at", ""),
        )


class SemanticManager:
    """Manages distilled facts -- the core persistent memory tier.

    Wraps existing MemoryStore CRUD with tier filtering.
    """

    def __init__(self, store: Any):
        self._store = store

    def get_semantic_entries(
        self,
        mem_type: MemoryType | None = None,
        session_key: str | None = None,
    ) -> list[MemoryEntry]:
        entries = self._store.list_all(mem_type=mem_type, session_key=session_key)
        return [e for e in entries if e.tier == MemoryTier.SEMANTIC]

    async def promote_from_episodic(
        self, episode: Episode, extracted_facts: list[dict[str, Any]],
    ) -> list[MemoryEntry]:
        """Promote extracted facts from an episode to semantic memory."""
        promoted: list[MemoryEntry] = []
        for fact in extracted_facts:
            entry = MemoryEntry(
                type=MemoryType(fact.get("type", "environment")),
                tier=MemoryTier.SEMANTIC,
                key=fact.get("key", ""),
                content=fact.get("content", ""),
                tags=fact.get("tags", []),
                importance=fact.get("importance", 0.5),
                episode_id=episode.id,
            )
            try:
                result = self._store.add(entry)
                promoted.append(result)
            except ValueError as e:
                logger.warning(
                    "Failed to promote fact from episode {}: {}", episode.id, e,
                )
        if promoted:
            logger.info("Promoted {} facts from episode {}", len(promoted), episode.id)
        return promoted


class ArchivalManager:
    """Compressed long-term storage for old/low-importance memories."""

    def __init__(self, storage: StorageBackend):
        self._storage = storage

    async def archive(self, entries: list[MemoryEntry]) -> int:
        """Move entries to archival tier. Returns count archived."""
        count = 0
        for entry in entries:
            entry.tier = MemoryTier.ARCHIVAL
            entry.updated_at = datetime.now().isoformat()
            await self._storage.execute_sql(
                "UPDATE memories SET data = ?, updated_at = ? WHERE id = ?",
                (
                    json.dumps(entry.to_dict(), ensure_ascii=False),
                    entry.updated_at,
                    entry.id,
                ),
            )
            count += 1
        if count:
            logger.info("Archived {} memories", count)
        return count

    async def search_archival(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        rows = await self._storage.fetch_sql(
            "SELECT data FROM memories "
            "WHERE json_extract(data, '$.tier') = 'archival' AND data LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        )
        results = []
        for row in rows:
            data = row.get("data", "{}")
            if isinstance(data, str):
                data = json.loads(data)
            results.append(MemoryEntry.from_dict(data))
        return results

    async def delete_forgotten(self, entries: list[MemoryEntry]) -> int:
        count = 0
        for entry in entries:
            await self._storage.execute_sql(
                "DELETE FROM memories WHERE id = ?", (entry.id,),
            )
            if entry.embedding_id:
                await self._storage.execute_sql(
                    "DELETE FROM vectors WHERE id = ?", (entry.embedding_id,),
                )
            count += 1
        if count:
            logger.info("Deleted {} forgotten memories", count)
        return count
