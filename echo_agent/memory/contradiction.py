"""Contradiction detection with versioned memory lattice.

Contradictions are not silently overwritten but stored as temporal edges,
supporting belief revision and history queries.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from echo_agent.memory.types import Contradiction, MemoryEntry

if TYPE_CHECKING:
    from echo_agent.memory.vectors import VectorIndex
    from echo_agent.storage.backend import StorageBackend

_CONTRADICTION_TOOL = {
    "type": "function",
    "function": {
        "name": "check_contradiction",
        "description": "Determine whether two memory entries contradict each other.",
        "parameters": {
            "type": "object",
            "properties": {
                "is_contradictory": {
                    "type": "boolean",
                    "description": "True if the two memories contradict each other.",
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of why they do or do not contradict.",
                },
            },
            "required": ["is_contradictory", "explanation"],
        },
    },
}


class ContradictionDetector:
    """Detects contradictions between memories using semantic similarity + LLM verification.

    Implements a versioned memory lattice: contradictions are not silently overwritten
    but stored as temporal edges, supporting belief revision and history queries.
    """

    SIMILARITY_THRESHOLD = 0.75

    def __init__(
        self,
        storage: StorageBackend,
        vector_index: VectorIndex | None = None,
    ) -> None:
        self._storage = storage
        self._vector_index = vector_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(
        self,
        new_entry: MemoryEntry,
        candidates: list[MemoryEntry],
        llm_call: Callable[..., Any] | None = None,
    ) -> list[Contradiction]:
        """Check new_entry against candidates for contradictions.

        If *llm_call* is provided, uses LLM to verify semantic contradiction.
        Otherwise falls back to heuristic (same key, different content).
        """
        contradictions: list[Contradiction] = []
        for candidate in candidates:
            if candidate.id == new_entry.id:
                continue
            result = await self._check_pair(new_entry, candidate, llm_call)
            if result is not None:
                contradictions.append(result)
        if contradictions:
            logger.info(
                "Detected {} contradiction(s) for memory {}",
                len(contradictions),
                new_entry.id,
            )
        return contradictions

    async def store_contradiction(self, contradiction: Contradiction) -> None:
        """Persist contradiction to storage."""
        await self._storage.execute_sql(
            "INSERT OR REPLACE INTO memory_contradictions"
            "(id, memory_id_a, memory_id_b, description, resolution, resolved_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                contradiction.id,
                contradiction.memory_id_a,
                contradiction.memory_id_b,
                contradiction.description,
                contradiction.resolution,
                contradiction.resolved_at,
                contradiction.created_at,
            ),
        )
        logger.debug("Stored contradiction {}", contradiction.id)

    async def resolve(
        self,
        contradiction_id: str,
        resolution: str,
        winner_id: str | None = None,
    ) -> None:
        """Resolve a contradiction. resolution: 'a_wins', 'b_wins', 'merged', 'user_decided'."""
        now = datetime.now().isoformat()
        await self._storage.execute_sql(
            "UPDATE memory_contradictions SET resolution = ?, resolved_at = ? WHERE id = ?",
            (resolution, now, contradiction_id),
        )
        logger.info("Resolved contradiction {} as '{}'", contradiction_id, resolution)

        if winner_id and resolution in ("a_wins", "b_wins"):
            rows = await self._storage.fetch_sql(
                "SELECT memory_id_a, memory_id_b FROM memory_contradictions WHERE id = ?",
                (contradiction_id,),
            )
            if rows:
                row = rows[0]
                loser_id = (
                    row["memory_id_b"] if winner_id == row["memory_id_a"] else row["memory_id_a"]
                )
                await self._storage.execute_sql(
                    "UPDATE memories SET superseded_by = ? WHERE id = ?",
                    (winner_id, loser_id),
                )

    async def get_unresolved(self, limit: int = 10) -> list[Contradiction]:
        """Get unresolved contradictions."""
        rows = await self._storage.fetch_sql(
            "SELECT * FROM memory_contradictions WHERE resolution IS NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [Contradiction.from_dict(r) for r in rows]

    async def get_history(self, memory_id: str) -> list[Contradiction]:
        """Get all contradictions involving a memory."""
        rows = await self._storage.fetch_sql(
            "SELECT * FROM memory_contradictions "
            "WHERE memory_id_a = ? OR memory_id_b = ? "
            "ORDER BY created_at DESC",
            (memory_id, memory_id),
        )
        return [Contradiction.from_dict(r) for r in rows]

    async def supersede(
        self,
        old_entry: MemoryEntry,
        new_entry: MemoryEntry,
        store: Any,
    ) -> None:
        """Mark old_entry as superseded by new_entry in the version lattice."""
        old_entry.superseded_by = new_entry.id
        new_entry.version = old_entry.version + 1
        await store.update(old_entry)
        await store.update(new_entry)
        logger.info(
            "Memory {} superseded by {} (v{})",
            old_entry.id,
            new_entry.id,
            new_entry.version,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _check_pair(
        self,
        new_entry: MemoryEntry,
        candidate: MemoryEntry,
        llm_call: Callable[..., Any] | None,
    ) -> Contradiction | None:
        if llm_call is not None:
            return await self._llm_check(new_entry, candidate, llm_call)
        return self._heuristic_check(new_entry, candidate)

    def _heuristic_check(
        self, a: MemoryEntry, b: MemoryEntry
    ) -> Contradiction | None:
        """Same key but different content implies contradiction."""
        if a.key and a.key == b.key and a.content.strip() != b.content.strip():
            return Contradiction(
                id=uuid.uuid4().hex[:12],
                memory_id_a=a.id,
                memory_id_b=b.id,
                description=f"Key '{a.key}' has conflicting content.",
            )
        return None

    async def _llm_check(
        self,
        a: MemoryEntry,
        b: MemoryEntry,
        llm_call: Callable[..., Any],
    ) -> Contradiction | None:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a contradiction detector. Determine whether the two "
                    "memory entries below contradict each other. Use the provided tool "
                    "to report your finding."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Memory A (key={a.key!r}):\n{a.content}\n\n"
                    f"Memory B (key={b.key!r}):\n{b.content}"
                ),
            },
        ]
        try:
            response = await llm_call(
                messages=messages,
                tools=[_CONTRADICTION_TOOL],
                tool_choice={"type": "function", "function": {"name": "check_contradiction"}},
            )
            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)
            if args.get("is_contradictory"):
                return Contradiction(
                    id=uuid.uuid4().hex[:12],
                    memory_id_a=a.id,
                    memory_id_b=b.id,
                    description=args.get("explanation", "LLM detected contradiction."),
                )
        except Exception:
            logger.opt(exception=True).warning(
                "LLM contradiction check failed for {} vs {}, falling back to heuristic",
                a.id,
                b.id,
            )
            return self._heuristic_check(a, b)
        return None
