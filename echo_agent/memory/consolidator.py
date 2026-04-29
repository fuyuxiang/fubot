"""Memory consolidator — summarizes conversation chunks into long-term memory."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from loguru import logger

from echo_agent.memory.store import MemoryStore
from echo_agent.memory.types import MemoryTier

_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "Summary paragraph starting with [YYYY-MM-DD HH:MM].",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class MemoryConsolidator:
    """Consolidates conversation history into MEMORY.md + HISTORY.md via LLM."""

    _MAX_ROUNDS = 3

    def __init__(
        self,
        memory_store: MemoryStore,
        llm_call: Callable[..., Awaitable[Any]],
        context_window_tokens: int = 65536,
        consolidation_threshold: int = 50,
    ):
        self.store = memory_store
        self._llm_call = llm_call
        self.context_window_tokens = context_window_tokens
        self._consolidation_threshold = consolidation_threshold
        self._episodic_manager = None  # set via set_episodic_manager()
        self._semantic_manager = None
        self._forgetting_curve = None
        self._contradiction_detector = None
        self._archival_manager = None

    def set_episodic_manager(self, mgr):
        self._episodic_manager = mgr

    def set_semantic_manager(self, mgr):
        self._semantic_manager = mgr

    def set_forgetting_curve(self, curve):
        self._forgetting_curve = curve

    def set_contradiction_detector(self, detector):
        self._contradiction_detector = detector

    def set_archival_manager(self, mgr):
        self._archival_manager = mgr

    async def consolidate_chunk(self, messages: list[dict[str, Any]]) -> bool:
        if not messages:
            return True

        current_memory = self.store.read_long_term()
        formatted = self._format_messages(messages)
        prompt = (
            "Process this conversation and call save_memory with your consolidation.\n\n"
            f"## Current Long-term Memory\n{current_memory or '(empty)'}\n\n"
            f"## Conversation to Process\n{formatted}"
        )

        try:
            response = await self._llm_call(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Call save_memory."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                tool_choice={"type": "function", "function": {"name": "save_memory"}},
            )

            if not response.tool_calls:
                logger.warning("Consolidation: LLM did not call save_memory")
                return False

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)

            history_entry = args.get("history_entry", "")
            memory_update = args.get("memory_update", "")

            if history_entry:
                self.store.append_history(history_entry)
            if memory_update:
                self.store.write_long_term(memory_update)

            logger.info("Memory consolidation complete: {} chars history, {} chars memory",
                        len(history_entry), len(memory_update))
            return True
        except ValueError as e:
            logger.warning("Memory consolidation rejected unsafe content: {}", e)
            return False
        except Exception as e:
            logger.error("Memory consolidation failed: {}", e)
            return False

    async def sleep_consolidate(self, session_key: str, messages: list[dict[str, Any]]) -> dict[str, int]:
        """Sleep-time consolidation pipeline:
        1. Create episode from messages
        2. Extract semantic facts and promote
        3. Run heuristic contradiction detection on promoted entries
        4. Run forgetting/archival pass
        Returns stats dict.
        """
        stats = {"episodes": 0, "promoted": 0, "contradictions": 0, "archived": 0, "forgotten": 0}
        promoted: list = []

        # Step 1: Create episode
        if self._episodic_manager and messages:
            summary_result = await self.consolidate_chunk(messages)
            if summary_result:
                current_memory = self.store.read_long_term()
                from echo_agent.memory.types import Episode
                episode = await self._episodic_manager.create_episode(
                    session_key=session_key,
                    messages=messages,
                    summary=current_memory[:500] if current_memory else "conversation episode",
                    message_range=(0, len(messages)),
                )
                stats["episodes"] = 1

                # Step 2: Promote to semantic
                if self._semantic_manager:
                    try:
                        response = await self._llm_call(
                            messages=[
                                {"role": "system", "content": "Extract key facts from this episode summary. Return a JSON array of objects with keys: type (user/environment), key, content, importance (0-1)."},
                                {"role": "user", "content": episode.summary},
                            ],
                        )
                        if response.content:
                            import json as _json
                            try:
                                facts = _json.loads(response.content)
                                if isinstance(facts, list):
                                    promoted = await self._semantic_manager.promote_from_episodic(episode, facts)
                                    stats["promoted"] = len(promoted)
                            except _json.JSONDecodeError:
                                pass
                    except Exception as e:
                        logger.warning("Fact extraction failed: {}", e)

        # Step 3: Run heuristic contradiction detection on newly promoted entries
        if self._contradiction_detector and promoted:
            try:
                all_entries = list(self.store._entries.values())
                for new_entry in promoted:
                    others = [e for e in all_entries if e.id != new_entry.id]
                    contradictions = await self._contradiction_detector.check(new_entry, others)
                    for c in contradictions:
                        await self._contradiction_detector.store_contradiction(c)
                        stats["contradictions"] += 1
            except Exception as e:
                logger.warning("Contradiction detection failed: {}", e)

        # Step 4: Run forgetting pass
        if self._forgetting_curve:
            all_entries = list(self.store._entries.values())
            to_archive, to_forget = await self._forgetting_curve.run_decay_pass(all_entries)
            if to_archive and self._archival_manager:
                stats["archived"] = await self._archival_manager.archive(to_archive)
            if to_forget and self._archival_manager:
                stats["forgotten"] = await self._archival_manager.delete_forgotten(to_forget)

        logger.info("Sleep consolidation complete: {}", stats)
        return stats

    def should_consolidate(self, session_message_count: int, last_consolidated: int) -> bool:
        unconsolidated = session_message_count - last_consolidated
        return unconsolidated >= self._consolidation_threshold

    def pick_boundary(self, messages: list[dict[str, Any]], start: int, target_tokens: int) -> int | None:
        """Find a safe consolidation boundary (end of a user turn)."""
        tokens = 0
        last_user_idx = None
        for i in range(start, len(messages)):
            content = messages[i].get("content", "")
            tokens += len(str(content)) // 3
            if messages[i].get("role") == "user":
                last_user_idx = i
            if tokens >= target_tokens and last_user_idx is not None:
                return last_user_idx
        return last_user_idx

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        lines = []
        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue
            ts = msg.get("timestamp", "?")[:16]
            role = msg.get("role", "?").upper()
            lines.append(f"[{ts}] {role}: {content}")
        return "\n".join(lines)
