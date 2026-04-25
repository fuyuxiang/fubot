"""Predictive pre-fetch — anticipatory working memory loading.

Before each conversation turn, predicts likely retrieval needs from recent
messages and pre-loads relevant memories into working memory.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from echo_agent.memory.retrieval import HybridRetriever
    from echo_agent.memory.tiers import WorkingMemory


_PREDICT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "predict_topics",
            "description": "Predict topics the user is likely to discuss next.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-5 predicted topic keywords or phrases",
                    },
                },
                "required": ["topics"],
            },
        },
    }
]


class PredictivePrefetch:
    """Predicts next-turn memory needs and pre-loads into working memory.

    Uses a lightweight LLM call on the last N messages to predict likely topics,
    then runs hybrid retrieval on those topics and loads results into working memory.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_call: Callable[..., Awaitable[Any]],
        max_recent_messages: int = 5,
        max_prefetch: int = 8,
    ):
        self._retriever = retriever
        self._llm_call = llm_call
        self._max_recent = max_recent_messages
        self._max_prefetch = max_prefetch

    async def prefetch(
        self,
        recent_messages: list[dict[str, Any]],
        working_memory: WorkingMemory,
        session_key: str = "",
    ) -> int:
        """Predict and pre-load memories. Returns count of memories loaded."""
        if len(recent_messages) < 2:
            return 0

        messages_slice = recent_messages[-self._max_recent :]
        formatted = "\n".join(
            f"{m.get('role', '?')}: {str(m.get('content', ''))[:200]}"
            for m in messages_slice
        )

        try:
            response = await self._llm_call(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Predict 2-5 topics the user will likely discuss next "
                            "based on the conversation. Call predict_topics."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Recent conversation:\n{formatted}",
                    },
                ],
                tools=_PREDICT_TOOL,
                tool_choice={
                    "type": "function",
                    "function": {"name": "predict_topics"},
                },
            )
        except Exception as e:
            logger.debug("Prefetch LLM call failed: {}", e)
            return 0

        if not response.tool_calls:
            return 0

        args = response.tool_calls[0].arguments
        if isinstance(args, str):
            args = json.loads(args)
        topics = args.get("topics", [])
        if not topics:
            return 0

        # Retrieve memories for predicted topics
        loaded = 0
        seen_ids: set[str] = set()
        for topic in topics[:5]:
            results = await self._retriever.retrieve(
                query=topic,
                limit=3,
                session_key=session_key,
            )
            for entry, _score in results:
                if entry.id not in seen_ids and loaded < self._max_prefetch:
                    working_memory.add(entry)
                    seen_ids.add(entry.id)
                    loaded += 1

        if loaded:
            logger.debug("Pre-fetched {} memories for topics: {}", loaded, topics)
        return loaded
