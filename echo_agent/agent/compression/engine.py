"""ContextEngine — abstract base class for context compression strategies."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from echo_agent.agent.compression.types import CompressionResult, CompressionStats


class ContextEngine(ABC):
    """Base class defining the compression engine lifecycle and interface.

    Subclasses implement the actual compression strategy while this class
    provides token estimation, lifecycle hooks, and stats tracking.
    """

    def __init__(self, context_window_tokens: int, trigger_ratio: float = 0.7):
        self.context_window_tokens = context_window_tokens
        self.trigger_ratio = trigger_ratio
        self._stats = CompressionStats()
        self._session_key: str | None = None
        self._token_counter = None

    def set_token_counter(self, counter) -> None:
        self._token_counter = counter

    def should_compress(self, messages: list[dict[str, Any]]) -> bool:
        threshold = int(self.context_window_tokens * self.trigger_ratio)
        return self.estimate_tokens(messages) > threshold

    @abstractmethod
    async def compress(
        self,
        messages: list[dict[str, Any]],
        focus_topic: str = "",
    ) -> CompressionResult:
        """Run the full compression pipeline and return the result."""

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += self._estimate_message_tokens(msg)
        return total

    def _estimate_message_tokens(self, msg: dict[str, Any]) -> int:
        if self._token_counter:
            content = msg.get("content", "")
            if isinstance(content, str):
                return self._token_counter.count(content) + 4
            elif isinstance(content, list):
                total = 4
                for block in content:
                    if isinstance(block, dict):
                        total += self._token_counter.count(block.get("text", ""))
                return total

        tokens = 4  # role + framing overhead
        content = msg.get("content", "")
        if isinstance(content, str):
            tokens += len(content) // 4
        elif isinstance(content, list):
            for block in content:
                tokens += len(str(block.get("text", ""))) // 4

        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            tokens += len(func.get("name", "")) // 4
            args = func.get("arguments", "")
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            tokens += len(args) // 4

        return tokens

    def get_stats(self) -> CompressionStats:
        return self._stats

    def on_session_start(self, session_key: str) -> None:
        self._session_key = session_key
        self._stats = CompressionStats()

    def on_session_end(self, session_key: str) -> None:
        self._session_key = None

    def on_session_reset(self, session_key: str) -> None:
        self._stats = CompressionStats()
