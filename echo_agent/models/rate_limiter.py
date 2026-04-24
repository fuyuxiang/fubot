"""Token-bucket rate limiter and rate-limited provider wrapper."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from echo_agent.models.provider import LLMProvider, LLMResponse, StreamDeltaCallback


class TokenBucketLimiter:

    def __init__(self, tokens_per_minute: int, burst: int = 0):
        self._rate = tokens_per_minute / 60.0
        self._capacity = burst or tokens_per_minute
        self._tokens = float(self._capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, count: int = 1) -> None:
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= count:
                    self._tokens -= count
                    return
                wait = (count - self._tokens) / self._rate
            await asyncio.sleep(min(wait, 5.0))

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


class RateLimitedProvider(LLMProvider):

    def __init__(self, inner: LLMProvider, limiter: TokenBucketLimiter):
        super().__init__(api_key=inner.api_key, api_base=inner.api_base)
        self._inner = inner
        self._limiter = limiter
        self.generation = inner.generation

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        await self._limiter.acquire()
        return await self._inner.chat(messages, tools, model, tool_choice, **kwargs)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tool_choice: str | dict | None = None,
        on_delta: StreamDeltaCallback | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        await self._limiter.acquire()
        return await self._inner.chat_stream(messages, tools, model, tool_choice, on_delta=on_delta, **kwargs)

    def get_default_model(self) -> str:
        return self._inner.get_default_model()
