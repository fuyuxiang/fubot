"""Model abstraction layer — multi-provider LLM interface.

Supports: multi-model switching, task-based routing, fallback/degradation,
cost control, context length handling, unified generation parameters.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

StreamDeltaCallback = Callable[[str], Awaitable[None] | None]


async def _invoke_stream_callback(callback: StreamDeltaCallback | None, delta: str) -> None:
    if callback is None or not delta:
        return
    result = callback(delta)
    if asyncio.iscoroutine(result):
        await result


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    reasoning_content: str | None = None
    thinking_blocks: list[dict[str, Any]] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    reasoning_effort: str | None = None


class LLMProvider(ABC):
    """Abstract base for LLM providers (OpenAI, Anthropic, etc.)."""

    _RETRY_DELAYS = (1, 2, 4)
    _TRANSIENT_MARKERS = ("429", "rate limit", "500", "502", "503", "504", "overloaded", "timeout")

    def __init__(self, api_key: str = "", api_base: str = ""):
        self.api_key = api_key
        self.api_base = api_base
        self.generation = GenerationParams()

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request."""

    @abstractmethod
    def get_default_model(self) -> str:
        """Return the default model identifier."""

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tool_choice: str | dict | None = None,
        on_delta: StreamDeltaCallback | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Stream text deltas when supported, otherwise fall back to a full response."""
        response = await self.chat(messages, tools, model, tool_choice, **kwargs)
        if response.finish_reason != "error" and response.content:
            await _invoke_stream_callback(on_delta, response.content)
        return response

    def _is_transient(self, error_text: str) -> bool:
        lower = error_text.lower()
        return any(m in lower for m in self._TRANSIENT_MARKERS)

    async def chat_with_retry(self, **kwargs: Any) -> LLMResponse:
        for attempt, delay in enumerate(self._RETRY_DELAYS):
            try:
                response = await self.chat(**kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                response = LLMResponse(content=f"Error: {e}", finish_reason="error")

            if response.finish_reason != "error":
                return response
            if not self._is_transient(response.content or ""):
                return response

            logger.warning("LLM transient error (attempt {}), retrying in {}s", attempt + 1, delay)
            await asyncio.sleep(delay)

        try:
            return await self.chat(**kwargs)
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tool_choice: str | dict | None = None,
        on_delta: StreamDeltaCallback | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        emitted = False

        async def wrapped(delta: str) -> None:
            nonlocal emitted
            emitted = True
            await _invoke_stream_callback(on_delta, delta)

        for attempt, delay in enumerate(self._RETRY_DELAYS):
            emitted = False
            try:
                response = await self.chat_stream(
                    messages=messages,
                    tools=tools,
                    model=model,
                    tool_choice=tool_choice,
                    on_delta=wrapped,
                    **kwargs,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                response = LLMResponse(content=f"Error: {e}", finish_reason="error")

            if response.finish_reason != "error":
                return response
            if emitted or not self._is_transient(response.content or ""):
                return response

            logger.warning("LLM transient stream error (attempt {}), retrying in {}s", attempt + 1, delay)
            await asyncio.sleep(delay)

        try:
            return await self.chat_stream(
                messages=messages,
                tools=tools,
                model=model,
                tool_choice=tool_choice,
                on_delta=wrapped,
                **kwargs,
            )
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")
