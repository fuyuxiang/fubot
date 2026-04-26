"""Anthropic provider — Messages API with prompt caching and thinking support."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from echo_agent.models.provider import LLMProvider, LLMResponse
from echo_agent.models.providers.format_utils import (
    anthropic_response_to_llm_fields,
    openai_to_anthropic_messages,
    openai_to_anthropic_tools,
)

_OUTPUT_LIMITS: dict[str, int] = {
    "opus-4": 128_000,
    "sonnet-4": 64_000,
    "haiku-4": 64_000,
    "claude-3.5": 8192,
    "claude-3": 4096,
}

_THINKING_BUDGETS: dict[str, int] = {
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}


def _max_output_for_model(model: str) -> int:
    for pattern, limit in _OUTPUT_LIMITS.items():
        if pattern in model:
            return limit
    return 64_000


def _supports_adaptive_thinking(model: str) -> bool:
    return any(tag in model for tag in ("opus-4", "sonnet-4.6", "sonnet-4-6"))


class AnthropicProvider(LLMProvider):

    def __init__(self, api_key: str = "", api_base: str = "", default_model: str = "", **kwargs: Any):
        super().__init__(api_key=api_key, api_base=api_base)
        self._default_model = default_model
        self._enable_cache = kwargs.get("enable_cache", True)
        self._thinking_effort: str = kwargs.get("thinking_effort", "")
        self._client = self._build_client()

    def _build_client(self) -> Any:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic SDK required: pip install echo-agent[anthropic]")

        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["base_url"] = self.api_base
        return AsyncAnthropic(**kwargs)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        target_model = model or self._default_model
        params = self._build_params(target_model, messages, tools, tool_choice, **kwargs)
        try:
            resp = await self._client.messages.create(**params)
        except Exception as e:
            logger.error("Anthropic API error: {}", e)
            return LLMResponse(content=f"Error: {e}", finish_reason="error")
        return self._parse_response(resp)

    def get_default_model(self) -> str:
        return self._default_model

    def _build_params(
        self,
        target_model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        system_blocks, converted = openai_to_anthropic_messages(
            messages, inject_cache_markers=self._enable_cache,
        )

        max_out = kwargs.get("max_tokens", _max_output_for_model(target_model))
        params: dict[str, Any] = {
            "model": target_model,
            "messages": converted,
            "max_tokens": max_out,
        }
        if system_blocks:
            params["system"] = system_blocks

        if tools:
            params["tools"] = openai_to_anthropic_tools(tools, inject_cache_markers=self._enable_cache)
        if tool_choice and tools:
            params["tool_choice"] = self._convert_tool_choice(tool_choice)

        self._apply_thinking(params, target_model, kwargs)

        if "thinking" not in params:
            temp = kwargs.get("temperature", self.generation.temperature)
            if temp is not None:
                params["temperature"] = temp

        return params

    def _apply_thinking(self, params: dict[str, Any], model: str, kwargs: dict[str, Any]) -> None:
        effort = kwargs.get("thinking_effort") or self._thinking_effort
        if not effort:
            return

        if _supports_adaptive_thinking(model):
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
            effort_map = {"low": "low", "medium": "medium", "high": "high"}
            params.setdefault("output_config", {})["effort"] = effort_map.get(effort, "medium")
        else:
            budget = _THINKING_BUDGETS.get(effort, 8192)
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
            params["temperature"] = 1
            params["max_tokens"] = max(params.get("max_tokens", 4096), budget + 4096)

    def _convert_tool_choice(self, choice: str | dict) -> dict[str, Any]:
        if isinstance(choice, dict):
            return choice
        mapping = {"auto": {"type": "auto"}, "none": {"type": "none"}, "required": {"type": "any"}}
        return mapping.get(choice, {"type": "auto"})

    def _parse_response(self, resp: Any) -> LLMResponse:
        blocks = []
        for block in resp.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})

        usage_dict: dict[str, Any] = {}
        if resp.usage:
            usage_dict["input_tokens"] = resp.usage.input_tokens
            usage_dict["output_tokens"] = resp.usage.output_tokens
            if hasattr(resp.usage, "cache_read_input_tokens"):
                usage_dict["cache_read_input_tokens"] = resp.usage.cache_read_input_tokens or 0
            if hasattr(resp.usage, "cache_creation_input_tokens"):
                usage_dict["cache_creation_input_tokens"] = resp.usage.cache_creation_input_tokens or 0

        fields = anthropic_response_to_llm_fields(
            content_blocks=blocks,
            stop_reason=resp.stop_reason or "",
            usage=usage_dict,
            model=resp.model or "",
        )
        return LLMResponse(**fields)
