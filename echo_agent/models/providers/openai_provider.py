"""OpenAI provider — chat completions via the openai SDK."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from echo_agent.models.provider import (
    GenerationParams,
    LLMProvider,
    LLMResponse,
    StreamDeltaCallback,
    ToolCallRequest,
    _invoke_stream_callback,
)


class OpenAIProvider(LLMProvider):

    def __init__(self, api_key: str = "", api_base: str = "", default_model: str = "gpt-4o", **kwargs: Any):
        super().__init__(api_key=api_key, api_base=api_base)
        self._default_model = default_model
        self._extra_headers: dict[str, str] = kwargs.get("extra_headers", {})
        self._client = self._build_client()

    def _build_client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai SDK required: pip install echo-agent[openai]")

        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["base_url"] = self.api_base
        if self._extra_headers:
            kwargs["default_headers"] = self._extra_headers
        return AsyncOpenAI(**kwargs)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        params = self._build_params(messages, tools, model, tool_choice, **kwargs)
        try:
            resp = await self._client.chat.completions.create(**params)
        except Exception as e:
            logger.error("OpenAI API error: {}", e)
            return LLMResponse(content=f"Error: {e}", finish_reason="error")
        return self._parse_response(resp)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tool_choice: str | dict | None = None,
        on_delta: StreamDeltaCallback | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        params = self._build_params(messages, tools, model, tool_choice, **kwargs)
        params["stream"] = True

        try:
            stream = await self._client.chat.completions.create(**params)
        except Exception as e:
            logger.warning("OpenAI stream init failed, falling back to non-streaming: {}", e)
            return await self.chat(messages, tools, model, tool_choice, **kwargs)

        text_parts: list[str] = []
        tool_parts: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}
        response_model = params.get("model", model or self._default_model)

        try:
            async for chunk in stream:
                if getattr(chunk, "model", None):
                    response_model = chunk.model or response_model

                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage["prompt_tokens"] = getattr(chunk_usage, "prompt_tokens", 0) or usage.get("prompt_tokens", 0)
                    usage["completion_tokens"] = getattr(chunk_usage, "completion_tokens", 0) or usage.get("completion_tokens", 0)

                choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
                if not choice:
                    continue
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if not delta:
                    continue

                content = self._delta_text(delta)
                if content:
                    text_parts.append(content)
                    await _invoke_stream_callback(on_delta, content)

                for tc in getattr(delta, "tool_calls", None) or []:
                    self._merge_tool_delta(tool_parts, tc)
        except Exception as e:
            logger.error("OpenAI streaming error: {}", e)
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

        tool_calls = self._finalize_tool_calls(tool_parts)
        if finish_reason == "stop" and tool_calls:
            finish_reason = "tool_calls"

        return LLMResponse(
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=response_model or "",
        )

    def get_default_model(self) -> str:
        return self._default_model

    def _build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        tool_choice: str | dict | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        clean_msgs = self._clean_messages(messages)
        params: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": clean_msgs,
            "temperature": kwargs.get("temperature", self.generation.temperature),
            "max_tokens": kwargs.get("max_tokens", self.generation.max_tokens),
        }
        if tools:
            params["tools"] = tools
        if tool_choice and tools:
            params["tool_choice"] = tool_choice
        if self.generation.top_p < 1.0:
            params["top_p"] = self.generation.top_p
        for key in ("extra_body", "extra_headers"):
            if key in kwargs:
                params[key] = kwargs[key]
        return params

    def _clean_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned = []
        for msg in messages:
            m = {"role": msg["role"]}
            if "content" in msg and msg["content"] is not None:
                m["content"] = msg["content"]
            if "tool_calls" in msg:
                m["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                m["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                m["name"] = msg["name"]
            cleaned.append(m)
        return cleaned

    def _parse_response(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0] if resp.choices else None
        if not choice:
            return LLMResponse(content="No response from model", finish_reason="error")

        msg = choice.message
        tool_calls: list[ToolCallRequest] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args))

        usage: dict[str, int] = {}
        if resp.usage:
            usage["prompt_tokens"] = resp.usage.prompt_tokens or 0
            usage["completion_tokens"] = resp.usage.completion_tokens or 0

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            model=resp.model or "",
        )

    @staticmethod
    def _delta_text(delta: Any) -> str:
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _merge_tool_delta(tool_parts: dict[int, dict[str, Any]], tc: Any) -> None:
        index = getattr(tc, "index", None)
        if index is None:
            index = len(tool_parts)
        entry = tool_parts.setdefault(index, {"id": "", "name": "", "arguments": []})
        if getattr(tc, "id", None):
            entry["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                entry["name"] = fn.name
            if getattr(fn, "arguments", None):
                entry["arguments"].append(fn.arguments)

    @staticmethod
    def _finalize_tool_calls(tool_parts: dict[int, dict[str, Any]]) -> list[ToolCallRequest]:
        tool_calls: list[ToolCallRequest] = []
        for index in sorted(tool_parts):
            entry = tool_parts[index]
            raw_args = "".join(entry["arguments"])
            try:
                args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                args = {"raw": raw_args}
            tool_calls.append(ToolCallRequest(
                id=entry["id"] or f"call_{index}",
                name=entry["name"],
                arguments=args,
            ))
        return tool_calls
