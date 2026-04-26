"""Google Gemini provider — generative AI SDK."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from echo_agent.models.provider import LLMProvider, LLMResponse, ToolCallRequest


class GeminiProvider(LLMProvider):

    def __init__(self, api_key: str = "", api_base: str = "", default_model: str = "", **kwargs: Any):
        super().__init__(api_key=api_key, api_base=api_base)
        self._default_model = default_model
        self._client = self._build_client()

    def _build_client(self) -> Any:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai required: pip install echo-agent[gemini]")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        return genai

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        target = model or self._default_model
        try:
            return await self._do_chat(target, messages, tools, **kwargs)
        except Exception as e:
            logger.error("Gemini API error: {}", e)
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def get_default_model(self) -> str:
        return self._default_model

    async def _do_chat(
        self, model_name: str, messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None, **kwargs: Any,
    ) -> LLMResponse:
        import asyncio
        genai = self._client

        system_text, contents = self._convert_messages(messages)
        gen_config = {
            "temperature": kwargs.get("temperature", self.generation.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.generation.max_tokens),
        }

        model_kwargs: dict[str, Any] = {"model_name": model_name, "generation_config": gen_config}
        if system_text:
            model_kwargs["system_instruction"] = system_text

        gemini_model = genai.GenerativeModel(**model_kwargs)

        tool_defs = self._convert_tools(tools) if tools else None
        send_kwargs: dict[str, Any] = {"content": contents}
        if tool_defs:
            send_kwargs["tools"] = tool_defs

        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: gemini_model.generate_content(**send_kwargs),
        )
        return self._parse_response(resp, model_name)

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(content or "")
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts: list[dict[str, Any]] = []

            if content:
                parts.append({"text": content})

            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {}
                parts.append({"function_call": {"name": fn.get("name", ""), "args": args}})

            if role == "tool":
                parts = [{"function_response": {
                    "name": msg.get("name", ""),
                    "response": {"result": content},
                }}]
                gemini_role = "user"

            if parts:
                if contents and contents[-1]["role"] == gemini_role:
                    contents[-1]["parts"].extend(parts)
                else:
                    contents.append({"role": gemini_role, "parts": parts})

        return "\n".join(system_parts), contents

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations = []
        for tool in tools:
            fn = tool.get("function", tool)
            decl: dict[str, Any] = {"name": fn.get("name", ""), "description": fn.get("description", "")}
            params = fn.get("parameters")
            if params:
                decl["parameters"] = params
            declarations.append(decl)
        return [{"function_declarations": declarations}]

    def _parse_response(self, resp: Any, model: str) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for candidate in resp.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(ToolCallRequest(
                        id=f"call_{fc.name}",
                        name=fc.name,
                        arguments=args,
                    ))

        finish = "tool_calls" if tool_calls else "stop"
        usage: dict[str, int] = {}
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            um = resp.usage_metadata
            usage["prompt_tokens"] = getattr(um, "prompt_token_count", 0) or 0
            usage["completion_tokens"] = getattr(um, "candidates_token_count", 0) or 0

        return LLMResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage,
            model=model,
        )
