"""Shared format converters between OpenAI and Anthropic message formats."""

from __future__ import annotations

import json
import re
from typing import Any


def openai_to_anthropic_messages(
    messages: list[dict[str, Any]],
    inject_cache_markers: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system_blocks: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            block: dict[str, Any] = {"type": "text", "text": content or "(empty)"}
            system_blocks.append(block)
            continue

        if role == "assistant":
            converted.append(_convert_assistant_msg(msg))
        elif role == "tool":
            _append_tool_result(converted, msg)
        elif role == "user":
            converted.append({"role": "user", "content": _ensure_content_blocks(content)})
        else:
            converted.append({"role": "user", "content": _ensure_content_blocks(content)})

    converted = _enforce_alternation(converted)

    if inject_cache_markers:
        _inject_cache_control(system_blocks, converted)

    return system_blocks, converted


def _convert_assistant_msg(msg: dict[str, Any]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    content = msg.get("content")
    if content:
        blocks.append({"type": "text", "text": content})

    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": args}
        tool_id = _sanitize_tool_id(tc.get("id", ""))
        blocks.append({"type": "tool_use", "id": tool_id, "name": fn.get("name", ""), "input": args})

    if not blocks:
        blocks.append({"type": "text", "text": "(empty)"})
    return {"role": "assistant", "content": blocks}


def _append_tool_result(converted: list[dict[str, Any]], msg: dict[str, Any]) -> None:
    result_block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": _sanitize_tool_id(msg.get("tool_call_id", "")),
        "content": msg.get("content", ""),
    }
    if converted and converted[-1]["role"] == "user":
        existing = converted[-1]["content"]
        if isinstance(existing, list):
            existing.append(result_block)
        else:
            converted[-1]["content"] = [{"type": "text", "text": existing}, result_block]
    else:
        converted.append({"role": "user", "content": [result_block]})


def _ensure_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return content
    text = str(content) if content else "(empty)"
    return [{"type": "text", "text": text}]


def _sanitize_tool_id(tool_id: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)
    return sanitized or "tool_0"


def _enforce_alternation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return messages
    merged: list[dict[str, Any]] = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            _merge_into(merged[-1], msg)
        else:
            merged.append(msg)
    return merged


def _merge_into(target: dict[str, Any], source: dict[str, Any]) -> None:
    tc = target["content"]
    sc = source["content"]
    if isinstance(tc, list) and isinstance(sc, list):
        target["content"] = tc + sc
    elif isinstance(tc, list):
        target["content"] = tc + _ensure_content_blocks(sc)
    elif isinstance(sc, list):
        target["content"] = _ensure_content_blocks(tc) + sc
    else:
        target["content"] = _ensure_content_blocks(tc) + _ensure_content_blocks(sc)


def _inject_cache_control(
    system_blocks: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> None:
    if system_blocks:
        system_blocks[-1]["cache_control"] = {"type": "ephemeral"}

    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    for idx in user_indices[-2:]:
        content = messages[idx]["content"]
        if isinstance(content, list) and content:
            content[-1]["cache_control"] = {"type": "ephemeral"}


def openai_to_anthropic_tools(
    tools: list[dict[str, Any]],
    inject_cache_markers: bool = False,
) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        fn = tool.get("function", tool)
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    if inject_cache_markers and result:
        result[-1]["cache_control"] = {"type": "ephemeral"}
    return result


def anthropic_response_to_llm_fields(
    content_blocks: list[dict[str, Any]],
    stop_reason: str = "",
    usage: dict[str, Any] | None = None,
    model: str = "",
) -> dict[str, Any]:
    from echo_agent.models.provider import LLMResponse, ToolCallRequest

    text_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []

    for block in content_blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(ToolCallRequest(
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=block.get("input", {}),
            ))

    finish = _map_stop_reason(stop_reason)
    resp_usage: dict[str, int] = {}
    if usage:
        resp_usage["prompt_tokens"] = usage.get("input_tokens", 0)
        resp_usage["completion_tokens"] = usage.get("output_tokens", 0)
        cached = usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
        if cached:
            resp_usage["cached_tokens"] = cached

    return {
        "content": "\n".join(text_parts) if text_parts else None,
        "tool_calls": tool_calls,
        "finish_reason": finish,
        "usage": resp_usage,
        "model": model,
    }


def _map_stop_reason(reason: str) -> str:
    mapping = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "content_filter": "content_filter",
        "refusal": "content_filter",
    }
    return mapping.get(reason, reason or "stop")
