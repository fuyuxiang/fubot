"""Clarify tool — ask user clarifying questions."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult
from echo_agent.bus.events import OutboundEvent
from echo_agent.bus.queue import MessageBus


class ClarifyTool(Tool):
    name = "clarify"
    description = "Ask the user a clarifying question, optionally with multiple-choice options."
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to ask the user."},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of choices for the user.",
            },
        },
        "required": ["question"],
    }

    def __init__(self, bus: MessageBus):
        self._bus = bus

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        question = params["question"]
        options = params.get("options", [])

        text = question
        if options:
            choices = "\n".join(f"  {i+1}. {opt}" for i, opt in enumerate(options))
            text = f"{question}\n{choices}"

        return ToolResult(
            output=text,
            metadata={"type": "clarify", "question": question, "options": options},
        )
