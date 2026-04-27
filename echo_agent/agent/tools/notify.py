"""Notify tool — send notifications via the message bus."""

from __future__ import annotations

from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult
from echo_agent.bus.events import OutboundEvent
from echo_agent.bus.queue import MessageBus


class NotifyTool(Tool):
    name = "notify"
    description = "Send a notification message to a specific channel or the current chat."
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Notification message text."},
            "channel": {"type": "string", "description": "Target channel (e.g., 'cli', 'telegram', 'webhook'). Defaults to 'cli'."},
            "chat_id": {"type": "string", "description": "Target chat ID. Defaults to 'default'."},
        },
        "required": ["message"],
    }

    def __init__(self, bus: MessageBus):
        self._bus = bus

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        message = params["message"]
        channel = params.get("channel", "cli")
        chat_id = params.get("chat_id", "default")

        event = OutboundEvent.text_reply(channel=channel, chat_id=chat_id, text=message)
        await self._bus.publish_outbound(event)
        return ToolResult(output=f"Notification sent to {channel}:{chat_id}")
