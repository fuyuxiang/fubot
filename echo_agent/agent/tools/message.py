"""Message tool — send messages to specific channels from within the agent loop."""

from __future__ import annotations

from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult
from echo_agent.bus.events import OutboundEvent


class MessageTool(Tool):
    name = "message"
    description = "Send a message to a specific channel and chat."
    parameters = {
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Target channel name."},
            "chat_id": {"type": "string", "description": "Target chat ID."},
            "text": {"type": "string", "description": "Message text."},
        },
        "required": ["channel", "chat_id", "text"],
    }

    def __init__(self, publish_fn=None):
        self._publish = publish_fn

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        if not self._publish:
            return ToolResult(success=False, error="Message bus not connected")
        event = OutboundEvent.text_reply(
            channel=params["channel"],
            chat_id=params["chat_id"],
            text=params["text"],
        )
        try:
            await self._publish(event)
            return ToolResult(output=f"Message sent to {params['channel']}:{params['chat_id']}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
