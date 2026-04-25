"""A2A HTTP server — agent card, task handling, SSE streaming."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from aiohttp import web
from loguru import logger

from echo_agent.a2a.models import AgentCard, A2ATask, A2AMessage, TaskState
from echo_agent.a2a.protocol import A2AProtocol

if TYPE_CHECKING:
    from echo_agent.agent.loop import AgentLoop


class A2AServer:
    def __init__(self, agent_loop: AgentLoop, agent_card: AgentCard):
        self._loop = agent_loop
        self._card = agent_card
        self._protocol = A2AProtocol(self._process_task)

    def register_routes(self, app: web.Application) -> None:
        app.router.add_get("/.well-known/agent.json", self._handle_agent_card)
        app.router.add_post("/a2a", self._handle_rpc)
        logger.info("A2A routes registered: /.well-known/agent.json, /a2a")

    async def _handle_agent_card(self, request: web.Request) -> web.Response:
        return web.json_response(self._card.to_dict())

    async def _handle_rpc(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
                status=400,
            )
        result = await self._protocol.handle(body)
        return web.json_response(result)

    async def _process_task(self, task: A2ATask) -> A2ATask:
        """Process an A2A task by routing through the agent loop."""
        user_text = ""
        for msg in task.messages:
            if msg.role == "user":
                user_text = msg.text_content
                break

        if not user_text:
            task.state = TaskState.FAILED
            task.messages.append(A2AMessage.text("agent", "No user message found"))
            return task

        try:
            from echo_agent.bus.events import InboundEvent
            event = InboundEvent.text_message(
                channel="a2a", sender_id="a2a-client",
                chat_id=f"a2a:{task.id}", text=user_text,
            )
            import uuid
            result = await self._loop._process_event(event, trace_id=uuid.uuid4().hex[:12])
            task.state = TaskState.COMPLETED
            task.messages.append(A2AMessage.text("agent", result.response_text or ""))
        except Exception as e:
            logger.error("A2A task processing failed: {}", e)
            task.state = TaskState.FAILED
            task.messages.append(A2AMessage.text("agent", f"Error: {e}"))

        return task
