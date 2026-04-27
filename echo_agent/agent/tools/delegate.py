"""Delegate tool — spawn a child agent with restricted tools and isolated context.

Supports synchronous (blocking) and background (async) execution modes.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from loguru import logger

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult
from echo_agent.models.provider import LLMProvider


class DelegateTool(Tool):
    name = "delegate_task"
    description = "Delegate a subtask to a child agent with its own context. Useful for parallel or isolated work."
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Description of the task to delegate."},
            "context": {"type": "string", "description": "Additional context or instructions for the child agent."},
            "max_iterations": {"type": "integer", "description": "Max tool-call iterations for the child.", "default": 10},
        },
        "required": ["task"],
    }
    timeout_seconds = 300

    def __init__(self, provider: LLMProvider, tool_registry_factory: Any = None):
        self._provider = provider
        self._tool_registry_factory = tool_registry_factory

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        task = params["task"]
        extra_context = params.get("context", "")
        max_iter = min(params.get("max_iterations", 10), 20)

        system = "You are a focused sub-agent. Complete the given task concisely."
        if extra_context:
            system += f"\n\nContext:\n{extra_context}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        tools = []
        if self._tool_registry_factory:
            registry = self._tool_registry_factory()
            tools = registry.get_definitions()

        iteration = 0
        while iteration < max_iter:
            resp = await self._provider.chat_with_retry(
                messages=messages, tools=tools if tools else None,
            )

            if not resp.has_tool_calls:
                return ToolResult(
                    output=resp.content or "(no response)",
                    metadata={"iterations": iteration + 1},
                )

            messages.append({
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [tc.to_openai_format() for tc in resp.tool_calls],
            })

            if self._tool_registry_factory:
                registry = self._tool_registry_factory()
                for tc in resp.tool_calls:
                    result = await registry.execute(tc.name, tc.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result.text[:8000],
                    })
            else:
                for tc in resp.tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": "[Tool execution not available in delegate mode]",
                    })

            iteration += 1

        return ToolResult(
            success=False,
            output=resp.content or "",
            error=f"Delegate reached max iterations ({max_iter})",
        )


class SpawnTool(Tool):
    name = "spawn_task"
    description = (
        "Spawn a background task that runs asynchronously. "
        "Returns immediately with a task ID. The result will be announced when complete."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Description of the task to run in background."},
            "context": {"type": "string", "description": "Additional context for the background agent."},
        },
        "required": ["task"],
    }

    def __init__(self, provider: LLMProvider, bus: Any = None):
        self._provider = provider
        self._bus = bus
        self._tasks: dict[str, asyncio.Task] = {}

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        task_desc = params["task"]
        extra = params.get("context", "")
        task_id = f"bg_{uuid.uuid4().hex[:8]}"

        async_task = asyncio.create_task(self._run_background(task_id, task_desc, extra, ctx))
        self._tasks[task_id] = async_task
        return ToolResult(
            output=f"Background task '{task_id}' started. Result will be announced when complete.",
            metadata={"task_id": task_id},
        )

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "side_effect"

    async def _run_background(self, task_id: str, task: str, context: str, ctx: ToolExecutionContext | None) -> None:
        system = "You are a background agent. Complete the task and provide a concise result."
        if context:
            system += f"\n\nContext:\n{context}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        try:
            resp = await self._provider.chat_with_retry(messages=messages)
            result = resp.content or "(no response)"
            logger.info("Background task {} completed: {}", task_id, result[:100])

            if self._bus:
                from echo_agent.bus.events import InboundEvent
                announce = InboundEvent.text_message(
                    channel="system", sender_id="system", chat_id=ctx.session_key.split(":")[1] if ctx else "system",
                    text=f"[Background task {task_id} completed]\n\n{result}",
                    session_key_override=ctx.session_key if ctx else "system:system",
                )
                await self._bus.publish_inbound(announce)
        except Exception as e:
            logger.error("Background task {} failed: {}", task_id, e)
        finally:
            self._tasks.pop(task_id, None)
