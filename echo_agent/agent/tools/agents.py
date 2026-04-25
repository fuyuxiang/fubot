"""Tools for inspecting multi-agent routing."""

from __future__ import annotations

import json
from typing import Any

from echo_agent.agent.multi_agent.runtime import MultiAgentRuntime
from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolPermission, ToolResult


class AgentsListTool(Tool):
    name = "agents_list"
    description = "List configured specialist agents, their capabilities, and allowed tools."
    parameters = {"type": "object", "properties": {}}
    required_permissions = [ToolPermission.READ]

    def __init__(self, runtime: MultiAgentRuntime):
        self._runtime = runtime

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        agents = []
        for profile in self._runtime.registry.list():
            agents.append({
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "capabilities": list(profile.capabilities),
                "task_types": list(profile.task_types),
                "tools_allow": list(profile.tools_allow),
                "model": profile.model,
                "provider": profile.provider,
                "priority": profile.priority,
            })
        return ToolResult(output=json.dumps({"agents": agents}, ensure_ascii=False, indent=2))

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"


class AgentsRouteTool(Tool):
    name = "agents_route"
    description = "Preview which specialist agent(s) would handle a request and why."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The user request to route."},
            "task_type": {"type": "string", "description": "Optional known task type.", "default": "chat"},
        },
        "required": ["query"],
    }
    required_permissions = [ToolPermission.READ]

    def __init__(self, runtime: MultiAgentRuntime):
        self._runtime = runtime

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        plan = self._runtime.plan(params["query"], task_type=params.get("task_type", "chat"))
        payload = {
            "strategy": plan.strategy,
            "task_type": plan.task_type,
            "primary_agent_id": plan.primary_agent_id,
            "selected_agent_ids": plan.selected_agent_ids,
            "confidence": plan.confidence,
            "rationale": plan.rationale,
            "candidates": [
                {
                    "agent_id": candidate.agent_id,
                    "score": candidate.score,
                    "confidence": candidate.confidence,
                    "reasons": candidate.reasons,
                }
                for candidate in plan.candidates
            ],
        }
        return ToolResult(output=json.dumps(payload, ensure_ascii=False, indent=2))

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"
