"""Agent-facing workflow tool — create and manage multi-step workflows."""

from __future__ import annotations

import json
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult
from echo_agent.tasks.workflow import WorkflowEngine


class WorkflowTool(Tool):
    name = "workflow"
    description = (
        "Create and manage multi-step workflows with DAG-based step dependencies. "
        "Actions: create, start, status, advance, pause, resume, cancel, list."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "start", "status", "advance", "pause", "resume", "cancel", "list"],
                "description": "Action to perform",
            },
            "workflow_id": {"type": "string", "description": "Workflow ID"},
            "name": {"type": "string", "description": "Workflow name (for create)"},
            "description": {"type": "string", "description": "Workflow description (for create)"},
            "steps": {
                "type": "array",
                "description": "Step definitions (for create). Each: {id, name, tool_name, tool_params, depends_on}",
                "items": {"type": "object"},
            },
            "status_filter": {"type": "string", "description": "Filter by status (for list)"},
        },
        "required": ["action"],
    }

    def __init__(self, engine: WorkflowEngine):
        self._engine = engine

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        action = params["action"]
        try:
            if action == "create":
                name = params.get("name", "Untitled workflow")
                steps = params.get("steps", [])
                if not steps:
                    return ToolResult(success=False, error="steps are required for create")
                wf = await self._engine.create(name, steps, description=params.get("description", ""))
                step_names = [s.name for s in wf.steps]
                return ToolResult(output=f"Workflow created: {wf.id} '{wf.name}'\nSteps: {', '.join(step_names)}")

            if action == "start":
                wf = await self._engine.start(params["workflow_id"])
                return ToolResult(output=f"Workflow {wf.id} started")

            if action == "status":
                wf = await self._engine.get(params.get("workflow_id", ""))
                if not wf:
                    return ToolResult(success=False, error="Workflow not found")
                return ToolResult(output=json.dumps(wf.to_dict(), ensure_ascii=False, indent=2))

            if action == "advance":
                wf = await self._engine.advance(params["workflow_id"])
                return ToolResult(output=f"Workflow {wf.id} status: {wf.status.value}")

            if action == "pause":
                wf = await self._engine.pause(params["workflow_id"])
                return ToolResult(output=f"Workflow {wf.id} paused")

            if action == "resume":
                wf = await self._engine.resume(params["workflow_id"])
                return ToolResult(output=f"Workflow {wf.id} resumed")

            if action == "cancel":
                wf = await self._engine.cancel(params["workflow_id"])
                return ToolResult(output=f"Workflow {wf.id} cancelled")

            if action == "list":
                workflows = await self._engine.list_all(status=params.get("status_filter"))
                if not workflows:
                    return ToolResult(output="No workflows found.")
                lines = [f"[{w.status.value}] {w.id}: {w.name}" for w in workflows[:50]]
                return ToolResult(output="\n".join(lines))

            return ToolResult(success=False, error=f"Unknown action: {action}")
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
