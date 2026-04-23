"""Cronjob tool — create, list, and delete scheduled tasks."""

from __future__ import annotations

from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolPermission, ToolResult
from echo_agent.scheduler.service import ScheduledJob, Scheduler, TriggerKind


class CronjobTool(Tool):
    name = "cronjob"
    description = "Manage scheduled tasks: create, list, update, or delete cron-based jobs."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "delete", "trigger"], "description": "Action to perform."},
            "name": {"type": "string", "description": "Job name (for create/delete/trigger)."},
            "schedule": {"type": "string", "description": "Cron expression (for create), e.g., '*/5 * * * *'."},
            "command": {"type": "string", "description": "Command or message to execute on schedule (for create)."},
            "job_id": {"type": "string", "description": "Job ID (for delete/trigger)."},
        },
        "required": ["action"],
    }
    required_permissions = [ToolPermission.ADMIN]

    def __init__(self, scheduler: Scheduler | None):
        self._scheduler = scheduler

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        if not self._scheduler:
            return ToolResult(success=False, error="Scheduler not enabled")

        action = params["action"]

        if action == "create":
            name = params.get("name", "unnamed")
            schedule = params.get("schedule", "")
            command = params.get("command", "")
            if not schedule or not command:
                return ToolResult(success=False, error="Both 'schedule' and 'command' are required for create")
            job = ScheduledJob(
                name=name,
                trigger=TriggerKind.CRON,
                cron_expr=schedule,
                payload={"command": command},
            )
            created = self._scheduler.add_job(job)
            return ToolResult(output=f"Created job '{name}' (id={created.id}): {schedule}", metadata={"job_id": created.id})

        if action == "list":
            jobs = self._scheduler.list_jobs()
            if not jobs:
                return ToolResult(output="No scheduled jobs.")
            lines = []
            for j in jobs:
                payload = j.payload.get("command", "") if isinstance(j.payload, dict) else str(j.payload)
                schedule = j.cron_expr or str(j.interval_ms) or str(j.at_ms)
                lines.append(f"{j.id}: [{schedule}] {j.name} — {payload[:60]}")
            return ToolResult(output="\n".join(lines))

        if action == "delete":
            job_id = params.get("job_id", "")
            if not job_id:
                return ToolResult(success=False, error="'job_id' required for delete")
            removed = self._scheduler.remove_job(job_id)
            if removed:
                return ToolResult(output=f"Deleted job {job_id}")
            return ToolResult(success=False, error=f"Job '{job_id}' not found")

        if action == "trigger":
            job_id = params.get("job_id", "")
            if not job_id:
                return ToolResult(success=False, error="'job_id' required for trigger")
            triggered = await self._scheduler.trigger_job(job_id)
            if triggered:
                return ToolResult(output=f"Triggered job {job_id}")
            return ToolResult(success=False, error=f"Job '{job_id}' not found or failed to trigger")

        return ToolResult(success=False, error=f"Unknown action: {action}")
