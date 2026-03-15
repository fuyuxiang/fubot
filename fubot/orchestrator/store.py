"""Persistent workflow and route-health storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fubot.orchestrator.models import AssignmentRecord, ExecutionLogRecord, TaskRecord, WorkflowRecord, utc_now
from fubot.utils.helpers import ensure_dir


class WorkflowStore:
    """Stores workflow records under the workspace for restart recovery and audits."""

    def __init__(self, workspace: Path, workflow_dir_name: str = "workflows", health_file: str = "provider-health.json"):
        self.base_dir = ensure_dir(workspace / workflow_dir_name)
        self.health_path = self.base_dir / health_file

    def _workflow_path(self, workflow_id: str) -> Path:
        return self.base_dir / f"{workflow_id}.json"

    def save_workflow(
        self,
        workflow: WorkflowRecord,
        tasks: list[TaskRecord],
        assignments: list[AssignmentRecord],
    ) -> None:
        workflow.updated_at = utc_now()
        payload = {
            "workflow": workflow.to_dict(),
            "tasks": [task.to_dict() for task in tasks],
            "assignments": [assignment.to_dict() for assignment in assignments],
        }
        self._workflow_path(workflow.id).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        path = self._workflow_path(workflow_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_workflows(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            data["_path"] = str(path)
            items.append(data)
        return items

    def recover_incomplete(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.list_workflows()
            if (item.get("workflow") or {}).get("status") not in {"completed", "cancelled", "failed"}
        ]

    def append_log(
        self,
        workflow: WorkflowRecord,
        tasks: list[TaskRecord],
        assignments: list[AssignmentRecord],
        record: ExecutionLogRecord,
    ) -> None:
        workflow.execution_logs.append(record.to_dict())
        self.save_workflow(workflow, tasks, assignments)

    def update_shared_board(
        self,
        workflow: WorkflowRecord,
        tasks: list[TaskRecord],
        assignments: list[AssignmentRecord],
        key: str,
        value: Any,
    ) -> None:
        workflow.shared_board[key] = value
        self.save_workflow(workflow, tasks, assignments)

    def load_health(self) -> dict[str, Any]:
        if not self.health_path.exists():
            return {}
        try:
            return json.loads(self.health_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_health(self, payload: dict[str, Any]) -> None:
        self.health_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
