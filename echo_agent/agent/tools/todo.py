"""Todo tool — task planning and tracking per session."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolPermission, ToolResult


class TodoTool(Tool):
    name = "todo"
    description = "Manage a task list: create, update, list, or complete tasks for planning multi-step work."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "update", "list", "complete", "delete"], "description": "Action to perform."},
            "title": {"type": "string", "description": "Task title (for create/update)."},
            "items": {
                "type": "array",
                "description": "Task titles for batch create. Each item may be a string or an object with title/notes.",
            },
            "task_id": {"type": "string", "description": "Task ID (for update/complete/delete)."},
            "status": {"type": "string", "enum": ["pending", "in_progress", "done"], "description": "New status (for update)."},
            "notes": {"type": "string", "description": "Additional notes."},
        },
        "required": ["action"],
    }
    required_permissions = [ToolPermission.WRITE]

    def __init__(self, store_dir: Path):
        self._store_dir = store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        action = params["action"]
        tasks = self._load()

        if action == "create":
            created = []
            for item in self._create_items(params):
                task_id = self._new_task_id(tasks)
                tasks[task_id] = {
                    "title": item["title"],
                    "status": "pending",
                    "notes": item.get("notes", ""),
                    "created": time.time(),
                }
                created.append((task_id, item["title"]))
            self._save(tasks)
            return ToolResult(output="\n".join(f"Created task {tid}: {title}" for tid, title in created))

        if action == "list":
            if not tasks:
                return ToolResult(output="No tasks.")
            lines = []
            for tid, t in tasks.items():
                lines.append(f"[{t['status']}] {tid}: {t['title']}" + (f" — {t['notes']}" if t.get("notes") else ""))
            return ToolResult(output="\n".join(lines))

        task_id = params.get("task_id", "")
        if not task_id and params.get("title"):
            task_id = self._find_by_title(tasks, params["title"])
        if task_id not in tasks:
            return ToolResult(success=False, error=f"Task '{task_id}' not found")

        if action == "update":
            if "title" in params:
                tasks[task_id]["title"] = params["title"]
            if "status" in params:
                tasks[task_id]["status"] = params["status"]
            if "notes" in params:
                tasks[task_id]["notes"] = params["notes"]
            self._save(tasks)
            return ToolResult(output=f"Updated {task_id}")

        if action == "complete":
            tasks[task_id]["status"] = "done"
            self._save(tasks)
            return ToolResult(output=f"Completed {task_id}: {tasks[task_id]['title']}")

        if action == "delete":
            title = tasks.pop(task_id)["title"]
            self._save(tasks)
            return ToolResult(output=f"Deleted {task_id}: {title}")

        return ToolResult(success=False, error=f"Unknown action: {action}")

    def _load(self) -> dict[str, Any]:
        path = self._store_dir / "todos.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _save(self, tasks: dict[str, Any]) -> None:
        path = self._store_dir / "todos.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _new_task_id(self, tasks: dict[str, Any]) -> str:
        while True:
            task_id = f"t_{uuid.uuid4().hex[:8]}"
            if task_id not in tasks:
                return task_id

    def _create_items(self, params: dict[str, Any]) -> list[dict[str, str]]:
        raw_items = params.get("items")
        if raw_items:
            items = raw_items if isinstance(raw_items, list) else [raw_items]
        else:
            items = [{"title": params.get("title", "Untitled"), "notes": params.get("notes", "")}]

        normalized: list[dict[str, str]] = []
        for item in items:
            if isinstance(item, str):
                title = item.strip()
                notes = ""
            elif isinstance(item, dict):
                title = str(item.get("title", "")).strip()
                notes = str(item.get("notes", "")).strip()
            else:
                title = str(item).strip()
                notes = ""
            if title:
                normalized.append({"title": title, "notes": notes})
        return normalized or [{"title": "Untitled", "notes": ""}]

    def _find_by_title(self, tasks: dict[str, Any], title: str) -> str:
        matches = [tid for tid, task in tasks.items() if task.get("title") == title]
        return matches[0] if len(matches) == 1 else ""
