"""Agent-facing skill tools — skills_list, skill_view, skill_manage.

Provides the LLM with progressive-disclosure access to the skill store
and the ability to create/edit/delete skills for self-learning.
"""

from __future__ import annotations

import json
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult
from echo_agent.skills.store import SkillStore


class SkillsListTool(Tool):
    name = "skills_list"
    description = (
        "List all available skills with compact metadata (name, description, category, version). "
        "Use this to discover what skills exist before viewing or managing them."
    )
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, store: SkillStore):
        self._store = store

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        skills = self._store.list_all()
        if not skills:
            return ToolResult(success=True, output="No skills found.")
        data = [s.to_dict() for s in skills]
        return ToolResult(success=True, output=json.dumps(data, ensure_ascii=False, indent=2))


class SkillViewTool(Tool):
    name = "skill_view"
    description = (
        "View the full content of a skill (SKILL.md) or a specific supporting file. "
        "Without file_path, returns the full SKILL.md and lists linked files. "
        "With file_path, returns that specific file from references/, templates/, scripts/, or assets/."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name to view"},
            "file_path": {
                "type": "string",
                "description": "Optional path to a supporting file (e.g. 'references/api.md')",
            },
        },
        "required": ["name"],
    }

    def __init__(self, store: SkillStore):
        self._store = store

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        name = params["name"]
        file_path = params.get("file_path", "")

        if file_path:
            content = self._store.read_file(name, file_path)
            if content is None:
                return ToolResult(success=False, error=f"File '{file_path}' not found in skill '{name}'")
            return ToolResult(success=True, output=content)

        content = self._store.read_skill(name)
        if content is None:
            return ToolResult(success=False, error=f"Skill '{name}' not found")

        files = self._store.list_files(name)
        output = content
        if files:
            output += "\n\n---\nLinked files:\n" + "\n".join(f"  - {f}" for f in files)
        return ToolResult(success=True, output=output)


class SkillManageTool(Tool):
    name = "skill_manage"
    description = (
        "Create, edit, patch, or delete skills. Use this to capture reusable knowledge "
        "from completed tasks. Actions: create, edit, patch, delete, write_file, remove_file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "edit", "patch", "delete", "write_file", "remove_file"],
                "description": "The operation to perform",
            },
            "name": {"type": "string", "description": "Skill name (lowercase, alphanumeric, hyphens)"},
            "category": {"type": "string", "description": "Optional category directory for create"},
            "content": {
                "type": "string",
                "description": "SKILL.md content with YAML frontmatter (for create/edit), or file content (for write_file)",
            },
            "file_path": {
                "type": "string",
                "description": "Supporting file path for patch/write_file/remove_file (e.g. 'references/notes.md')",
            },
            "old_text": {"type": "string", "description": "Text to find (for patch action)"},
            "new_text": {"type": "string", "description": "Replacement text (for patch action)"},
        },
        "required": ["action", "name"],
    }

    def __init__(self, store: SkillStore):
        self._store = store

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        action = params["action"]
        name = params["name"]

        if action == "create":
            content = params.get("content", "")
            if not content:
                return ToolResult(success=False, error="content is required for create")
            err = self._store.create_skill(name, content, category=params.get("category", ""))
            if err:
                return ToolResult(success=False, error=err)
            return ToolResult(success=True, output=f"Skill '{name}' created.")

        elif action == "edit":
            content = params.get("content", "")
            if not content:
                return ToolResult(success=False, error="content is required for edit")
            err = self._store.update_skill(name, content)
            if err:
                return ToolResult(success=False, error=err)
            return ToolResult(success=True, output=f"Skill '{name}' updated.")

        elif action == "patch":
            old_text = params.get("old_text", "")
            new_text = params.get("new_text", "")
            if not old_text:
                return ToolResult(success=False, error="old_text is required for patch")
            err = self._store.patch_skill(name, old_text, new_text, file_path=params.get("file_path", ""))
            if err:
                return ToolResult(success=False, error=err)
            return ToolResult(success=True, output=f"Skill '{name}' patched.")

        elif action == "delete":
            err = self._store.delete_skill(name)
            if err:
                return ToolResult(success=False, error=err)
            return ToolResult(success=True, output=f"Skill '{name}' deleted.")

        elif action == "write_file":
            file_path = params.get("file_path", "")
            content = params.get("content", "")
            if not file_path or not content:
                return ToolResult(success=False, error="file_path and content are required for write_file")
            err = self._store.write_file(name, file_path, content)
            if err:
                return ToolResult(success=False, error=err)
            return ToolResult(success=True, output=f"File '{file_path}' written to skill '{name}'.")

        elif action == "remove_file":
            file_path = params.get("file_path", "")
            if not file_path:
                return ToolResult(success=False, error="file_path is required for remove_file")
            err = self._store.remove_file(name, file_path)
            if err:
                return ToolResult(success=False, error=err)
            return ToolResult(success=True, output=f"File '{file_path}' removed from skill '{name}'.")

        return ToolResult(success=False, error=f"Unknown action '{action}'")
