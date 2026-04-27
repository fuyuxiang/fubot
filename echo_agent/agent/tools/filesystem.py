"""Filesystem tools — read, write, edit, list directory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult


class _PathSafety:
    """Validates file paths stay within workspace boundaries."""

    _SENSITIVE_EXACT = {
        "/etc/passwd",
        "/private/etc/passwd",
        "/etc/shadow",
        "/private/etc/shadow",
        "/etc/sudoers",
        "/private/etc/sudoers",
        "/etc/gshadow",
        "/private/etc/gshadow",
        "/etc/security/opasswd",
        "/private/etc/security/opasswd",
    }
    _SENSITIVE_PREFIXES = (
        "/root/.ssh/",
        "/root/.gnupg/",
        "/etc/ssh/",
    )

    def __init__(self, workspace: str, restrict: bool = False):
        self._workspace = Path(workspace).resolve()
        self._restrict = restrict

    def resolve(self, path: str) -> Path:
        raw = Path(path).expanduser()
        return raw.resolve() if raw.is_absolute() else (self._workspace / raw).resolve()

    def check(self, path: str) -> str | None:
        resolved = self.resolve(path)
        raw_text = str(Path(path).expanduser())
        resolved_text = str(resolved)
        if raw_text in self._SENSITIVE_EXACT or resolved_text in self._SENSITIVE_EXACT or any(
            resolved_text.startswith(prefix) for prefix in self._SENSITIVE_PREFIXES
        ):
            return f"Path {path} is blocked by sensitive system file policy"
        if not self._restrict:
            return None
        try:
            resolved.relative_to(self._workspace)
            return None
        except ValueError:
            return f"Path {path} is outside workspace {self._workspace}"


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read."},
            "offset": {"type": "integer", "description": "Start line (0-based)."},
            "limit": {"type": "integer", "description": "Max lines to read."},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: str, restrict: bool = False):
        self._safety = _PathSafety(workspace, restrict)

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        path = params["path"]
        violation = self._safety.check(path)
        if violation:
            return ToolResult(success=False, error=violation)
        try:
            target = self._safety.resolve(path)
            with open(target, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            offset = params.get("offset", 0)
            limit = params.get("limit", 2000)
            selected = lines[offset:offset + limit]
            numbered = "".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(selected))
            return ToolResult(output=numbered, metadata={"total_lines": len(lines)})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file, creating it if needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: str, restrict: bool = False):
        self._safety = _PathSafety(workspace, restrict)

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        path = params["path"]
        violation = self._safety.check(path)
        if violation:
            return ToolResult(success=False, error=violation)
        try:
            p = self._safety.resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(params["content"], encoding="utf-8")
            return ToolResult(output=f"Written {len(params['content'])} chars to {path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class EditFileTool(Tool):
    name = "edit_file"
    description = "Replace a string in a file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path."},
            "old_string": {"type": "string", "description": "Text to find."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences.", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, workspace: str, restrict: bool = False):
        self._safety = _PathSafety(workspace, restrict)

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        path = params["path"]
        violation = self._safety.check(path)
        if violation:
            return ToolResult(success=False, error=violation)
        try:
            target = self._safety.resolve(path)
            content = target.read_text(encoding="utf-8")
            old = params["old_string"]
            if old not in content:
                return ToolResult(success=False, error="old_string not found in file")
            if not params.get("replace_all", False) and content.count(old) > 1:
                return ToolResult(success=False, error=f"old_string found {content.count(old)} times, not unique")
            new_content = content.replace(old, params["new_string"]) if params.get("replace_all") else content.replace(old, params["new_string"], 1)
            target.write_text(new_content, encoding="utf-8")
            return ToolResult(output=f"Edited {path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories at a path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path."},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: str, restrict: bool = False):
        self._safety = _PathSafety(workspace, restrict)

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        path = params["path"]
        violation = self._safety.check(path)
        if violation:
            return ToolResult(success=False, error=violation)
        try:
            target = self._safety.resolve(path)
            entries = sorted(os.listdir(target))
            lines = []
            for entry in entries:
                full = os.path.join(target, entry)
                kind = "dir" if os.path.isdir(full) else "file"
                lines.append(f"{kind}\t{entry}")
            return ToolResult(output="\n".join(lines))
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"
