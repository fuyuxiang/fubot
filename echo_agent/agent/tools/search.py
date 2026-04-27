"""Search files tool — regex and glob search across workspace."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult


class SearchFilesTool(Tool):
    name = "search_files"
    description = "Search file contents by regex pattern or find files by glob pattern within the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search in file contents, or glob pattern for file names."},
            "path": {"type": "string", "description": "Subdirectory to search in (relative to workspace). Defaults to '.'."},
            "mode": {"type": "string", "enum": ["content", "glob"], "description": "Search mode: 'content' for regex in files, 'glob' for filename matching."},
            "max_results": {"type": "integer", "description": "Maximum results to return.", "default": 50},
        },
        "required": ["pattern"],
    }
    timeout_seconds = 30

    def __init__(self, workspace: str, restrict: bool = False):
        self._workspace = Path(workspace).resolve()
        self._restrict = restrict

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        pattern = params["pattern"]
        mode = params.get("mode", "content")
        sub = params.get("path", ".")
        max_results = params.get("max_results", 50)

        search_root = (self._workspace / sub).resolve()
        if self._restrict:
            try:
                search_root.relative_to(self._workspace)
            except ValueError:
                return ToolResult(success=False, error="Path outside workspace")

        if not search_root.is_dir():
            return ToolResult(success=False, error=f"Directory not found: {sub}")

        if mode == "glob":
            return self._glob_search(search_root, pattern, max_results)
        return self._content_search(search_root, pattern, max_results)

    def _glob_search(self, root: Path, pattern: str, limit: int) -> ToolResult:
        matches = []
        for p in root.rglob(pattern):
            if p.is_file():
                matches.append(str(p.relative_to(self._workspace)))
                if len(matches) >= limit:
                    break
        return ToolResult(output="\n".join(matches) if matches else "No files matched.", metadata={"count": len(matches)})

    def _content_search(self, root: Path, pattern: str, limit: int) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}")

        results: list[str] = []
        _SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv"}

        for path in root.rglob("*"):
            if any(part in _SKIP for part in path.parts):
                continue
            if not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            try:
                text = path.read_text(errors="replace")
            except (OSError, PermissionError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = path.relative_to(self._workspace)
                    results.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                    if len(results) >= limit:
                        return ToolResult(output="\n".join(results), metadata={"count": len(results), "truncated": True})

        return ToolResult(output="\n".join(results) if results else "No matches found.", metadata={"count": len(results)})

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"
