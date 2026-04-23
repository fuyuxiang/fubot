"""Patch tool — apply unified diffs with fuzzy matching."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolPermission, ToolResult


class PatchTool(Tool):
    name = "patch"
    description = "Apply a text patch to a file using unified diff format or search-and-replace blocks. Supports fuzzy matching."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to patch (relative to workspace)."},
            "patch": {"type": "string", "description": "Unified diff content, or search/replace block in <<<SEARCH ... === ... REPLACE>>> format."},
            "fuzzy_threshold": {"type": "number", "description": "Similarity threshold for fuzzy matching (0.0-1.0).", "default": 0.6},
        },
        "required": ["file_path", "patch"],
    }
    required_permissions = [ToolPermission.WRITE]
    timeout_seconds = 15

    def __init__(self, workspace: str, restrict: bool = False):
        self._workspace = Path(workspace).resolve()
        self._restrict = restrict

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        rel = params["file_path"]
        target = (self._workspace / rel).resolve()
        if self._restrict:
            try:
                target.relative_to(self._workspace)
            except ValueError:
                return ToolResult(success=False, error="Path outside workspace")

        patch_text = params["patch"]
        threshold = params.get("fuzzy_threshold", 0.6)

        if "<<<SEARCH" in patch_text or "<<<search" in patch_text.lower():
            return self._apply_search_replace(target, patch_text, threshold)
        return self._apply_unified_diff(target, patch_text, threshold)

    def _apply_search_replace(self, target: Path, patch_text: str, threshold: float) -> ToolResult:
        if not target.exists():
            return ToolResult(success=False, error=f"File not found: {target.name}")

        content = target.read_text()
        blocks = self._parse_search_replace(patch_text)
        if not blocks:
            return ToolResult(success=False, error="No valid search/replace blocks found")

        applied = 0
        for search, replace in blocks:
            if search in content:
                content = content.replace(search, replace, 1)
                applied += 1
            else:
                match = self._fuzzy_find(content, search, threshold)
                if match:
                    content = content[:match[0]] + replace + content[match[1]:]
                    applied += 1

        if applied == 0:
            return ToolResult(success=False, error="No blocks matched (even with fuzzy matching)")

        target.write_text(content)
        return ToolResult(output=f"Applied {applied}/{len(blocks)} blocks to {target.name}")

    def _parse_search_replace(self, text: str) -> list[tuple[str, str]]:
        blocks = []
        parts = text.split("<<<SEARCH")
        for part in parts[1:]:
            if "===" not in part or "REPLACE>>>" not in part:
                continue
            search_part, rest = part.split("===", 1)
            replace_part = rest.split("REPLACE>>>", 1)[0]
            blocks.append((search_part.strip(), replace_part.strip()))
        return blocks

    def _apply_unified_diff(self, target: Path, patch_text: str, threshold: float) -> ToolResult:
        original = target.read_text().splitlines(keepends=True) if target.exists() else []
        hunks = self._parse_diff_hunks(patch_text)
        if not hunks:
            return ToolResult(success=False, error="Could not parse unified diff")

        result_lines = list(original)
        offset = 0
        applied = 0

        for start, removals, additions in hunks:
            idx = start - 1 + offset
            if idx < 0:
                idx = 0

            if removals:
                match_pos = self._find_hunk_position(result_lines, removals, idx, threshold)
                if match_pos is not None:
                    result_lines[match_pos:match_pos + len(removals)] = additions
                    offset += len(additions) - len(removals)
                    applied += 1
                    continue
            else:
                result_lines[idx:idx] = additions
                offset += len(additions)
                applied += 1
                continue

        if applied == 0:
            return ToolResult(success=False, error="No hunks could be applied")

        target.write_text("".join(result_lines))
        return ToolResult(output=f"Applied {applied}/{len(hunks)} hunks to {target.name}")

    def _parse_diff_hunks(self, text: str) -> list[tuple[int, list[str], list[str]]]:
        import re
        hunks = []
        hunk_header = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
        lines = text.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            m = hunk_header.match(lines[i])
            if m:
                start = int(m.group(1))
                removals, additions = [], []
                i += 1
                while i < len(lines) and not hunk_header.match(lines[i]):
                    line = lines[i]
                    if line.startswith("-"):
                        removals.append(line[1:])
                    elif line.startswith("+"):
                        additions.append(line[1:])
                    elif line.startswith(" "):
                        removals.append(line[1:])
                        additions.append(line[1:])
                    i += 1
                hunks.append((start, removals, additions))
            else:
                i += 1
        return hunks

    def _find_hunk_position(self, lines: list[str], target: list[str], hint: int, threshold: float) -> int | None:
        target_text = "".join(target)
        window = len(target)
        best_pos, best_ratio = None, threshold
        search_range = range(max(0, hint - 50), min(len(lines) - window + 1, hint + 50))
        for pos in search_range:
            candidate = "".join(lines[pos:pos + window])
            ratio = difflib.SequenceMatcher(None, target_text, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = pos
        return best_pos

    def _fuzzy_find(self, content: str, search: str, threshold: float) -> tuple[int, int] | None:
        lines = content.splitlines(keepends=True)
        search_lines = search.splitlines(keepends=True)
        window = len(search_lines)
        if window == 0:
            return None
        best_pos, best_ratio = None, threshold
        for i in range(len(lines) - window + 1):
            candidate = "".join(lines[i:i + window])
            ratio = difflib.SequenceMatcher(None, search, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                start = sum(len(line) for line in lines[:i])
                end = start + len(candidate)
                best_pos = (start, end)
        return best_pos
