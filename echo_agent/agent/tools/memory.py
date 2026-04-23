"""Agent-facing memory tool — add, replace, remove, search, list memories.

Gives the LLM direct control over what to remember about the user,
project, and environment across sessions.
"""

from __future__ import annotations

from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolPermission, ToolResult
from echo_agent.memory.store import MemoryEntry, MemoryStore, MemoryType


class MemoryTool(Tool):
    name = "memory"
    description = (
        "Manage persistent memory across sessions. Actions: "
        "add (save a new memory), replace (update existing by key or substring match), "
        "remove (delete by key or substring), search (find relevant memories), "
        "list (show all memories of a type)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "search", "list"],
                "description": "The operation to perform",
            },
            "target": {
                "type": "string",
                "enum": ["user", "environment"],
                "description": "Memory type: 'user' for preferences/habits, 'environment' for project/tool facts",
            },
            "key": {
                "type": "string",
                "description": "Short label for the memory entry (for add/replace)",
            },
            "content": {
                "type": "string",
                "description": "Memory content (for add/replace)",
            },
            "old_text": {
                "type": "string",
                "description": "Substring to find the entry to replace/remove",
            },
            "query": {
                "type": "string",
                "description": "Search query (for search action)",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags (for add)",
            },
            "importance": {
                "type": "number",
                "description": "Importance score 0.0-1.0 (default 0.5)",
            },
        },
        "required": ["action"],
    }
    required_permissions = [ToolPermission.WRITE]

    def __init__(self, store: MemoryStore):
        self._store = store

    def _resolve_entry(
        self,
        key: str,
        old_text: str,
        mem_type: MemoryType,
        session_key: str = "",
    ) -> tuple[MemoryEntry | None, str | None]:
        if key:
            entry = self._store.find_by_key(key, mem_type, session_key=session_key)
            if entry:
                return entry, None

        if old_text:
            matches = self._store.find_by_content_matches(
                old_text,
                mem_type=mem_type,
                limit=6,
                session_key=session_key,
            )
            if not matches:
                return None, None
            if len(matches) > 1:
                previews = ", ".join(entry.key or entry.content[:30] for entry in matches[:5])
                return None, (
                    f"Multiple matching memories found for old_text='{old_text}'. "
                    f"Be more specific. Matches: {previews}"
                )
            return matches[0], None

        return None, None

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        action = params.get("action", "")
        target = params.get("target", "user")
        mem_type = MemoryType.USER if target == "user" else MemoryType.ENVIRONMENT
        session_key = ctx.session_key if ctx else ""

        if action == "add":
            return self._add(params, mem_type, session_key)
        elif action == "replace":
            return self._replace(params, mem_type, session_key)
        elif action == "remove":
            return self._remove(params, mem_type, session_key)
        elif action == "search":
            return self._search(params, mem_type, session_key)
        elif action == "list":
            return self._list(mem_type, session_key)
        return ToolResult(success=False, error=f"Unknown action '{action}'")

    def _add(self, params: dict[str, Any], mem_type: MemoryType, session_key: str) -> ToolResult:
        key = params.get("key", "")
        content = params.get("content", "")
        if not key or not content:
            return ToolResult(success=False, error="key and content are required for add")
        tags_str = params.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        importance = min(1.0, max(0.0, params.get("importance", 0.5)))

        entry = MemoryEntry(
            type=mem_type, key=key, content=content,
            tags=tags, importance=importance,
            source_session=session_key if mem_type == MemoryType.USER else "",
        )
        try:
            result = self._store.add(entry)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=True, output=f"Memory saved: [{result.type.value}] {result.key}")

    def _replace(self, params: dict[str, Any], mem_type: MemoryType, session_key: str) -> ToolResult:
        key = params.get("key", "")
        old_text = params.get("old_text", "")
        content = params.get("content", "")
        if not content:
            return ToolResult(success=False, error="content is required for replace")

        entry, resolve_error = self._resolve_entry(key, old_text, mem_type, session_key)
        if resolve_error:
            return ToolResult(success=False, error=resolve_error)
        if not entry:
            return ToolResult(success=False, error=f"No matching memory found for key='{key}' old_text='{old_text}'")

        try:
            self._store.update(entry.id, content=content)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=True, output=f"Memory updated: [{entry.type.value}] {entry.key}")

    def _remove(self, params: dict[str, Any], mem_type: MemoryType, session_key: str) -> ToolResult:
        key = params.get("key", "")
        old_text = params.get("old_text", "")

        entry, resolve_error = self._resolve_entry(key, old_text, mem_type, session_key)
        if resolve_error:
            return ToolResult(success=False, error=resolve_error)
        if not entry:
            return ToolResult(success=False, error=f"No matching memory found for key='{key}' old_text='{old_text}'")

        self._store.delete(entry.id)
        return ToolResult(success=True, output=f"Memory removed: [{entry.type.value}] {entry.key}")

    def _search(self, params: dict[str, Any], mem_type: MemoryType, session_key: str) -> ToolResult:
        query = params.get("query", "")
        if not query:
            return ToolResult(success=False, error="query is required for search")

        results = self._store.search_scored(query, mem_type, limit=10, session_key=session_key)
        if not results:
            return ToolResult(success=True, output="No matching memories found.")

        lines = []
        for entry, score in results:
            tags = f" [{', '.join(entry.tags)}]" if entry.tags else ""
            lines.append(f"- [{entry.type.value}] **{entry.key}**{tags} (score={score:.2f}): {entry.content}")
        return ToolResult(success=True, output="\n".join(lines))

    def _list(self, mem_type: MemoryType, session_key: str) -> ToolResult:
        entries = self._store.list_all(mem_type, session_key=session_key)
        if not entries:
            return ToolResult(success=True, output=f"No {mem_type.value} memories found.")

        lines = []
        for e in entries[:50]:
            tags = f" [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(f"- **{e.key}**{tags}: {e.content}")
        total = len(entries)
        if total > 50:
            lines.append(f"... and {total - 50} more")
        return ToolResult(success=True, output="\n".join(lines))
