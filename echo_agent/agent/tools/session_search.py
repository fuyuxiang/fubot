"""Session search tool — search past session messages."""

from __future__ import annotations

import re
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolPermission, ToolResult
from echo_agent.session.manager import SessionManager


class SessionSearchTool(Tool):
    name = "session_search"
    description = "Search past conversation messages across sessions by keyword or regex."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (keyword or regex pattern)."},
            "session_key": {"type": "string", "description": "Limit search to a specific session. Omit to search all."},
            "max_results": {"type": "integer", "description": "Maximum results to return.", "default": 20},
            "role_filter": {"type": "string", "enum": ["user", "assistant", "all"], "description": "Filter by message role."},
        },
        "required": ["query"],
    }
    required_permissions = [ToolPermission.READ]

    def __init__(self, session_manager: SessionManager):
        self._sessions = session_manager

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        query = params["query"]
        session_key = params.get("session_key")
        max_results = params.get("max_results", 20)
        role_filter = params.get("role_filter", "all")

        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        results: list[str] = []

        if session_key:
            session = await self._sessions.get_or_create(session_key)
            self._search_session(session, pattern, role_filter, results, max_results)
        else:
            for key in self._sessions.list_sessions():
                if len(results) >= max_results:
                    break
                session = await self._sessions.get_or_create(key)
                self._search_session(session, pattern, role_filter, results, max_results)

        if not results:
            return ToolResult(output="No matches found.")
        return ToolResult(output="\n---\n".join(results), metadata={"count": len(results)})

    def _search_session(self, session, pattern, role_filter, results, limit):
        for msg in session.messages:
            if len(results) >= limit:
                return
            role = msg.get("role", "")
            if role_filter != "all" and role != role_filter:
                continue
            content = msg.get("content", "")
            if not content or not isinstance(content, str):
                continue
            if pattern.search(content):
                snippet = content[:300]
                results.append(f"[{session.key}] {role}: {snippet}")

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"
