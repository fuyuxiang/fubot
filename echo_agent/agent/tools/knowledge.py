"""Knowledge base tools."""

from __future__ import annotations

import json
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult
from echo_agent.knowledge.index import KnowledgeIndex


class KnowledgeSearchTool(Tool):
    name = "knowledge_search"
    description = "Search the local internal knowledge base and return cited snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {"type": "integer", "description": "Maximum cited snippets to return.", "default": 5},
        },
        "required": ["query"],
    }
    timeout_seconds = 10

    def __init__(self, index: KnowledgeIndex, default_limit: int = 5):
        self._index = index
        self._default_limit = default_limit

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        query = params["query"].strip()
        limit = max(1, min(int(params.get("max_results", self._default_limit)), 20))
        if not query:
            return ToolResult(success=False, error="query is required")
        user_id = ctx.user_id if ctx else ""
        results = self._index.search(query, limit=limit, user_id=user_id)
        text = self._index.format_results(results)
        if not text:
            return ToolResult(output="No matching internal knowledge found.", metadata={"count": 0})
        return ToolResult(
            output=text,
            metadata={
                "count": len(results),
                "citations": [
                    {
                        "id": result.citation_id,
                        "path": result.path,
                        "chunk_id": result.chunk_id,
                        "score": result.score,
                    }
                    for result in results
                ],
            },
        )

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"


class KnowledgeIndexTool(Tool):
    name = "knowledge_index"
    description = "Inspect or rebuild the local internal knowledge index."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "rebuild"], "description": "Index action."},
        },
        "required": ["action"],
    }
    timeout_seconds = 60

    def __init__(self, index: KnowledgeIndex):
        self._index = index

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        action = params["action"]
        if action == "status":
            return ToolResult(output=json.dumps(self._index.status(), ensure_ascii=False, indent=2))
        if action == "rebuild":
            return ToolResult(output=json.dumps(self._index.rebuild(), ensure_ascii=False, indent=2))
        return ToolResult(success=False, error=f"Unsupported action: {action}")
