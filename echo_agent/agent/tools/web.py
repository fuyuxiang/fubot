"""Web tools — fetch URLs and search the web."""

from __future__ import annotations

from typing import Any

import aiohttp

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolPermission, ToolResult


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch content from a URL."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch."},
            "max_chars": {"type": "integer", "description": "Max response chars.", "default": 16000},
        },
        "required": ["url"],
    }
    required_permissions = [ToolPermission.NETWORK]
    timeout_seconds = 30

    def __init__(self, proxy: str | None = None):
        self._proxy = proxy

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        url = params["url"]
        max_chars = params.get("max_chars", 16000)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, proxy=self._proxy, timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as resp:
                    text = await resp.text()
                    original_len = len(text)
                    if len(text) > max_chars:
                        text = text[:max_chars] + f"\n... (truncated, {original_len} total)"
                    content_type = resp.headers.get("content-type", "")
                    header = (
                        f"HTTP {resp.status} {resp.reason or ''}\n"
                        f"URL: {resp.url}\n"
                        f"Content-Type: {content_type}\n\n"
                    )
                    metadata = {"status": resp.status, "url": str(resp.url), "content_type": content_type}
                    if resp.status >= 400:
                        return ToolResult(success=False, error=header + text, metadata=metadata)
                    return ToolResult(output=header + text, metadata=metadata)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for information."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {"type": "integer", "description": "Max results.", "default": 5},
        },
        "required": ["query"],
    }
    required_permissions = [ToolPermission.NETWORK]
    timeout_seconds = 30

    def __init__(self, api_key: str = "", proxy: str | None = None):
        self._api_key = api_key
        self._proxy = proxy

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        if not self._api_key:
            return ToolResult(success=False, error="Web search API key not configured")
        return ToolResult(success=False, error="Web search provider not implemented — configure a search API")

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"
