"""Web tools — fetch URLs and search the web."""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

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

    def __init__(
        self,
        api_key: str = "",
        *,
        provider: str = "brave",
        api_base: str = "",
        proxy: str | None = None,
        timeout_seconds: int = 30,
    ):
        self._api_key = api_key
        self._provider = provider.lower().strip()
        self._api_base = api_base.strip()
        self._proxy = proxy
        self.timeout_seconds = timeout_seconds

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        query = params["query"].strip()
        max_results = max(1, min(int(params.get("max_results", 5)), 20))
        if not query:
            return ToolResult(success=False, error="query is required")
        if self._provider != "searxng" and not self._api_key:
            return ToolResult(success=False, error=f"{self._provider} search API key not configured")

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session:
                if self._provider == "brave":
                    results = await self._search_brave(session, query, max_results)
                elif self._provider == "tavily":
                    results = await self._search_tavily(session, query, max_results)
                elif self._provider == "serpapi":
                    results = await self._search_serpapi(session, query, max_results)
                elif self._provider == "searxng":
                    results = await self._search_searxng(session, query, max_results)
                else:
                    return ToolResult(success=False, error=f"Unsupported search provider: {self._provider}")
        except Exception as e:
            return ToolResult(success=False, error=str(e), metadata={"provider": self._provider})

        if not results:
            return ToolResult(output="No search results.", metadata={"provider": self._provider, "count": 0})

        lines: list[str] = [f"Search results for: {query}"]
        for idx, item in enumerate(results[:max_results], 1):
            title = item.get("title", "").strip() or "(untitled)"
            url = item.get("url", "").strip()
            snippet = item.get("snippet", "").strip()
            lines.append(f"[{idx}] {title}\nURL: {url}\nSnippet: {snippet}".rstrip())
        return ToolResult(
            output="\n\n".join(lines),
            metadata={"provider": self._provider, "count": len(results), "results": results[:max_results]},
        )

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"

    def _base_url(self, default: str) -> str:
        return self._api_base or default

    async def _read_json(self, resp: aiohttp.ClientResponse) -> dict[str, Any]:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"search provider returned HTTP {resp.status}: {text[:500]}")
        try:
            return await resp.json(content_type=None)
        except Exception as exc:
            raise RuntimeError(f"search provider returned invalid JSON: {text[:500]}") from exc

    async def _search_brave(self, session: aiohttp.ClientSession, query: str, max_results: int) -> list[dict[str, str]]:
        url = self._base_url("https://api.search.brave.com/res/v1/web/search")
        headers = {"Accept": "application/json", "X-Subscription-Token": self._api_key}
        params = {"q": query, "count": max_results}
        async with session.get(url, headers=headers, params=params, proxy=self._proxy) as resp:
            data = await self._read_json(resp)
        raw = data.get("web", {}).get("results", [])
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            }
            for item in raw
        ]

    async def _search_tavily(self, session: aiohttp.ClientSession, query: str, max_results: int) -> list[dict[str, str]]:
        url = self._base_url("https://api.tavily.com/search")
        payload = {"api_key": self._api_key, "query": query, "max_results": max_results, "include_answer": False}
        async with session.post(url, json=payload, proxy=self._proxy) as resp:
            data = await self._read_json(resp)
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            for item in data.get("results", [])
        ]

    async def _search_serpapi(self, session: aiohttp.ClientSession, query: str, max_results: int) -> list[dict[str, str]]:
        url = self._base_url("https://serpapi.com/search.json")
        params = {"engine": "google", "q": query, "api_key": self._api_key, "num": max_results}
        async with session.get(url, params=params, proxy=self._proxy) as resp:
            data = await self._read_json(resp)
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            for item in data.get("organic_results", [])
        ]

    async def _search_searxng(self, session: aiohttp.ClientSession, query: str, max_results: int) -> list[dict[str, str]]:
        if not self._api_base:
            raise RuntimeError("SearXNG search_api_base is required")
        url = urljoin(self._api_base.rstrip("/") + "/", "search")
        params = {"q": query, "format": "json", "categories": "general", "language": "auto"}
        async with session.get(url, params=params, proxy=self._proxy) as resp:
            data = await self._read_json(resp)
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            for item in data.get("results", [])[:max_results]
        ]
