"""MCP tool adapter — wraps MCP tools as echo-agent Tool instances."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult
from echo_agent.mcp.client import MCPClient


def _sanitize_name(server: str, tool: str) -> str:
    raw = f"mcp_{server}_{tool}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw)


def _convert_mcp_schema(mcp_tool: dict[str, Any]) -> dict[str, Any]:
    schema = mcp_tool.get("inputSchema", {})
    if not schema:
        schema = {"type": "object", "properties": {}}
    return schema


class MCPToolAdapter(Tool):

    timeout_seconds = 120

    def __init__(self, server_name: str, mcp_tool: dict[str, Any], client: MCPClient):
        self._server_name = server_name
        self._mcp_tool_name = mcp_tool.get("name", "")
        self._client = client

        self.name = _sanitize_name(server_name, self._mcp_tool_name)
        self.description = mcp_tool.get("description", f"MCP tool from {server_name}")
        self.parameters = _convert_mcp_schema(mcp_tool)

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        try:
            resp = await self._client.call_tool(self._mcp_tool_name, params, timeout=self.timeout_seconds)
        except TimeoutError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            logger.error("MCP tool '{}' call failed: {}", self.name, e)
            return ToolResult(success=False, error=f"MCP call failed: {e}")

        content_parts = resp.get("content", [])
        is_error = resp.get("isError", False)

        text_parts: list[str] = []
        for part in content_parts:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image":
                    text_parts.append(f"[image: {part.get('mimeType', 'unknown')}]")
                elif part.get("type") == "resource":
                    res = part.get("resource", {})
                    text_parts.append(f"[resource: {res.get('uri', '')}]\n{res.get('text', '')}")
            elif isinstance(part, str):
                text_parts.append(part)

        output = "\n".join(text_parts)
        return ToolResult(
            success=not is_error,
            output=output if not is_error else "",
            error=output if is_error else "",
            metadata={"mcp_server": self._server_name, "mcp_tool": self._mcp_tool_name},
        )

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "side_effect"
