"""MCP manager — orchestrates multiple MCP server connections and tool registration."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

from echo_agent.agent.tools.registry import ToolRegistry
from echo_agent.config.schema import MCPServerConfig
from echo_agent.mcp.client import MCPClient
from echo_agent.mcp.security import validate_mcp_tools
from echo_agent.mcp.tool_adapter import MCPToolAdapter
from echo_agent.mcp.transport import HttpTransport, StdioTransport, StreamableHttpTransport


_RECONNECT_DELAYS = (1, 2, 4, 8, 16, 30, 60)
_MAX_RECONNECT_ATTEMPTS = 5


class MCPManager:

    def __init__(self, workspace: Path, security_policy: str = "block"):
        self._workspace = workspace
        self._security_policy = security_policy
        self._clients: dict[str, MCPClient] = {}
        self._configs: dict[str, MCPServerConfig] = {}
        self._registered_tools: dict[str, list[str]] = {}

    async def start_all(self, servers: dict[str, MCPServerConfig]) -> None:
        tasks = []
        for name, cfg in servers.items():
            if not cfg.enabled:
                logger.debug("MCP server '{}' disabled, skipping", name)
                continue
            self._configs[name] = cfg
            tasks.append(self._connect_server(name, cfg))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip([n for n, c in servers.items() if c.enabled], results):
                if isinstance(result, Exception):
                    logger.error("Failed to connect MCP server '{}': {}", name, result)

    async def stop_all(self) -> None:
        tasks = [client.disconnect() for client in self._clients.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._clients.clear()
        self._registered_tools.clear()
        logger.info("All MCP servers disconnected")

    async def discover_tools(self, registry: ToolRegistry) -> int:
        total = 0
        builtin_names = set(registry.tool_names)

        for name, client in self._clients.items():
            if not client.is_connected:
                continue
            try:
                count = await self._register_server_tools(name, client, registry, builtin_names)
                total += count
            except Exception as e:
                logger.error("Tool discovery failed for '{}': {}", name, e)

        logger.info("Discovered {} MCP tools from {} servers", total, len(self._clients))
        return total

    async def refresh_tools(self, server_name: str, registry: ToolRegistry) -> int:
        client = self._clients.get(server_name)
        if not client or not client.is_connected:
            return 0

        for tool_name in self._registered_tools.get(server_name, []):
            registry.unregister(tool_name)
        self._registered_tools[server_name] = []

        builtin_names = set(registry.tool_names)
        return await self._register_server_tools(server_name, client, registry, builtin_names)

    @property
    def connected_servers(self) -> list[str]:
        return [name for name, client in self._clients.items() if client.is_connected]

    async def _connect_server(self, name: str, cfg: MCPServerConfig) -> None:
        transport = await self._create_transport(name, cfg)
        client = MCPClient(name, transport)

        for attempt in range(_MAX_RECONNECT_ATTEMPTS):
            try:
                await client.connect(timeout=cfg.connect_timeout)
                self._clients[name] = client
                logger.info("Connected to MCP server '{}'", name)
                return
            except Exception as e:
                if attempt < _MAX_RECONNECT_ATTEMPTS - 1:
                    delay = _RECONNECT_DELAYS[min(attempt, len(_RECONNECT_DELAYS) - 1)]
                    logger.warning("MCP '{}' connect attempt {} failed: {}. Retrying in {}s", name, attempt + 1, e, delay)
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(f"Failed to connect to MCP server '{name}' after {_MAX_RECONNECT_ATTEMPTS} attempts: {e}")

    async def _create_transport(self, name: str, cfg: MCPServerConfig) -> Any:
        if cfg.url:
            headers = self._resolve_env_vars(cfg.headers)
            if cfg.auth == "oauth":
                token_dir = self._workspace / "data" / "mcp_tokens"
                from echo_agent.mcp.oauth import MCPOAuthClient
                oauth = MCPOAuthClient(name, cfg.url, token_dir)
                token = await oauth.ensure_token()
                headers["Authorization"] = f"Bearer {token}"
            return StreamableHttpTransport(url=cfg.url, headers=headers)
        elif cfg.command:
            env = self._resolve_env_vars(cfg.env)
            return StdioTransport(command=cfg.command, args=cfg.args, env=env)
        else:
            raise ValueError(f"MCP server '{name}' has neither 'url' nor 'command' configured")

    async def _register_server_tools(
        self, name: str, client: MCPClient, registry: ToolRegistry, builtin_names: set[str],
    ) -> int:
        raw_tools = await client.list_tools()
        cfg = self._configs.get(name)
        include = cfg.tools_include if cfg else []
        exclude = cfg.tools_exclude if cfg else []

        accepted = validate_mcp_tools(
            server_name=name,
            tools=raw_tools,
            builtin_names=builtin_names,
            include_filter=include or None,
            exclude_filter=exclude or None,
            policy=self._security_policy,
        )

        registered_names: list[str] = []
        for mcp_tool in accepted:
            adapter = MCPToolAdapter(server_name=name, mcp_tool=mcp_tool, client=client)
            if cfg:
                adapter.timeout_seconds = cfg.timeout
            registry.register(adapter)
            registered_names.append(adapter.name)
            builtin_names.add(adapter.name)

        self._registered_tools[name] = registered_names
        logger.info("Registered {} tools from MCP server '{}'", len(registered_names), name)
        return len(registered_names)

    def _resolve_env_vars(self, mapping: dict[str, str]) -> dict[str, str]:
        resolved = {}
        for key, value in mapping.items():
            resolved[key] = re.sub(
                r"\$\{([^}]+)\}",
                lambda m: os.environ.get(m.group(1), m.group(0)),
                value,
            )
        return resolved
