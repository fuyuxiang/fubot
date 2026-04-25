"""A2A client — discover and delegate to remote agents."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import aiohttp
from loguru import logger

from echo_agent.a2a.models import AgentCard, A2ATask, A2AMessage


class A2AClient:
    def __init__(self, timeout: float = 60):
        self._timeout = timeout

    async def discover(self, base_url: str) -> AgentCard:
        url = f"{base_url.rstrip('/')}/.well-known/agent.json"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                data = await resp.json()
                return AgentCard(
                    name=data.get("name", ""),
                    description=data.get("description", ""),
                    url=base_url,
                    version=data.get("version", ""),
                    capabilities=[],
                )

    async def send_task(self, base_url: str, message: str, task_id: str = "") -> A2ATask:
        url = f"{base_url.rstrip('/')}/a2a"
        payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": A2AMessage.text("user", message).to_dict(),
            },
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                data = await resp.json()
                result = data.get("result", {})
                return A2ATask.from_dict(result)

    async def get_task(self, base_url: str, task_id: str) -> A2ATask:
        url = f"{base_url.rstrip('/')}/a2a"
        payload = {
            "jsonrpc": "2.0", "id": "1",
            "method": "tasks/get",
            "params": {"id": task_id},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                data = await resp.json()
                return A2ATask.from_dict(data.get("result", {}))

    async def cancel_task(self, base_url: str, task_id: str) -> A2ATask:
        url = f"{base_url.rstrip('/')}/a2a"
        payload = {
            "jsonrpc": "2.0", "id": "1",
            "method": "tasks/cancel",
            "params": {"id": task_id},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                data = await resp.json()
                return A2ATask.from_dict(data.get("result", {}))
