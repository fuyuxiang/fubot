"""A2A JSON-RPC protocol handler."""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

from loguru import logger

from echo_agent.a2a.models import A2ATask, A2AMessage, TaskState

# JSON-RPC 2.0 标准错误码
_JSONRPC_PARSE_ERROR = -32700
_JSONRPC_METHOD_NOT_FOUND = -32601
_JSONRPC_INTERNAL_ERROR = -32603


class A2AProtocol:
    """Handles A2A JSON-RPC methods: tasks/send, tasks/get, tasks/cancel."""

    def __init__(self, process_fn: Callable[[A2ATask], Awaitable[A2ATask]]):
        self._process = process_fn
        self._tasks: dict[str, A2ATask] = {}

    async def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """分发 JSON-RPC 请求到对应的处理方法。

        支持的方法: tasks/send（发送任务）, tasks/get（查询任务）, tasks/cancel（取消任务）。
        """
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        try:
            if method == "tasks/send":
                result = await self._handle_send(params)
            elif method == "tasks/get":
                result = self._handle_get(params)
            elif method == "tasks/cancel":
                result = self._handle_cancel(params)
            else:
                return self._error(req_id, _JSONRPC_METHOD_NOT_FOUND, f"Method not found: {method}")
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as e:
            logger.error("A2A protocol error: {}", e)
            return self._error(req_id, _JSONRPC_INTERNAL_ERROR, str(e))

    async def _handle_send(self, params: dict[str, Any]) -> dict[str, Any]:
        """处理任务发送请求。如果任务 ID 已存在则追加消息，否则创建新任务。"""
        task_id = params.get("id", "")
        message = params.get("message", {})

        if task_id and task_id in self._tasks:
            task = self._tasks[task_id]
            task.messages.append(A2AMessage.from_dict(message))
        else:
            task = A2ATask()
            if task_id:
                task.id = task_id
            task.messages.append(A2AMessage.from_dict(message))
            self._tasks[task.id] = task

        task.state = TaskState.WORKING
        task = await self._process(task)
        self._tasks[task.id] = task
        return task.to_dict()

    def _handle_get(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        return task.to_dict()

    def _handle_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        task.state = TaskState.CANCELED
        return task.to_dict()

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
