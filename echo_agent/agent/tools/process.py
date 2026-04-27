"""Process management tool — start, list, poll, and stop background processes."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult

_PROCESSES: dict[str, dict[str, Any]] = {}


class ProcessTool(Tool):
    name = "process"
    description = "Manage background processes: start, list, poll output, or stop."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "list", "poll", "stop"], "description": "Action to perform."},
            "command": {"type": "string", "description": "Shell command to run (for 'start')."},
            "process_id": {"type": "string", "description": "Process ID (for 'poll'/'stop')."},
            "timeout": {"type": "integer", "description": "Timeout in seconds for start.", "default": 300},
        },
        "required": ["action"],
    }
    timeout_seconds = 10

    def __init__(self, workspace: str):
        self._workspace = workspace

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        action = params["action"]

        if action == "start":
            return await self._start(params)
        elif action == "list":
            return self._list()
        elif action == "poll":
            return await self._poll(params.get("process_id", ""))
        elif action == "stop":
            return await self._stop(params.get("process_id", ""))
        return ToolResult(success=False, error=f"Unknown action: {action}")

    async def _start(self, params: dict[str, Any]) -> ToolResult:
        cmd = params.get("command", "")
        if not cmd:
            return ToolResult(success=False, error="No command provided")

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._workspace,
        )
        pid = f"proc_{proc.pid}"
        _PROCESSES[pid] = {
            "process": proc,
            "command": cmd,
            "started": time.time(),
            "stdout_buf": b"",
            "stderr_buf": b"",
        }

        asyncio.create_task(self._collect_output(pid))
        return ToolResult(output=f"Started process {pid}: {cmd}", metadata={"process_id": pid})

    def _list(self) -> ToolResult:
        if not _PROCESSES:
            return ToolResult(output="No background processes.")
        lines = []
        for pid, info in _PROCESSES.items():
            proc = info["process"]
            status = "running" if proc.returncode is None else f"exited({proc.returncode})"
            elapsed = int(time.time() - info["started"])
            lines.append(f"{pid}: [{status}] {elapsed}s — {info['command'][:80]}")
        return ToolResult(output="\n".join(lines))

    async def _poll(self, pid: str) -> ToolResult:
        if pid not in _PROCESSES:
            return ToolResult(success=False, error=f"Process '{pid}' not found")
        info = _PROCESSES[pid]
        proc = info["process"]
        status = "running" if proc.returncode is None else f"exited({proc.returncode})"
        stdout = info["stdout_buf"].decode(errors="replace")[-8000:]
        stderr = info["stderr_buf"].decode(errors="replace")[-4000:]
        output = f"[{status}]\n{stdout}"
        if stderr:
            output += f"\n[stderr]\n{stderr}"
        return ToolResult(output=output, metadata={"status": status})

    async def _stop(self, pid: str) -> ToolResult:
        if pid not in _PROCESSES:
            return ToolResult(success=False, error=f"Process '{pid}' not found")
        info = _PROCESSES[pid]
        proc = info["process"]
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        del _PROCESSES[pid]
        return ToolResult(output=f"Stopped {pid}")

    async def _collect_output(self, pid: str) -> None:
        info = _PROCESSES.get(pid)
        if not info:
            return
        proc = info["process"]
        try:
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                info["stdout_buf"] += chunk
                if len(info["stdout_buf"]) > 100_000:
                    info["stdout_buf"] = info["stdout_buf"][-50_000:]
        except Exception as e:
            logger.debug("Error reading process stdout: {}", e)
