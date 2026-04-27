"""Shell execution tool — runs commands with isolation and safety controls."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from echo_agent.agent.executors.base import BaseExecutor, ExecRequest
from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult


class ShellTool(Tool):
    name = "exec"
    description = "Execute a shell command in the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute."},
            "timeout": {"type": "integer", "description": "Timeout in seconds.", "default": 30},
            "cwd": {"type": "string", "description": "Working directory override."},
        },
        "required": ["command"],
    }
    timeout_seconds = 60
    _BUILTIN_BLOCKED_PATTERNS = [
        (re.compile(r"/etc/(passwd|shadow|sudoers|gshadow)\b"), "sensitive system account file"),
        (re.compile(r"(^|\s)(cat|less|more|head|tail|sed|awk|grep)\s+[^\n;|&]*(/root/\.ssh|/etc/ssh|/root/\.gnupg)"), "sensitive credential path"),
        (re.compile(r"\brm\s+-[^\n;|&]*[rf][^\n;|&]*\s+/(?:\s|$)"), "destructive root removal"),
        (re.compile(r"\bdd\s+[^\n;|&]*\bof=/dev/"), "destructive block device write"),
        (re.compile(r"\bmkfs(?:\.\w+)?\b"), "filesystem formatting"),
        (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "system shutdown"),
    ]

    def __init__(
        self,
        workspace: str,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        max_output: int = 16000,
        executor: BaseExecutor | None = None,
    ):
        self._workspace = str(Path(workspace).resolve())
        self._allowed = allowed or []
        self._blocked = blocked or []
        self._max_output = max_output
        self._executor = executor

    def _check_command(self, command: str) -> str | None:
        cmd_name = command.strip().split()[0] if command.strip() else ""
        for pattern, reason in self._BUILTIN_BLOCKED_PATTERNS:
            if pattern.search(command):
                return f"Command blocked by safety policy: {reason}"
        for pattern in self._blocked:
            if pattern in command:
                return f"Command blocked: contains '{pattern}'"
        if self._allowed and cmd_name not in self._allowed:
            return f"Command not in allowlist: {cmd_name}"
        return None

    def _resolve_cwd(self, cwd: str) -> str:
        raw = Path(cwd).expanduser()
        resolved = raw.resolve() if raw.is_absolute() else (Path(self._workspace) / raw).resolve()
        resolved.relative_to(Path(self._workspace))
        return str(resolved)

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        command = params["command"]
        timeout = params.get("timeout", 30)
        cwd = params.get("cwd", self._workspace)

        violation = self._check_command(command)
        if violation:
            return ToolResult(success=False, error=violation)

        try:
            try:
                cwd = self._resolve_cwd(cwd)
            except ValueError:
                return ToolResult(success=False, error=f"cwd is outside workspace: {cwd}")
            if self._executor:
                response = await self._executor.execute(ExecRequest(
                    command=command,
                    cwd=cwd,
                    timeout=timeout,
                    env={"WORKSPACE": self._workspace},
                    credentials=ctx.credentials if ctx else {},
                ))
                output = response.stdout
                err_output = response.stderr
                return_code = response.return_code
                executor_name = response.executor
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env={**os.environ, "WORKSPACE": self._workspace},
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode(errors="replace")
                err_output = stderr.decode(errors="replace")
                return_code = proc.returncode or 0
                executor_name = "direct"

            if len(output) > self._max_output:
                output = output[:self._max_output] + f"\n... (truncated, {len(output)} total chars)"

            combined = output
            if err_output:
                combined += f"\nSTDERR:\n{err_output}"

            return ToolResult(
                success=return_code == 0,
                output=combined,
                error=err_output if return_code != 0 else "",
                metadata={"return_code": return_code, "executor": executor_name},
            )
        except asyncio.TimeoutError:
            return ToolResult(success=False, error=f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "side_effect"
