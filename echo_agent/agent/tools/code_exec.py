"""Code execution tool — run Python/JS/Bash snippets through the configured executor."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from echo_agent.agent.executors.base import BaseExecutor, ExecRequest
from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolPermission, ToolResult


_RUNNERS: dict[str, str] = {
    "python": "python3 -",
    "javascript": "node",
    "bash": "bash",
}


class CodeExecTool(Tool):
    name = "execute_code"
    description = "Execute a code snippet in a sandboxed subprocess. Supports Python, JavaScript, and Bash."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The code to execute."},
            "language": {"type": "string", "enum": ["python", "javascript", "bash"], "description": "Programming language."},
            "timeout": {"type": "integer", "description": "Execution timeout in seconds.", "default": 30},
        },
        "required": ["code", "language"],
    }
    required_permissions = [ToolPermission.EXECUTE]
    timeout_seconds = 60
    _SENSITIVE_PATTERNS = [
        re.compile(r"/etc/(passwd|shadow|sudoers|gshadow)\b"),
        re.compile(r"/root/\.ssh|/etc/ssh|/root/\.gnupg"),
    ]

    def __init__(
        self,
        workspace: str,
        *,
        executor: BaseExecutor | None = None,
        allowed_languages: list[str] | None = None,
        max_output: int = 32000,
        timeout_seconds: int = 60,
    ):
        self._workspace = Path(workspace)
        self._executor = executor
        self._allowed_languages = set(allowed_languages or _RUNNERS.keys())
        self._max_output = max_output
        self.timeout_seconds = timeout_seconds
        enum = [lang for lang in _RUNNERS if lang in self._allowed_languages]
        self.parameters = {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The code to execute."},
                "language": {"type": "string", "enum": enum, "description": "Programming language."},
                "timeout": {"type": "integer", "description": "Execution timeout in seconds.", "default": 30},
            },
            "required": ["code", "language"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        code = params["code"]
        lang = params["language"]
        timeout = min(int(params.get("timeout", 30)), self.timeout_seconds)
        for pattern in self._SENSITIVE_PATTERNS:
            if pattern.search(code):
                return ToolResult(success=False, error="Code blocked by sensitive system file policy")

        if lang not in self._allowed_languages:
            return ToolResult(success=False, error=f"Language not allowed: {lang}")

        command = _RUNNERS.get(lang)
        if not command:
            return ToolResult(success=False, error=f"Unsupported language: {lang}")

        try:
            if self._executor:
                response = await self._executor.execute(ExecRequest(
                    command=command,
                    cwd=str(self._workspace),
                    timeout=timeout,
                    stdin=code,
                    env={"WORKSPACE": str(self._workspace)},
                    credentials=ctx.credentials if ctx else {},
                ))
                out = response.stdout
                err = response.stderr
                exit_code = response.return_code
                executor_name = response.executor
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.PIPE,
                    cwd=str(self._workspace),
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(code.encode()), timeout=timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return ToolResult(success=False, error=f"Execution timed out after {timeout}s")
                out = stdout.decode(errors="replace")
                err = stderr.decode(errors="replace")
                exit_code = proc.returncode or 0
                executor_name = "direct"

            output = ""
            if out:
                output += out
            if err:
                output += f"\n[stderr]\n{err}" if output else f"[stderr]\n{err}"

            if len(output) > self._max_output:
                output = output[:self._max_output] + "\n...(truncated)"

            return ToolResult(
                success=exit_code == 0,
                output=output or "(no output)",
                error=f"Exit code {exit_code}" if exit_code != 0 else "",
                metadata={"exit_code": exit_code, "language": lang, "executor": executor_name},
            )
        except asyncio.TimeoutError:
            return ToolResult(success=False, error=f"Execution timed out after {timeout}s")
