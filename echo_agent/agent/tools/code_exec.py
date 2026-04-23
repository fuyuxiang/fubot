"""Code execution tool — run Python/JS snippets in a subprocess."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolPermission, ToolResult


_RUNNERS: dict[str, tuple[str, str]] = {
    "python": ("python3", ".py"),
    "javascript": ("node", ".js"),
    "bash": ("bash", ".sh"),
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

    def __init__(self, workspace: str):
        self._workspace = Path(workspace)

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        code = params["code"]
        lang = params["language"]
        timeout = min(params.get("timeout", 30), 60)
        for pattern in self._SENSITIVE_PATTERNS:
            if pattern.search(code):
                return ToolResult(success=False, error="Code blocked by sensitive system file policy")

        runner_info = _RUNNERS.get(lang)
        if not runner_info:
            return ToolResult(success=False, error=f"Unsupported language: {lang}")

        binary, ext = runner_info

        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, dir=self._workspace, delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                binary, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._workspace),
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(success=False, error=f"Execution timed out after {timeout}s")

            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            exit_code = proc.returncode

            output = ""
            if out:
                output += out
            if err:
                output += f"\n[stderr]\n{err}" if output else f"[stderr]\n{err}"

            if len(output) > 32000:
                output = output[:32000] + "\n...(truncated)"

            return ToolResult(
                success=exit_code == 0,
                output=output or "(no output)",
                error=f"Exit code {exit_code}" if exit_code != 0 else "",
                metadata={"exit_code": exit_code, "language": lang},
            )
        finally:
            Path(script_path).unlink(missing_ok=True)
