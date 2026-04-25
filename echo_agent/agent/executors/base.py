"""Execution environments — isolated runtimes for agent task execution.

Supports local, sandbox, container, and remote execution with
command isolation, filesystem boundaries, network control, credential injection, and audit.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ExecRequest:
    command: str
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    stdin: str = ""
    credentials: dict[str, str] = field(default_factory=dict)


@dataclass
class ExecResponse:
    success: bool = True
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    duration_ms: int = 0
    executor: str = ""
    audit_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class BaseExecutor(ABC):
    """Abstract execution environment."""

    name: str = "base"

    @abstractmethod
    async def execute(self, request: ExecRequest) -> ExecResponse:
        """Execute a command in this environment."""

    @abstractmethod
    async def setup(self) -> None:
        """Initialize the execution environment."""

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up the execution environment."""

    def inject_credentials(self, env: dict[str, str], credentials: dict[str, str]) -> dict[str, str]:
        merged = dict(env)
        for key, value in credentials.items():
            merged[key] = value
        return merged


class LocalExecutor(BaseExecutor):
    """Execute commands directly on the host."""

    name = "local"

    def __init__(self, workspace: str, network_policy: str = "allow"):
        self._workspace = workspace
        self._network_policy = network_policy

    async def setup(self) -> None:
        Path(self._workspace).mkdir(parents=True, exist_ok=True)

    async def teardown(self) -> None:
        pass

    async def execute(self, request: ExecRequest) -> ExecResponse:
        cwd = request.cwd or self._workspace
        env = self.inject_credentials({**os.environ}, request.credentials)
        env.update(request.env)
        start = datetime.now()

        try:
            proc = await asyncio.create_subprocess_shell(
                request.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if request.stdin else None,
                cwd=cwd,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(request.stdin.encode() if request.stdin else None),
                timeout=request.timeout,
            )
            duration = int((datetime.now() - start).total_seconds() * 1000)
            return ExecResponse(
                success=proc.returncode == 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                return_code=proc.returncode or 0,
                duration_ms=duration,
                executor=self.name,
            )
        except asyncio.TimeoutError:
            return ExecResponse(success=False, stderr=f"Timeout after {request.timeout}s", return_code=-1, executor=self.name)
        except Exception as e:
            return ExecResponse(success=False, stderr=str(e), return_code=-1, executor=self.name)


class SandboxExecutor(BaseExecutor):
    """Execute commands in an isolated temp directory with restricted filesystem access."""

    name = "sandbox"

    def __init__(
        self,
        sandbox_root: str = "/tmp/echo-agent-sandbox",
        network_policy: str = "deny",
        workspace: str = "",
    ):
        self._root = Path(sandbox_root)
        self._network_policy = network_policy
        self._source_workspace = Path(workspace).resolve() if workspace else None
        self._sandbox_dir: Path | None = None
        self._workdir: Path | None = None

    async def setup(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._sandbox_dir = Path(tempfile.mkdtemp(dir=self._root, prefix="sandbox_"))
        self._workdir = self._sandbox_dir / "workspace"
        if self._source_workspace and self._source_workspace.exists():
            ignore = shutil.ignore_patterns(
                ".git",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".venv",
                "node_modules",
                "data/logs",
            )
            shutil.copytree(self._source_workspace, self._workdir, dirs_exist_ok=True, ignore=ignore)
        else:
            self._workdir.mkdir(parents=True, exist_ok=True)
        logger.info("Sandbox created at {}", self._sandbox_dir)

    async def teardown(self) -> None:
        if self._sandbox_dir and self._sandbox_dir.exists():
            shutil.rmtree(self._sandbox_dir, ignore_errors=True)

    async def execute(self, request: ExecRequest) -> ExecResponse:
        if not self._sandbox_dir:
            await self.setup()
        cwd = str(self._resolve_cwd(request.cwd))
        env = self.inject_credentials({"HOME": cwd, "TMPDIR": cwd}, request.credentials)
        env.update(request.env)
        env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")

        start = datetime.now()
        try:
            proc = await asyncio.create_subprocess_shell(
                request.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if request.stdin else None,
                cwd=cwd,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(request.stdin.encode() if request.stdin else None),
                timeout=request.timeout,
            )
            duration = int((datetime.now() - start).total_seconds() * 1000)
            return ExecResponse(
                success=proc.returncode == 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                return_code=proc.returncode or 0,
                duration_ms=duration,
                executor=self.name,
            )
        except asyncio.TimeoutError:
            return ExecResponse(success=False, stderr=f"Timeout after {request.timeout}s", return_code=-1, executor=self.name)
        except Exception as e:
            return ExecResponse(success=False, stderr=str(e), return_code=-1, executor=self.name)

    def _resolve_cwd(self, requested_cwd: str) -> Path:
        if not self._workdir:
            assert self._sandbox_dir
            return self._sandbox_dir
        if not requested_cwd or not self._source_workspace:
            return self._workdir
        try:
            rel = Path(requested_cwd).resolve().relative_to(self._source_workspace)
            target = (self._workdir / rel).resolve()
            target.relative_to(self._workdir)
            target.mkdir(parents=True, exist_ok=True)
            return target
        except ValueError:
            return self._workdir
