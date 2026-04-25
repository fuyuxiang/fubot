"""Container and remote executors."""

from __future__ import annotations

import asyncio
import shlex
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from echo_agent.agent.executors.base import BaseExecutor, ExecRequest, ExecResponse


class ContainerExecutor(BaseExecutor):
    """Execute commands inside a Docker container."""

    name = "container"

    def __init__(self, image: str = "", network_policy: str = "restricted", workspace: str = ""):
        self._image = image
        self._network_policy = network_policy
        self._workspace = Path(workspace).resolve() if workspace else None
        self._container_id: str | None = None

    async def setup(self) -> None:
        if not self._image:
            raise ValueError("Container image not configured")
        try:
            mount_args: list[str] = []
            if self._workspace:
                mount_args = ["-v", f"{self._workspace}:/workspace", "-w", "/workspace"]
            proc = await asyncio.create_subprocess_exec(
                "docker", "create", "--rm",
                "--network", "none" if self._network_policy == "deny" else "bridge",
                "--name", f"echo-agent-{uuid.uuid4().hex[:8]}",
                *mount_args,
                self._image, "sleep", "infinity",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"docker create failed: {stderr.decode()}")
            self._container_id = stdout.decode().strip()
            await asyncio.create_subprocess_exec("docker", "start", self._container_id)
            logger.info("Container {} started from {}", self._container_id[:12], self._image)
        except FileNotFoundError:
            raise RuntimeError("Docker not found — install Docker to use container execution")

    async def teardown(self) -> None:
        if self._container_id:
            try:
                await asyncio.create_subprocess_exec("docker", "rm", "-f", self._container_id)
            except Exception as e:
                logger.warning("Failed to remove container: {}", e)

    async def execute(self, request: ExecRequest) -> ExecResponse:
        if not self._container_id:
            await self.setup()

        env_args = []
        merged_env = self.inject_credentials({}, request.credentials)
        merged_env.update(request.env)
        for k, v in merged_env.items():
            env_args.extend(["-e", f"{k}={v}"])

        cmd = ["docker", "exec"]
        if request.stdin:
            cmd.append("-i")
        cmd += env_args
        cwd = self._container_cwd(request.cwd)
        if cwd:
            cmd.extend(["-w", cwd])
        cmd.extend([self._container_id, "sh", "-c", request.command])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if request.stdin else None,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(request.stdin.encode() if request.stdin else None),
                timeout=request.timeout,
            )
            return ExecResponse(
                success=proc.returncode == 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                return_code=proc.returncode or 0,
                executor=self.name,
            )
        except asyncio.TimeoutError:
            return ExecResponse(success=False, stderr=f"Timeout after {request.timeout}s", return_code=-1, executor=self.name)
        except Exception as e:
            return ExecResponse(success=False, stderr=str(e), return_code=-1, executor=self.name)

    def _container_cwd(self, cwd: str) -> str:
        if not self._workspace:
            return cwd
        if not cwd:
            return "/workspace"
        try:
            rel = Path(cwd).resolve().relative_to(self._workspace)
            return str(Path("/workspace") / rel)
        except ValueError:
            return "/workspace"


class RemoteExecutor(BaseExecutor):
    """Execute commands on a remote host via SSH with proper input sanitization."""

    name = "remote"

    def __init__(
        self,
        host: str = "",
        user: str = "root",
        key_path: str = "",
        strict_host_key: str = "accept-new",
        connect_timeout: int = 10,
    ):
        self._host = host
        self._user = user
        self._key_path = key_path
        self._strict_host_key = strict_host_key
        self._connect_timeout = connect_timeout

    async def setup(self) -> None:
        if not self._host:
            raise ValueError("Remote host not configured")

    async def teardown(self) -> None:
        pass

    def _build_ssh_base(self) -> list[str]:
        cmd = [
            "ssh",
            "-o", f"StrictHostKeyChecking={self._strict_host_key}",
            "-o", f"ConnectTimeout={self._connect_timeout}",
        ]
        if self._key_path:
            cmd.extend(["-i", self._key_path])
        cmd.append(f"{self._user}@{self._host}")
        return cmd

    def _build_remote_command(self, request: ExecRequest) -> str:
        merged_env = self.inject_credentials({}, request.credentials)
        merged_env.update(request.env)

        env_prefix = " ".join(
            f"{shlex.quote(k)}={shlex.quote(v)}" for k, v in merged_env.items()
        )
        safe_cmd = request.command
        if env_prefix:
            safe_cmd = f"{env_prefix} {safe_cmd}"
        if request.cwd:
            safe_cmd = f"cd {shlex.quote(request.cwd)} && {safe_cmd}"
        return safe_cmd

    async def execute(self, request: ExecRequest) -> ExecResponse:
        ssh_cmd = self._build_ssh_base()
        ssh_cmd.append(self._build_remote_command(request))

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if request.stdin else None,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(request.stdin.encode() if request.stdin else None),
                timeout=request.timeout,
            )
            return ExecResponse(
                success=proc.returncode == 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                return_code=proc.returncode or 0,
                executor=self.name,
            )
        except asyncio.TimeoutError:
            return ExecResponse(success=False, stderr=f"Timeout after {request.timeout}s", return_code=-1, executor=self.name)
        except Exception as e:
            return ExecResponse(success=False, stderr=str(e), return_code=-1, executor=self.name)
