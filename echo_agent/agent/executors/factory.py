"""Executor factory for tool execution isolation."""

from __future__ import annotations

from pathlib import Path

from echo_agent.agent.executors.base import BaseExecutor, LocalExecutor, SandboxExecutor
from echo_agent.agent.executors.remote import ContainerExecutor, RemoteExecutor
from echo_agent.config.schema import ExecutionConfig


def create_executor(config: ExecutionConfig, workspace: Path) -> BaseExecutor:
    """Create the configured execution backend.

    The returned executor is intentionally long-lived so sandbox/container setup
    cost is paid once per AgentLoop rather than once per tool call.
    """
    kind = config.default_executor
    if kind == "local":
        return LocalExecutor(str(workspace), network_policy=config.network_policy)
    if kind == "sandbox":
        return SandboxExecutor(config.sandbox_root, network_policy=config.network_policy, workspace=str(workspace))
    if kind == "container":
        return ContainerExecutor(config.container_image, network_policy=config.network_policy, workspace=str(workspace))
    if kind == "remote":
        return RemoteExecutor(
            host=config.remote_host,
            user=config.remote_user,
            key_path=config.remote_key_path,
            strict_host_key=config.remote_strict_host_key,
            connect_timeout=config.remote_connect_timeout,
        )
    raise ValueError(f"Unsupported executor: {kind}")
