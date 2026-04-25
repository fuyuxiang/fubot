"""Tests for RemoteExecutor security hardening."""

from __future__ import annotations

import pytest

from echo_agent.agent.executors.base import ExecRequest
from echo_agent.agent.executors.remote import RemoteExecutor


def _make_executor(**kwargs) -> RemoteExecutor:
    defaults = {"host": "10.0.0.1", "user": "deploy", "key_path": "/keys/id_rsa"}
    defaults.update(kwargs)
    return RemoteExecutor(**defaults)


class TestSSHBaseCommand:
    def test_default_strict_host_key(self) -> None:
        ex = _make_executor()
        cmd = ex._build_ssh_base()
        assert "-o" in cmd
        idx = cmd.index("StrictHostKeyChecking=accept-new")
        assert cmd[idx - 1] == "-o"

    def test_custom_strict_host_key(self) -> None:
        ex = _make_executor(strict_host_key="yes")
        cmd = ex._build_ssh_base()
        assert "StrictHostKeyChecking=yes" in cmd

    def test_connect_timeout(self) -> None:
        ex = _make_executor(connect_timeout=30)
        cmd = ex._build_ssh_base()
        assert "ConnectTimeout=30" in cmd

    def test_key_path_included(self) -> None:
        ex = _make_executor(key_path="/my/key")
        cmd = ex._build_ssh_base()
        assert "-i" in cmd
        assert "/my/key" in cmd

    def test_user_host_format(self) -> None:
        ex = _make_executor(user="admin", host="example.com")
        cmd = ex._build_ssh_base()
        assert "admin@example.com" in cmd


class TestCommandSanitization:
    def test_cwd_is_quoted(self) -> None:
        ex = _make_executor()
        req = ExecRequest(command="ls", cwd="/path with spaces/dir")
        remote_cmd = ex._build_remote_command(req)
        assert "'/path with spaces/dir'" in remote_cmd

    def test_env_values_are_quoted(self) -> None:
        ex = _make_executor()
        req = ExecRequest(command="echo ok", env={"FOO": "bar; rm -rf /"})
        remote_cmd = ex._build_remote_command(req)
        assert "'bar; rm -rf /'" in remote_cmd

    def test_credentials_are_injected(self) -> None:
        ex = _make_executor()
        req = ExecRequest(command="deploy", credentials={"API_KEY": "secret123"})
        remote_cmd = ex._build_remote_command(req)
        assert "API_KEY=" in remote_cmd
        assert "secret123" in remote_cmd
        assert "deploy" in remote_cmd

    def test_command_with_special_chars(self) -> None:
        ex = _make_executor()
        req = ExecRequest(command="echo $(whoami) && cat /etc/passwd")
        remote_cmd = ex._build_remote_command(req)
        assert "echo $(whoami) && cat /etc/passwd" in remote_cmd

    def test_no_cwd_no_env(self) -> None:
        ex = _make_executor()
        req = ExecRequest(command="uptime")
        remote_cmd = ex._build_remote_command(req)
        assert remote_cmd == "uptime"
