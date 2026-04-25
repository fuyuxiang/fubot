from __future__ import annotations

import pytest

from echo_agent.agent.executors.base import BaseExecutor, ExecRequest, ExecResponse
from echo_agent.agent.tools.code_exec import CodeExecTool
from echo_agent.agent.tools.shell import ShellTool


class RecordingExecutor(BaseExecutor):
    name = "recording"

    def __init__(self):
        self.requests: list[ExecRequest] = []

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def execute(self, request: ExecRequest) -> ExecResponse:
        self.requests.append(request)
        return ExecResponse(success=True, stdout=request.stdin or request.command, executor=self.name)


@pytest.mark.asyncio
async def test_shell_tool_uses_configured_executor(tmp_path) -> None:
    executor = RecordingExecutor()
    tool = ShellTool(str(tmp_path), executor=executor)

    result = await tool.execute({"command": "echo hello", "timeout": 3})

    assert result.success
    assert result.metadata["executor"] == "recording"
    assert executor.requests[0].command == "echo hello"


@pytest.mark.asyncio
async def test_code_exec_uses_stdin_and_language_allowlist(tmp_path) -> None:
    executor = RecordingExecutor()
    tool = CodeExecTool(str(tmp_path), executor=executor, allowed_languages=["python"])

    denied = await tool.execute({"language": "bash", "code": "echo no"})
    allowed = await tool.execute({"language": "python", "code": "print('ok')"})

    assert not denied.success
    assert "Language not allowed" in denied.error
    assert allowed.success
    assert executor.requests[0].command == "python3 -"
    assert executor.requests[0].stdin == "print('ok')"
