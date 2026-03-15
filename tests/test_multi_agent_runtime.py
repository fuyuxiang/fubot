from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fubot.agent.loop import AgentLoop
from fubot.bus.queue import MessageBus
from fubot.config.schema import AgentProfile, Config
from fubot.orchestrator.models import ExecutionLogRecord, TaskRecord, WorkflowRecord
from fubot.orchestrator.runtime import CoordinatorRuntime, ExecutorResult
from fubot.orchestrator.store import WorkflowStore
from fubot.providers.base import LLMResponse, ToolCallRequest


def _config() -> Config:
    config = Config()
    config.orchestration.routing.max_parallel_executors = 2
    config.observability.show_executor_progress = True
    return config


@pytest.mark.asyncio
async def test_coordinator_dispatches_executor_and_persists_shared_board(tmp_path: Path) -> None:
    config = _config()
    store = WorkflowStore(tmp_path)
    runtime = CoordinatorRuntime(config, store)
    seen_profiles: list[str] = []

    async def _execute_task(
        profile,
        task,
        workflow_id,
        session_key,
        channel,
        chat_id,
        media,
        shared_board,
        on_progress,
    ) -> ExecutorResult:
        seen_profiles.append(profile.id)
        await on_progress("working", False)
        return ExecutorResult(
            profile=profile,
            task=task,
            content=f"{profile.id}:{task.kind}",
            route={"model": "test-model", "provider": "dashscope"},
        )

    workflow, tasks, assignments, results, final = await runtime.run(
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        content="say hello",
        media=None,
        execute_task=_execute_task,
    )

    assert seen_profiles == ["generalist"]
    assert workflow.status == "completed"
    assert len(tasks) == 1
    assert len(assignments) == 1
    assert len(results) == 1
    assert final == "generalist:communication"

    saved = store.load_workflow(workflow.id)
    assert saved is not None
    workflow_payload = saved["workflow"]
    assert workflow_payload["shared_board"]["generalist"]["content"] == "generalist:communication"
    assert workflow_payload["execution_logs"][0]["agent_name"] == "Generalist"


@pytest.mark.asyncio
async def test_coordinator_runs_multiple_executors_concurrently_for_coding(tmp_path: Path) -> None:
    config = _config()
    store = WorkflowStore(tmp_path)
    runtime = CoordinatorRuntime(config, store)
    active = 0
    max_active = 0

    async def _execute_task(
        profile,
        task,
        workflow_id,
        session_key,
        channel,
        chat_id,
        media,
        shared_board,
        on_progress,
    ) -> ExecutorResult:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await on_progress(f"{profile.id} start", False)
        await asyncio.sleep(0.05)
        active -= 1
        return ExecutorResult(
            profile=profile,
            task=task,
            content=f"{profile.id} done",
            route={"model": "test-model", "provider": "dashscope"},
        )

    workflow, tasks, assignments, results, final = await runtime.run(
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        content="implement a new feature and refactor the module for production use",
        media=None,
        execute_task=_execute_task,
    )

    assert workflow.status == "completed"
    assert len(tasks) == 2
    assert len(assignments) == 2
    assert len(results) == 2
    assert max_active >= 2
    assert "[Builder] builder done" in final
    assert "[Verifier] verifier done" in final


def test_workflow_store_recovers_incomplete_workflows(tmp_path: Path) -> None:
    store = WorkflowStore(tmp_path)
    workflow = WorkflowRecord(
        id="wf_incomplete",
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        user_message="hello",
        status="running",
    )
    task = TaskRecord(id="task1", workflow_id=workflow.id, title="t", kind="communication")
    store.save_workflow(workflow, [task], [])

    recovered = store.recover_incomplete()

    assert len(recovered) == 1
    assert recovered[0]["workflow"]["id"] == "wf_incomplete"


@pytest.mark.asyncio
async def test_agent_identity_metadata_is_emitted_on_progress(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, runtime_config=_config())
    record = ExecutionLogRecord(
        workflow_id="wf1",
        task_id="task1",
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        message_kind="progress",
        content="working",
        is_final=False,
    )

    await loop._emit_workflow_update(record, "cli", "direct")

    outbound = await loop.bus.consume_outbound()
    assert outbound.content == "[Builder] working"
    assert outbound.agent_id == "builder"
    assert outbound.agent_name == "Builder"
    assert outbound.agent_role == "coding"
    assert outbound.workflow_id == "wf1"
    assert outbound.task_id == "task1"
    assert outbound.message_kind == "progress"
    assert outbound.is_final is False


@pytest.mark.asyncio
async def test_executor_tool_allowlist_is_enforced(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="call1", name="message", arguments={"content": "hi"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        runtime_config=_config(),
    )
    task = TaskRecord(
        id="task1",
        workflow_id="wf1",
        title="limited",
        kind="research",
        metadata={"input": "send a message"},
    )
    profile = AgentProfile(
        id="limited",
        name="Limited",
        role="research",
        tool_allowlist=["read_file"],
    )

    result = await loop._execute_profile_task(
        profile=profile,
        task=task,
        workflow_id="wf1",
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        media=None,
        shared_board={},
        on_progress=None,
    )

    tool_messages = [message for message in result.all_messages if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "Tool 'message' not found" in tool_messages[0]["content"]
