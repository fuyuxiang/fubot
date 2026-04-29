"""Comprehensive tests for echo_agent/tasks/ — models, manager, workflow."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from echo_agent.tasks.models import (
    TaskRecord,
    TaskStatus,
    VALID_TASK_TRANSITIONS,
    TERMINAL_TASK_STATUSES,
    WorkflowRecord,
    WorkflowStatus,
    VALID_WORKFLOW_TRANSITIONS,
    TERMINAL_WORKFLOW_STATUSES,
    StepDefinition,
)
from echo_agent.tasks.manager import TaskManager
from echo_agent.tasks.workflow import WorkflowEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage():
    """Dict-backed async storage mock."""
    store: dict[str, Any] = {}
    s = AsyncMock()
    s.store_task = AsyncMock(side_effect=lambda tid, d: store.__setitem__(f"task:{tid}", d))
    s.load_task = AsyncMock(side_effect=lambda tid: store.get(f"task:{tid}"))
    s.list_tasks = AsyncMock(side_effect=lambda **kw: [
        v for k, v in store.items()
        if k.startswith("task:")
        and (not kw.get("status") or v.get("status") == kw["status"])
        and (not kw.get("workflow_id") or v.get("workflow_id") == kw["workflow_id"])
    ])
    s.store_workflow = AsyncMock(side_effect=lambda wid, d: store.__setitem__(f"wf:{wid}", d))
    s.load_workflow = AsyncMock(side_effect=lambda wid: store.get(f"wf:{wid}"))
    s.list_workflows = AsyncMock(side_effect=lambda **kw: [
        v for k, v in store.items()
        if k.startswith("wf:")
        and (not kw.get("status") or v.get("status") == kw["status"])
    ])
    return s

# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — models.py
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskStatus:
    """TaskStatus enum values and transition table."""

    def test_all_members(self):
        names = {s.name for s in TaskStatus}
        assert names == {"PENDING", "QUEUED", "RUNNING", "SUCCESS", "FAILED", "CANCELLED", "SUSPENDED"}

    def test_terminal_statuses(self):
        assert TERMINAL_TASK_STATUSES == {TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED}

    def test_terminal_states_have_no_outgoing(self):
        for ts in TERMINAL_TASK_STATUSES:
            assert VALID_TASK_TRANSITIONS[ts] == set() or ts == TaskStatus.FAILED
        # FAILED can retry -> QUEUED
        assert VALID_TASK_TRANSITIONS[TaskStatus.FAILED] == {TaskStatus.QUEUED}

    def test_every_status_in_transition_table(self):
        for s in TaskStatus:
            assert s in VALID_TASK_TRANSITIONS


class TestTaskRecord:
    """TaskRecord dataclass, to_dict / from_dict roundtrip."""

    def test_defaults(self):
        t = TaskRecord()
        assert t.status == TaskStatus.PENDING
        assert t.priority == 5
        assert t.max_retries == 3
        assert t.id.startswith("t_")

    def test_roundtrip(self):
        t = TaskRecord(title="do stuff", description="desc", priority=1, metadata={"k": "v"})
        d = t.to_dict()
        t2 = TaskRecord.from_dict(d)
        assert t2.title == t.title
        assert t2.priority == t.priority
        assert t2.metadata == {"k": "v"}
        assert t2.status == TaskStatus.PENDING

    def test_from_dict_missing_optional_fields(self):
        t = TaskRecord.from_dict({"id": "t_abc"})
        assert t.id == "t_abc"
        assert t.workflow_id == ""
        assert t.status == TaskStatus.PENDING


class TestWorkflowStatus:
    def test_all_members(self):
        names = {s.name for s in WorkflowStatus}
        assert names == {"PENDING", "RUNNING", "WAITING", "BLOCKED", "SUCCESS", "FAILED", "CANCELLED"}

    def test_terminal_statuses(self):
        assert TERMINAL_WORKFLOW_STATUSES == {WorkflowStatus.SUCCESS, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}

    def test_every_status_in_transition_table(self):
        for s in WorkflowStatus:
            assert s in VALID_WORKFLOW_TRANSITIONS


class TestStepDefinition:
    def test_roundtrip(self):
        sd = StepDefinition(id="s1", name="fetch", tool_name="http_get",
                            tool_params={"url": "https://x"}, depends_on=["s0"],
                            condition="ok", retry_max=2, timeout_seconds=60)
        d = sd.to_dict()
        sd2 = StepDefinition.from_dict(d)
        assert sd2.id == "s1"
        assert sd2.depends_on == ["s0"]
        assert sd2.timeout_seconds == 60

    def test_defaults(self):
        sd = StepDefinition()
        assert sd.retry_max == 0
        assert sd.timeout_seconds == 300


class TestWorkflowRecord:
    def test_roundtrip(self):
        step = StepDefinition(id="s0", name="a", tool_name="t")
        wf = WorkflowRecord(name="wf1", steps=[step], step_tasks={"s0": "t_1"}, state={"x": 1})
        d = wf.to_dict()
        wf2 = WorkflowRecord.from_dict(d)
        assert wf2.name == "wf1"
        assert len(wf2.steps) == 1
        assert wf2.step_tasks == {"s0": "t_1"}
        assert wf2.state == {"x": 1}

    def test_defaults(self):
        wf = WorkflowRecord()
        assert wf.status == WorkflowStatus.PENDING
        assert wf.id.startswith("wf_")


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — manager.py (TaskManager)
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskManagerCreate:
    @pytest.mark.asyncio
    async def test_create_basic(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("my task")
        assert t.title == "my task"
        assert t.status == TaskStatus.PENDING
        assert t.id.startswith("t_")

    @pytest.mark.asyncio
    async def test_create_with_all_params(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create(
            title="t", description="d", workflow_id="wf_1",
            parent_task_id="t_0", priority=1, max_retries=5,
            metadata={"a": 1},
        )
        assert t.workflow_id == "wf_1"
        assert t.parent_task_id == "t_0"
        assert t.priority == 1
        assert t.max_retries == 5
        assert t.metadata == {"a": 1}

    @pytest.mark.asyncio
    async def test_create_persists(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("x")
        loaded = await mgr.get(t.id)
        assert loaded is not None
        assert loaded.title == "x"


class TestTaskManagerGet:
    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        mgr = TaskManager(_make_storage())
        assert await mgr.get("nonexistent") is None


class TestTaskManagerTransition:
    @pytest.mark.asyncio
    async def test_valid_transition_pending_to_running(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        t2 = await mgr.transition(t.id, TaskStatus.RUNNING)
        assert t2.status == TaskStatus.RUNNING
        assert t2.started_at != ""

    @pytest.mark.asyncio
    async def test_running_to_success_sets_completed_at(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        await mgr.transition(t.id, TaskStatus.RUNNING)
        t2 = await mgr.transition(t.id, TaskStatus.SUCCESS, result="done")
        assert t2.status == TaskStatus.SUCCESS
        assert t2.completed_at != ""
        assert t2.result == "done"

    @pytest.mark.asyncio
    async def test_running_to_failed_with_error(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        await mgr.transition(t.id, TaskStatus.RUNNING)
        t2 = await mgr.transition(t.id, TaskStatus.FAILED, error="boom")
        assert t2.status == TaskStatus.FAILED
        assert t2.error == "boom"
        assert t2.completed_at != ""

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        with pytest.raises(ValueError, match="Invalid transition"):
            await mgr.transition(t.id, TaskStatus.SUCCESS)

    @pytest.mark.asyncio
    async def test_transition_nonexistent_raises(self):
        mgr = TaskManager(_make_storage())
        with pytest.raises(ValueError, match="not found"):
            await mgr.transition("nope", TaskStatus.RUNNING)

    @pytest.mark.asyncio
    async def test_terminal_to_anything_raises(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        await mgr.transition(t.id, TaskStatus.RUNNING)
        await mgr.transition(t.id, TaskStatus.SUCCESS)
        with pytest.raises(ValueError, match="Invalid transition"):
            await mgr.transition(t.id, TaskStatus.RUNNING)

    @pytest.mark.asyncio
    async def test_started_at_set_only_once(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        t2 = await mgr.transition(t.id, TaskStatus.RUNNING)
        first_started = t2.started_at
        await mgr.transition(t.id, TaskStatus.SUSPENDED)
        t3 = await mgr.transition(t.id, TaskStatus.RUNNING)
        assert t3.started_at == first_started


class TestTaskManagerRetry:
    @pytest.mark.asyncio
    async def test_retry_from_failed(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t", max_retries=3)
        await mgr.transition(t.id, TaskStatus.RUNNING)
        await mgr.transition(t.id, TaskStatus.FAILED, error="err")
        t2 = await mgr.retry(t.id)
        assert t2.status == TaskStatus.QUEUED
        assert t2.retry_count == 1
        assert t2.error == ""
        assert t2.completed_at == ""

    @pytest.mark.asyncio
    async def test_retry_not_failed_raises(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        with pytest.raises(ValueError, match="only retry failed"):
            await mgr.retry(t.id)

    @pytest.mark.asyncio
    async def test_retry_exceeds_max_raises(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t", max_retries=1)
        await mgr.transition(t.id, TaskStatus.RUNNING)
        await mgr.transition(t.id, TaskStatus.FAILED)
        await mgr.retry(t.id)  # retry_count=1
        # Now run and fail again
        await mgr.transition(t.id, TaskStatus.RUNNING)
        await mgr.transition(t.id, TaskStatus.FAILED)
        with pytest.raises(ValueError, match="Max retries"):
            await mgr.retry(t.id)

    @pytest.mark.asyncio
    async def test_retry_nonexistent_raises(self):
        mgr = TaskManager(_make_storage())
        with pytest.raises(ValueError, match="not found"):
            await mgr.retry("nope")



class TestTaskManagerCancel:
    @pytest.mark.asyncio
    async def test_cancel_pending(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        t2 = await mgr.cancel(t.id)
        assert t2.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_running(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        await mgr.transition(t.id, TaskStatus.RUNNING)
        t2 = await mgr.cancel(t.id)
        assert t2.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_already_terminal_raises(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        await mgr.transition(t.id, TaskStatus.RUNNING)
        await mgr.transition(t.id, TaskStatus.SUCCESS)
        with pytest.raises(ValueError, match="Invalid transition"):
            await mgr.cancel(t.id)


class TestTaskManagerUpdate:
    @pytest.mark.asyncio
    async def test_update_fields(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("old title")
        t2 = await mgr.update(t.id, title="new title", priority=1, description="d", metadata={"x": 1})
        assert t2.title == "new title"
        assert t2.priority == 1
        assert t2.description == "d"
        assert t2.metadata == {"x": 1}

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises(self):
        mgr = TaskManager(_make_storage())
        with pytest.raises(ValueError, match="not found"):
            await mgr.update("nope", title="x")

    @pytest.mark.asyncio
    async def test_update_ignores_unknown_fields(self):
        mgr = TaskManager(_make_storage())
        t = await mgr.create("t")
        t2 = await mgr.update(t.id, status="running")  # status not in allowed keys
        assert t2.status == TaskStatus.PENDING  # unchanged


class TestTaskManagerList:
    @pytest.mark.asyncio
    async def test_list_by_status(self):
        mgr = TaskManager(_make_storage())
        await mgr.create("a")
        t2 = await mgr.create("b")
        await mgr.transition(t2.id, TaskStatus.RUNNING)
        running = await mgr.list_by_status(TaskStatus.RUNNING)
        assert len(running) == 1
        assert running[0].title == "b"

    @pytest.mark.asyncio
    async def test_list_by_workflow(self):
        mgr = TaskManager(_make_storage())
        await mgr.create("a", workflow_id="wf_1")
        await mgr.create("b", workflow_id="wf_2")
        await mgr.create("c", workflow_id="wf_1")
        wf1_tasks = await mgr.list_by_workflow("wf_1")
        assert len(wf1_tasks) == 2



# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — workflow.py (WorkflowEngine)
# ═══════════════════════════════════════════════════════════════════════════

def _make_engine():
    """Return (engine, task_manager, storage) wired together."""
    storage = _make_storage()
    tm = TaskManager(storage)
    engine = WorkflowEngine(storage, tm)
    return engine, tm, storage


def _simple_steps():
    """Two sequential steps: step_0 -> step_1."""
    return [
        {"name": "fetch", "tool_name": "http_get", "tool_params": {"url": "https://x"}},
        {"name": "parse", "tool_name": "json_parse", "depends_on": ["step_0"]},
    ]


def _parallel_steps():
    """Two independent steps (no deps)."""
    return [
        {"name": "a", "tool_name": "tool_a"},
        {"name": "b", "tool_name": "tool_b"},
    ]


class TestWorkflowEngineCreate:
    @pytest.mark.asyncio
    async def test_create_auto_ids(self):
        engine, _, _ = _make_engine()
        wf = await engine.create("wf", _simple_steps())
        assert wf.name == "wf"
        assert wf.status == WorkflowStatus.PENDING
        assert len(wf.steps) == 2
        assert wf.steps[0].id == "step_0"
        assert wf.steps[1].id == "step_1"

    @pytest.mark.asyncio
    async def test_create_preserves_explicit_id(self):
        engine, _, _ = _make_engine()
        wf = await engine.create("wf", [{"id": "my_step", "tool_name": "t"}])
        assert wf.steps[0].id == "my_step"

    @pytest.mark.asyncio
    async def test_create_auto_name_from_tool(self):
        engine, _, _ = _make_engine()
        wf = await engine.create("wf", [{"tool_name": "do_thing"}])
        assert wf.steps[0].name == "do_thing"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        engine, _, _ = _make_engine()
        assert await engine.get("nonexistent") is None


class TestWorkflowEngineStart:
    @pytest.mark.asyncio
    async def test_start_transitions_to_running(self):
        engine, _, _ = _make_engine()
        wf = await engine.create("wf", _simple_steps())
        wf2 = await engine.start(wf.id)
        assert wf2.status == WorkflowStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_queues_first_step(self):
        engine, tm, _ = _make_engine()
        wf = await engine.create("wf", _simple_steps())
        wf2 = await engine.start(wf.id)
        # step_0 has no deps, should be queued; step_1 depends on step_0
        assert "step_0" in wf2.step_tasks
        assert "step_1" not in wf2.step_tasks

    @pytest.mark.asyncio
    async def test_start_queues_parallel_steps(self):
        engine, _, _ = _make_engine()
        wf = await engine.create("wf", _parallel_steps())
        wf2 = await engine.start(wf.id)
        assert "step_0" in wf2.step_tasks
        assert "step_1" in wf2.step_tasks

    @pytest.mark.asyncio
    async def test_start_nonexistent_raises(self):
        engine, _, _ = _make_engine()
        with pytest.raises(ValueError, match="not found"):
            await engine.start("nope")

    @pytest.mark.asyncio
    async def test_start_already_running_raises(self):
        engine, _, _ = _make_engine()
        wf = await engine.create("wf", _simple_steps())
        await engine.start(wf.id)
        with pytest.raises(ValueError, match="Invalid workflow transition"):
            await engine.start(wf.id)



class TestWorkflowEngineAdvance:
    @pytest.mark.asyncio
    async def test_advance_queues_next_step_after_success(self):
        engine, tm, _ = _make_engine()
        wf = await engine.create("wf", _simple_steps())
        wf = await engine.start(wf.id)
        # Complete step_0's task
        tid0 = wf.step_tasks["step_0"]
        await tm.transition(tid0, TaskStatus.RUNNING)
        await tm.transition(tid0, TaskStatus.SUCCESS)
        wf = await engine.advance(wf.id)
        assert "step_1" in wf.step_tasks
        assert wf.status == WorkflowStatus.RUNNING

    @pytest.mark.asyncio
    async def test_advance_all_done_marks_success(self):
        engine, tm, _ = _make_engine()
        wf = await engine.create("wf", _parallel_steps())
        wf = await engine.start(wf.id)
        for step_id in ("step_0", "step_1"):
            tid = wf.step_tasks[step_id]
            await tm.transition(tid, TaskStatus.RUNNING)
            await tm.transition(tid, TaskStatus.SUCCESS)
        wf = await engine.advance(wf.id)
        assert wf.status == WorkflowStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_advance_any_failed_marks_failed(self):
        engine, tm, _ = _make_engine()
        wf = await engine.create("wf", _parallel_steps())
        wf = await engine.start(wf.id)
        tid0 = wf.step_tasks["step_0"]
        await tm.transition(tid0, TaskStatus.RUNNING)
        await tm.transition(tid0, TaskStatus.FAILED)
        wf = await engine.advance(wf.id)
        assert wf.status == WorkflowStatus.FAILED

    @pytest.mark.asyncio
    async def test_advance_terminal_workflow_is_noop(self):
        engine, tm, _ = _make_engine()
        wf = await engine.create("wf", _parallel_steps())
        wf = await engine.start(wf.id)
        for step_id in ("step_0", "step_1"):
            tid = wf.step_tasks[step_id]
            await tm.transition(tid, TaskStatus.RUNNING)
            await tm.transition(tid, TaskStatus.SUCCESS)
        wf = await engine.advance(wf.id)
        assert wf.status == WorkflowStatus.SUCCESS
        wf2 = await engine.advance(wf.id)
        assert wf2.status == WorkflowStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_advance_nonexistent_raises(self):
        engine, _, _ = _make_engine()
        with pytest.raises(ValueError, match="not found"):
            await engine.advance("nope")


class TestWorkflowEngineOnTaskComplete:
    @pytest.mark.asyncio
    async def test_on_task_complete_triggers_advance(self):
        engine, tm, _ = _make_engine()
        wf = await engine.create("wf", _parallel_steps())
        wf = await engine.start(wf.id)
        tid0 = wf.step_tasks["step_0"]
        await tm.transition(tid0, TaskStatus.RUNNING)
        await tm.transition(tid0, TaskStatus.SUCCESS)
        tid1 = wf.step_tasks["step_1"]
        await tm.transition(tid1, TaskStatus.RUNNING)
        await tm.transition(tid1, TaskStatus.SUCCESS)
        await engine.on_task_complete(tid1)
        wf = await engine.get(wf.id)
        assert wf.status == WorkflowStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_on_task_complete_no_workflow_is_noop(self):
        engine, tm, _ = _make_engine()
        t = await tm.create("standalone")
        await engine.on_task_complete(t.id)  # should not raise

    @pytest.mark.asyncio
    async def test_on_task_complete_nonexistent_task_is_noop(self):
        engine, _, _ = _make_engine()
        await engine.on_task_complete("nonexistent")  # should not raise



class TestWorkflowEnginePauseResume:
    @pytest.mark.asyncio
    async def test_pause_running_workflow(self):
        engine, _, _ = _make_engine()
        wf = await engine.create("wf", _simple_steps())
        await engine.start(wf.id)
        wf = await engine.pause(wf.id)
        assert wf.status == WorkflowStatus.WAITING

    @pytest.mark.asyncio
    async def test_resume_paused_workflow(self):
        engine, _, _ = _make_engine()
        wf = await engine.create("wf", _simple_steps())
        await engine.start(wf.id)
        await engine.pause(wf.id)
        wf = await engine.resume(wf.id)
        assert wf.status == WorkflowStatus.RUNNING

    @pytest.mark.asyncio
    async def test_pause_nonexistent_raises(self):
        engine, _, _ = _make_engine()
        with pytest.raises(ValueError, match="not found"):
            await engine.pause("nope")

    @pytest.mark.asyncio
    async def test_resume_nonexistent_raises(self):
        engine, _, _ = _make_engine()
        with pytest.raises(ValueError, match="not found"):
            await engine.resume("nope")

    @pytest.mark.asyncio
    async def test_pause_pending_raises(self):
        engine, _, _ = _make_engine()
        wf = await engine.create("wf", _simple_steps())
        with pytest.raises(ValueError, match="Invalid workflow transition"):
            await engine.pause(wf.id)


class TestWorkflowEngineCancel:
    @pytest.mark.asyncio
    async def test_cancel_running_workflow(self):
        engine, tm, _ = _make_engine()
        wf = await engine.create("wf", _parallel_steps())
        wf = await engine.start(wf.id)
        wf = await engine.cancel(wf.id)
        assert wf.status == WorkflowStatus.CANCELLED
        # Non-terminal tasks should be cancelled
        for tid in wf.step_tasks.values():
            t = await tm.get(tid)
            assert t.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_skips_already_terminal_tasks(self):
        engine, tm, _ = _make_engine()
        wf = await engine.create("wf", _parallel_steps())
        wf = await engine.start(wf.id)
        # Complete one task first
        tid0 = wf.step_tasks["step_0"]
        await tm.transition(tid0, TaskStatus.RUNNING)
        await tm.transition(tid0, TaskStatus.SUCCESS)
        wf = await engine.cancel(wf.id)
        t0 = await tm.get(tid0)
        assert t0.status == TaskStatus.SUCCESS  # unchanged

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_raises(self):
        engine, _, _ = _make_engine()
        with pytest.raises(ValueError, match="not found"):
            await engine.cancel("nope")

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_raises(self):
        engine, _, _ = _make_engine()
        wf = await engine.create("wf", _simple_steps())
        await engine.start(wf.id)
        await engine.cancel(wf.id)
        with pytest.raises(ValueError, match="Invalid workflow transition"):
            await engine.cancel(wf.id)


class TestWorkflowEngineListAll:
    @pytest.mark.asyncio
    async def test_list_all_no_filter(self):
        engine, _, _ = _make_engine()
        await engine.create("a", [{"tool_name": "t"}])
        await engine.create("b", [{"tool_name": "t"}])
        result = await engine.list_all()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_all_with_status_filter(self):
        engine, _, _ = _make_engine()
        wf1 = await engine.create("a", [{"tool_name": "t"}])
        await engine.create("b", [{"tool_name": "t"}])
        await engine.start(wf1.id)
        running = await engine.list_all(status="running")
        assert len(running) == 1
        assert running[0].name == "a"


class TestWorkflowDAGExecution:
    """End-to-end DAG: step_0 -> step_1 -> step_2."""

    @pytest.mark.asyncio
    async def test_three_step_chain(self):
        engine, tm, _ = _make_engine()
        steps = [
            {"name": "s0", "tool_name": "t0"},
            {"name": "s1", "tool_name": "t1", "depends_on": ["step_0"]},
            {"name": "s2", "tool_name": "t2", "depends_on": ["step_1"]},
        ]
        wf = await engine.create("chain", steps)
        wf = await engine.start(wf.id)
        assert "step_0" in wf.step_tasks
        assert "step_1" not in wf.step_tasks

        # Complete step_0
        tid0 = wf.step_tasks["step_0"]
        await tm.transition(tid0, TaskStatus.RUNNING)
        await tm.transition(tid0, TaskStatus.SUCCESS)
        wf = await engine.advance(wf.id)
        assert "step_1" in wf.step_tasks
        assert "step_2" not in wf.step_tasks

        # Complete step_1
        tid1 = wf.step_tasks["step_1"]
        await tm.transition(tid1, TaskStatus.RUNNING)
        await tm.transition(tid1, TaskStatus.SUCCESS)
        wf = await engine.advance(wf.id)
        assert "step_2" in wf.step_tasks

        # Complete step_2
        tid2 = wf.step_tasks["step_2"]
        await tm.transition(tid2, TaskStatus.RUNNING)
        await tm.transition(tid2, TaskStatus.SUCCESS)
        wf = await engine.advance(wf.id)
        assert wf.status == WorkflowStatus.SUCCESS