"""Comprehensive tests for echo_agent/agent/tools: base, registry, todo, patch."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from echo_agent.agent.tools.base import (
    Tool,
    ToolExecutionContext,
    ToolResult,
    _validate_json_schema,
    build_idempotency_key,
)
from echo_agent.agent.tools.registry import ToolRegistry
from echo_agent.agent.tools.todo import TodoTool
from echo_agent.agent.tools.patch import PatchTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_tool(**overrides) -> Tool:
    """Factory for a concrete Tool subclass with sensible defaults."""

    class DummyTool(Tool):
        name = overrides.get("name", "dummy")
        description = overrides.get("description", "A dummy tool")
        parameters = overrides.get("parameters", {
            "type": "object",
            "properties": {
                "msg": {"type": "string"},
                "count": {"type": "integer"},
                "flag": {"type": "boolean"},
                "mode": {"type": "string", "enum": ["fast", "slow"]},
            },
            "required": ["msg"],
        })
        timeout_seconds = overrides.get("timeout_seconds", 30)
        max_retries = overrides.get("max_retries", 0)

        async def execute(self, params, ctx=None):
            cb = overrides.get("execute_cb")
            if cb:
                return await cb(params, ctx)
            return ToolResult(output="ok")

    return DummyTool()


# ---------------------------------------------------------------------------
# TestToolBase
# ---------------------------------------------------------------------------


class TestToolBase:
    """Tests for Tool, ToolResult, and _validate_json_schema."""

    # -- validate_params -----------------------------------------------------

    def test_validate_params_missing_required(self):
        tool = _make_dummy_tool()
        errors = tool.validate_params({})
        assert any("msg" in e for e in errors)

    def test_validate_params_wrong_type_string(self):
        tool = _make_dummy_tool()
        errors = tool.validate_params({"msg": 123})
        assert any("string" in e for e in errors)

    def test_validate_params_wrong_type_integer(self):
        tool = _make_dummy_tool()
        errors = tool.validate_params({"msg": "hi", "count": "not_int"})
        assert any("integer" in e for e in errors)

    def test_validate_params_wrong_type_boolean(self):
        tool = _make_dummy_tool()
        errors = tool.validate_params({"msg": "hi", "flag": "yes"})
        assert any("boolean" in e for e in errors)

    def test_validate_params_enum_violation(self):
        tool = _make_dummy_tool()
        errors = tool.validate_params({"msg": "hi", "mode": "turbo"})
        assert any("enum" in e or "one of" in e for e in errors)

    def test_validate_params_valid(self):
        tool = _make_dummy_tool()
        errors = tool.validate_params({"msg": "hello", "count": 5, "flag": True, "mode": "fast"})
        assert errors == []

    # -- to_schema -----------------------------------------------------------

    def test_to_schema_structure(self):
        tool = _make_dummy_tool()
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "dummy"
        assert "parameters" in schema["function"]

    def test_to_schema_array_missing_items_raises(self):
        tool = _make_dummy_tool(parameters={
            "type": "object",
            "properties": {
                "tags": {"type": "array"},  # missing items
            },
        })
        with pytest.raises(ValueError, match="array schema missing items"):
            tool.to_schema()

    # -- execution_mode ------------------------------------------------------

    def test_execution_mode_returns_side_effect(self):
        tool = _make_dummy_tool()
        assert tool.execution_mode({}) == "side_effect"

    # -- ToolResult.text -----------------------------------------------------

    def test_tool_result_text_success(self):
        r = ToolResult(success=True, output="done")
        assert r.text == "done"

    def test_tool_result_text_error(self):
        r = ToolResult(success=False, error="boom")
        assert r.text == "Error: boom"


# ---------------------------------------------------------------------------
# TestToolExecutionContext
# ---------------------------------------------------------------------------


class TestToolExecutionContext:

    def test_frozen_immutable(self):
        ctx = ToolExecutionContext(execution_id="abc")
        with pytest.raises(FrozenInstanceError):
            ctx.execution_id = "xyz"

    def test_log_fields_truncates_idempotency_key(self):
        long_key = "a" * 64
        ctx = ToolExecutionContext(idempotency_key=long_key)
        fields = ctx.log_fields()
        assert len(fields["idempotency_key"]) == 16

    def test_log_fields_empty_idempotency_key(self):
        ctx = ToolExecutionContext()
        fields = ctx.log_fields()
        assert fields["idempotency_key"] == ""

    def test_build_idempotency_key_deterministic(self):
        k1 = build_idempotency_key("t1", "tool", 0, {"a": 1})
        k2 = build_idempotency_key("t1", "tool", 0, {"a": 1})
        assert k1 == k2
        assert len(k1) == 24

    def test_build_idempotency_key_differs_on_input(self):
        k1 = build_idempotency_key("t1", "tool", 0, {"a": 1})
        k2 = build_idempotency_key("t1", "tool", 0, {"a": 2})
        assert k1 != k2


# ---------------------------------------------------------------------------
# TestToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:

    def _registry_with_tool(self, **kw):
        reg = ToolRegistry()
        tool = _make_dummy_tool(**kw)
        reg.register(tool)
        return reg, tool

    def test_register_and_get(self):
        reg, tool = self._registry_with_tool()
        assert reg.get("dummy") is tool

    def test_alias_resolution(self):
        reg = ToolRegistry()
        tool = _make_dummy_tool(name="exec")
        reg.register(tool)
        assert reg.get("bash") is tool
        assert reg.get("shell") is tool
        assert reg.has("bash")

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = await reg.execute("nonexistent", {})
        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_invalid_params(self):
        reg, _ = self._registry_with_tool()
        result = await reg.execute("dummy", {})  # missing required 'msg'
        assert not result.success
        assert "Invalid parameters" in result.error

    @pytest.mark.asyncio
    async def test_execute_success_with_logging(self):
        reg, _ = self._registry_with_tool()
        result = await reg.execute("dummy", {"msg": "hi"})
        assert result.success
        log = reg.get_execution_log()
        assert len(log) == 1
        assert log[0]["tool"] == "dummy"
        assert log[0]["success"] is True

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        async def slow_exec(params, ctx):
            await asyncio.sleep(10)
            return ToolResult(output="done")

        reg, _ = self._registry_with_tool(timeout_seconds=0.05, execute_cb=slow_exec)
        result = await reg.execute("dummy", {"msg": "hi"})
        assert not result.success
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_execute_retry_on_exception(self):
        call_count = 0

        async def flaky_exec(params, ctx):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return ToolResult(output="recovered")

        reg, _ = self._registry_with_tool(max_retries=2, execute_cb=flaky_exec)
        result = await reg.execute("dummy", {"msg": "hi"})
        assert result.success
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_replay_guard_blocks_duplicate(self):
        reg, _ = self._registry_with_tool()
        ctx = ToolExecutionContext(
            execution_id="e1",
            trace_id="t1",
            idempotency_key="unique-key-123",
        )
        r1 = await reg.execute("dummy", {"msg": "hi"}, ctx=ctx)
        assert r1.success

        r2 = await reg.execute("dummy", {"msg": "hi"}, ctx=ctx)
        assert not r2.success
        assert "Replay prevented" in r2.error

    def test_get_definitions(self):
        reg, _ = self._registry_with_tool()
        defs = reg.get_definitions()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"

    def test_unregister(self):
        reg, _ = self._registry_with_tool()
        reg.unregister("dummy")
        assert reg.get("dummy") is None
        assert "dummy" not in reg.tool_names

    def test_clear_log(self):
        reg = ToolRegistry()
        reg._execution_log.append({"test": True})
        reg.clear_log()
        assert reg.get_execution_log() == []


# ---------------------------------------------------------------------------
# TestTodoTool
# ---------------------------------------------------------------------------


class TestTodoTool:

    @pytest.fixture
    def todo(self, tmp_path: Path) -> TodoTool:
        return TodoTool(store_dir=tmp_path / "todos")

    @pytest.mark.asyncio
    async def test_create_single_task(self, todo: TodoTool):
        result = await todo.execute({"action": "create", "title": "Buy milk"})
        assert result.success
        assert "Buy milk" in result.output

    @pytest.mark.asyncio
    async def test_create_batch_items(self, todo: TodoTool):
        items = [{"title": "Task A"}, {"title": "Task B", "notes": "urgent"}]
        result = await todo.execute({"action": "create", "items": items})
        assert result.success
        assert "Task A" in result.output
        assert "Task B" in result.output

    @pytest.mark.asyncio
    async def test_list_tasks(self, todo: TodoTool):
        await todo.execute({"action": "create", "title": "Alpha"})
        result = await todo.execute({"action": "list"})
        assert result.success
        assert "Alpha" in result.output
        assert "pending" in result.output

    @pytest.mark.asyncio
    async def test_list_empty(self, todo: TodoTool):
        result = await todo.execute({"action": "list"})
        assert result.output == "No tasks."

    @pytest.mark.asyncio
    async def test_complete_task(self, todo: TodoTool):
        create_result = await todo.execute({"action": "create", "title": "Finish report"})
        task_id = create_result.output.split("Created task ")[1].split(":")[0]
        result = await todo.execute({"action": "complete", "task_id": task_id})
        assert result.success
        assert "Completed" in result.output

        listing = await todo.execute({"action": "list"})
        assert "done" in listing.output

    @pytest.mark.asyncio
    async def test_delete_task(self, todo: TodoTool):
        create_result = await todo.execute({"action": "create", "title": "Temp task"})
        task_id = create_result.output.split("Created task ")[1].split(":")[0]
        result = await todo.execute({"action": "delete", "task_id": task_id})
        assert result.success
        assert "Deleted" in result.output

        listing = await todo.execute({"action": "list"})
        assert listing.output == "No tasks."

    @pytest.mark.asyncio
    async def test_update_task(self, todo: TodoTool):
        create_result = await todo.execute({"action": "create", "title": "Draft"})
        task_id = create_result.output.split("Created task ")[1].split(":")[0]
        result = await todo.execute({
            "action": "update",
            "task_id": task_id,
            "title": "Final",
            "status": "in_progress",
            "notes": "WIP",
        })
        assert result.success
        assert "Updated" in result.output

        listing = await todo.execute({"action": "list"})
        assert "Final" in listing.output
        assert "in_progress" in listing.output

    @pytest.mark.asyncio
    async def test_find_by_title(self, todo: TodoTool):
        await todo.execute({"action": "create", "title": "Unique title"})
        result = await todo.execute({"action": "complete", "title": "Unique title"})
        assert result.success
        assert "Completed" in result.output

    @pytest.mark.asyncio
    async def test_task_not_found(self, todo: TodoTool):
        result = await todo.execute({"action": "complete", "task_id": "nonexistent"})
        assert not result.success
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# TestPatchTool
# ---------------------------------------------------------------------------


class TestPatchTool:

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "workspace"
        ws.mkdir()
        return ws

    @pytest.fixture
    def patch_tool(self, workspace: Path) -> PatchTool:
        return PatchTool(workspace=str(workspace), restrict=False)

    @pytest.fixture
    def patch_tool_restricted(self, workspace: Path) -> PatchTool:
        return PatchTool(workspace=str(workspace), restrict=True)

    @pytest.fixture
    def sample_file(self, workspace: Path) -> Path:
        f = workspace / "hello.py"
        f.write_text("def greet():\n    print('hello')\n    return True\n")
        return f

    @pytest.mark.asyncio
    async def test_search_replace_exact_match(self, patch_tool: PatchTool, sample_file: Path):
        patch_text = (
            "<<<SEARCH\n"
            "    print('hello')\n"
            "===\n"
            "    print('world')\n"
            "REPLACE>>>"
        )
        result = await patch_tool.execute({"file_path": "hello.py", "patch": patch_text})
        assert result.success
        assert "Applied 1/1" in result.output
        assert "print('world')" in sample_file.read_text()

    @pytest.mark.asyncio
    async def test_search_replace_fuzzy_match(self, patch_tool: PatchTool, workspace: Path):
        f = workspace / "fuzzy.py"
        f.write_text("def greet():\n    print('hello world')\n    return True\n")
        patch_text = (
            "<<<SEARCH\n"
            "    print('hello  world')\n"
            "===\n"
            "    print('goodbye')\n"
            "REPLACE>>>"
        )
        result = await patch_tool.execute({
            "file_path": "fuzzy.py",
            "patch": patch_text,
            "fuzzy_threshold": 0.5,
        })
        assert result.success
        assert "goodbye" in f.read_text()

    @pytest.mark.asyncio
    async def test_unified_diff_apply(self, patch_tool: PatchTool, sample_file: Path):
        diff_text = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def greet():\n"
            "-    print('hello')\n"
            "+    print('patched')\n"
            "     return True\n"
        )
        result = await patch_tool.execute({"file_path": "hello.py", "patch": diff_text})
        assert result.success
        content = sample_file.read_text()
        assert "patched" in content

    @pytest.mark.asyncio
    async def test_path_restriction_blocks_outside(self, patch_tool_restricted: PatchTool):
        patch_text = "<<<SEARCH\nfoo\n===\nbar\nREPLACE>>>"
        result = await patch_tool_restricted.execute({
            "file_path": "../../etc/passwd",
            "patch": patch_text,
        })
        assert not result.success
        assert "outside workspace" in result.error.lower()

    @pytest.mark.asyncio
    async def test_file_not_found(self, patch_tool: PatchTool):
        patch_text = "<<<SEARCH\nfoo\n===\nbar\nREPLACE>>>"
        result = await patch_tool.execute({"file_path": "missing.py", "patch": patch_text})
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_blocks_matched(self, patch_tool: PatchTool, sample_file: Path):
        patch_text = (
            "<<<SEARCH\n"
            "completely unrelated text that does not exist\n"
            "===\n"
            "replacement\n"
            "REPLACE>>>"
        )
        result = await patch_tool.execute({
            "file_path": "hello.py",
            "patch": patch_text,
            "fuzzy_threshold": 0.99,
        })
        assert not result.success
        assert "No blocks matched" in result.error
