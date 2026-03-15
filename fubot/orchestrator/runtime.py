"""Coordinator runtime that dispatches routed executor work."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from fubot.config.schema import AgentProfile, Config
from fubot.orchestrator.models import AssignmentRecord, ExecutionLogRecord, TaskRecord, WorkflowRecord
from fubot.orchestrator.router import RoutePlanner
from fubot.orchestrator.store import WorkflowStore


@dataclass
class ExecutorResult:
    """Result returned by an executor runtime."""

    profile: AgentProfile
    task: TaskRecord
    content: str
    route: dict[str, Any]
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    sent_direct_message: bool = False


class CoordinatorRuntime:
    """Plans tasks, dispatches executors, and persists workflow state."""

    def __init__(self, config: Config, store: WorkflowStore):
        self.config = config
        self.store = store
        self.router = RoutePlanner(config, self.store.load_health())

    async def run(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        content: str,
        media: list[str] | None,
        execute_task: Callable[
            [AgentProfile, TaskRecord, str, str, str, str, list[str] | None, dict[str, Any], Callable[[str, bool], Awaitable[None]] | None],
            Awaitable[ExecutorResult],
        ],
        emit_update: Callable[[ExecutionLogRecord], Awaitable[None]] | None = None,
    ) -> tuple[WorkflowRecord, list[TaskRecord], list[AssignmentRecord], list[ExecutorResult], str]:
        workflow = WorkflowRecord(
            id=f"wf_{uuid.uuid4().hex[:10]}",
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            user_message=content,
            shared_board={"user_message": content, "media": media or []},
        )
        task_type = self.router.classify(content, media)
        profiles = self.router.choose_executors(task_type, content)
        tasks: list[TaskRecord] = []
        assignments: list[AssignmentRecord] = []

        for profile in profiles:
            task = TaskRecord(
                id=f"task_{uuid.uuid4().hex[:8]}",
                workflow_id=workflow.id,
                title=f"{profile.name or profile.id} {task_type}",
                kind=task_type,
                status="queued",
                metadata={"input": content, "media": media or []},
            )
            assignment = AssignmentRecord(
                id=f"assign_{uuid.uuid4().hex[:8]}",
                workflow_id=workflow.id,
                task_id=task.id,
                agent_id=profile.id,
                agent_name=profile.name or profile.id,
                agent_role=profile.role,
            )
            workflow.task_ids.append(task.id)
            workflow.assignment_ids.append(assignment.id)
            tasks.append(task)
            assignments.append(assignment)

        self.store.save_workflow(workflow, tasks, assignments)

        def _progress_factory(profile: AgentProfile, task: TaskRecord) -> Callable[[str, bool], Awaitable[None]]:
            async def _emit(message: str, tool_hint: bool = False) -> None:
                if not self.config.observability.show_executor_progress:
                    return
                kind = "tool" if tool_hint else "progress"
                record = ExecutionLogRecord(
                    workflow_id=workflow.id,
                    task_id=task.id,
                    agent_id=profile.id,
                    agent_name=profile.name or profile.id,
                    agent_role=profile.role,
                    message_kind=kind,
                    content=message,
                    is_final=False,
                )
                self.store.append_log(workflow, tasks, assignments, record)
                if emit_update:
                    await emit_update(record)

            return _emit

        async def _run_one(profile: AgentProfile, task: TaskRecord, assignment: AssignmentRecord) -> ExecutorResult:
            self.router.begin_load(profile.id)
            assignment.status = "running"
            task.status = "running"
            self.store.save_workflow(workflow, tasks, assignments)
            try:
                result = await execute_task(
                    profile,
                    task,
                    workflow.id,
                    session_key,
                    channel,
                    chat_id,
                    media,
                    workflow.shared_board,
                    _progress_factory(profile, task),
                )
            except Exception:
                task.status = "failed"
                assignment.status = "failed"
                self.router.mark_failure(profile.id, None)
                self.router.end_load(profile.id)
                self.store.save_workflow(workflow, tasks, assignments)
                raise

            task.status = "completed"
            task.assigned_agent_id = profile.id
            task.assigned_agent_name = profile.name or profile.id
            task.route = result.route
            task.result = result.content
            task.summary = result.content[:400]
            assignment.status = "completed"
            self.router.mark_success(profile.id, result.route.get("provider"))
            self.router.end_load(profile.id)
            self.store.update_shared_board(
                workflow,
                tasks,
                assignments,
                profile.id,
                {"content": result.content, "route": result.route, "tools_used": result.tools_used},
            )
            final_log = ExecutionLogRecord(
                workflow_id=workflow.id,
                task_id=task.id,
                agent_id=profile.id,
                agent_name=profile.name or profile.id,
                agent_role=profile.role,
                message_kind="result",
                content=result.content,
                is_final=False,
            )
            self.store.append_log(workflow, tasks, assignments, final_log)
            if emit_update:
                await emit_update(final_log)
            return result

        results = await asyncio.gather(
            *[_run_one(profile, task, assignment) for profile, task, assignment in zip(profiles, tasks, assignments)],
        )
        final_response = self._synthesize_final(results)
        workflow.status = "completed"
        workflow.final_response = final_response
        self.store.save_workflow(workflow, tasks, assignments)
        self.store.save_health(self.router.export_health())
        return workflow, tasks, assignments, results, final_response

    def _synthesize_final(self, results: list[ExecutorResult]) -> str:
        if not results:
            return "No executor produced a result."
        if len(results) == 1:
            return results[0].content
        lines = []
        for result in results:
            header = result.profile.name or result.profile.id
            lines.append(f"[{header}] {result.content}")
        return "\n\n".join(lines)
