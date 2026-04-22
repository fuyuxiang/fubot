"""Agent loop — the core processing engine.

Receives events → builds context → calls LLM → executes tools → sends responses.
Integrates all subsystems: session, memory, tools, permissions, observability.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from echo_agent.agent.context import ContextBuilder, build_skills_context, build_memory_context
from echo_agent.agent.compression import ConversationCompressor
from echo_agent.agent.tools.base import ToolExecutionContext, ToolResult, build_idempotency_key
from echo_agent.agent.tools.registry import ToolRegistry
from echo_agent.bus.events import InboundEvent, OutboundEvent
from echo_agent.bus.queue import MessageBus
from echo_agent.config.schema import Config
from echo_agent.memory.consolidator import MemoryConsolidator
from echo_agent.memory.store import MemoryStore
from echo_agent.models.inference import InferenceController
from echo_agent.models.provider import LLMProvider, LLMResponse
from echo_agent.observability.monitor import TraceLogger
from echo_agent.permissions.manager import ApprovalManager, CredentialManager, PermissionManager
from echo_agent.session.manager import Session, SessionManager
from echo_agent.skills.store import SkillStore
from echo_agent.utils.text import strip_thinking


class AgentLoop:
    """Core processing engine that ties all subsystems together."""

    _MAX_TOOL_RESULT_CHARS = 16000

    def __init__(
        self,
        bus: MessageBus,
        config: Config,
        provider: LLMProvider,
        workspace: Path,
        scheduler: Any = None,
        storage: Any = None,
        task_manager: Any = None,
        workflow_engine: Any = None,
    ):
        self.bus = bus
        self.config = config
        self.provider = provider
        self.workspace = workspace

        self.sessions = SessionManager(
            sessions_dir=workspace / config.storage.sessions_dir,
            expiry_hours=config.session.expiry_hours,
            storage=storage,
        )
        self.memory = MemoryStore(
            memory_dir=workspace / config.storage.memory_dir,
            max_user=config.memory.max_user_memories,
            max_env=config.memory.max_env_memories,
            decay_half_life_days=config.memory.importance_decay_days,
            storage=storage,
        )
        self.tools = ToolRegistry()
        self.context = ContextBuilder(workspace)
        self.compressor = ConversationCompressor(
            config=config.compression,
            context_window_tokens=config.session.context_window_tokens,
            provider=provider,
            default_model=config.models.default_model,
        )
        self.permissions = PermissionManager(admin_users=config.permissions.admin_users)
        self.approval = ApprovalManager(
            require_approval=config.permissions.approval.require_approval,
            auto_approve=config.permissions.approval.auto_approve,
            auto_deny=config.permissions.approval.auto_deny,
            default_policy=config.permissions.approval.default_policy,
        )
        self.inference = InferenceController()
        if config.permissions.approval.require_approval:
            from echo_agent.models.inference import InferenceConstraints
            self.inference.set_constraints(InferenceConstraints(
                require_confirmation_for=list(config.permissions.approval.require_approval),
                blocked_tools=list(config.permissions.approval.auto_deny),
            ))
        self.credentials = CredentialManager(store_path=workspace / "data" / "credentials.json")
        self.tracer = TraceLogger(logs_dir=workspace / config.storage.logs_dir)
        self.consolidator = MemoryConsolidator(
            memory_store=self.memory,
            llm_call=self.provider.chat_with_retry,
            context_window_tokens=config.session.context_window_tokens,
            consolidation_threshold=config.memory.consolidation_threshold,
        )
        self.mcp_manager: Any = None

        skills_dir = workspace / config.skills.skills_dir
        self.skill_store = SkillStore(
            user_dir=Path.home() / ".echo-agent" / "skills",
            builtin_dir=skills_dir if skills_dir.exists() else None,
            external_dirs=[Path(d) for d in config.skills.external_dirs],
            disabled=config.skills.disabled,
        )

        self._running = False
        self._max_iterations = 40
        self._nudge_interval = config.skills.creation_nudge_interval
        self._memory_nudge_interval = config.memory.memory_nudge_interval
        self._tool_iters_since_skill_check = 0
        self._tool_iters_since_memory_check = 0
        self._snapshot_enabled = config.memory.snapshot_enabled
        self._memory_snapshots: dict[str, str] = {}
        self._register_tools(scheduler=scheduler, task_manager=task_manager, workflow_engine=workflow_engine)

    def _register_tools(self, scheduler: Any = None, task_manager: Any = None, workflow_engine: Any = None) -> None:
        from echo_agent.agent.tools import discover_tools
        all_tools = discover_tools(
            config=self.config,
            workspace=self.workspace,
            bus=self.bus,
            provider=self.provider,
            scheduler=scheduler,
            session_manager=self.sessions,
            skill_store=self.skill_store,
            memory_store=self.memory,
            task_manager=task_manager,
            workflow_engine=workflow_engine,
        )
        for tool in all_tools:
            self.tools.register(tool)

    async def start(self) -> None:
        self._running = True
        await self._start_mcp()
        self.bus.subscribe_inbound(self._on_inbound)
        logger.info("Agent loop started")

    async def stop(self) -> None:
        self._running = False
        if self.mcp_manager:
            await self.mcp_manager.stop_all()
        logger.info("Agent loop stopped")

    async def _start_mcp(self) -> None:
        mcp_servers = self.config.tools.mcp_servers
        if not mcp_servers:
            return
        from echo_agent.mcp.manager import MCPManager
        self.mcp_manager = MCPManager(workspace=self.workspace)
        await self.mcp_manager.start_all(mcp_servers)
        await self.mcp_manager.discover_tools(self.tools)

    async def _on_inbound(self, event: InboundEvent) -> None:
        if not self._running:
            return
        trace_id = uuid.uuid4().hex[:12]
        span = self.tracer.start_span(trace_id, f"s_{trace_id}", "process_message", "input")
        try:
            response = await self._process_event(event, trace_id)
            if response:
                out = OutboundEvent.text_reply(
                    channel=event.channel,
                    chat_id=event.chat_id,
                    text=response,
                    reply_to_id=event.event_id,
                )
                await self.bus.publish_outbound(out)
            self.tracer.end_span(span, metadata={"response_len": len(response or "")})
        except Exception as e:
            logger.error("Processing failed for event {}: {}", event.event_id, e)
            self.tracer.end_span(span, error=str(e))
            error_out = OutboundEvent.text_reply(
                channel=event.channel, chat_id=event.chat_id,
                text=f"Sorry, an error occurred: {e}",
            )
            await self.bus.publish_outbound(error_out)
        finally:
            self.tracer.flush_trace(trace_id)

    async def _process_event(self, event: InboundEvent, trace_id: str) -> str | None:
        session = await self.sessions.get_or_create(event.session_key)

        if self._snapshot_enabled:
            if event.session_key not in self._memory_snapshots:
                self._memory_snapshots[event.session_key] = self.memory.get_snapshot()
            memory_ctx = build_memory_context(self.memory, snapshot=self._memory_snapshots[event.session_key])
        else:
            memory_ctx = build_memory_context(self.memory)

        skills_ctx = build_skills_context(self.skill_store)
        system_prompt = self.context.build_system_prompt(memory_context=memory_ctx, skills_context=skills_ctx)

        history = session.get_history(self.config.session.max_history_messages)
        if self.compressor.should_compress(history):
            result = await self.compressor.compress(history, focus_topic=event.text)
            history = result.messages
            if result.was_compressed:
                logger.info("Context compressed: {} → {} tokens", result.tokens_before, result.tokens_after)
                session.messages = session.messages[:session.last_consolidated] + result.messages
                await self.sessions.save(session)
        session.add_message("user", event.text)

        retrieval = ""
        if self.config.memory.enabled:
            scored = self.memory.search_scored(event.text, limit=5)
            if scored:
                retrieval = "\n".join(f"- {r.key}: {r.content}" for r, _ in scored)

        messages = self.context.build_messages(
            history=history,
            current_message=event.text,
            channel=event.channel,
            chat_id=event.chat_id,
            system_prompt=system_prompt,
            retrieval_context=retrieval,
        )

        tool_defs = self.inference.filter_tools(self.tools.get_definitions())

        async def _emit_progress(text: str, *, tool_hint: bool = False) -> None:
            out = OutboundEvent.text_reply(
                channel=event.channel, chat_id=event.chat_id, text=text,
            )
            out.is_final = False
            out.message_kind = "tool" if tool_hint else "progress"
            out.metadata = {"_progress": True, "_tool_hint": tool_hint}
            await self.bus.publish_outbound(out)

        response_text = ""
        should_review_skills = False
        should_review_memory = False
        total_tool_calls = 0
        for iteration in range(self._max_iterations):
            llm_span = self.tracer.start_span(trace_id, f"llm_{iteration}", "llm_call", "llm_call")
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                model=self.config.models.default_model,
            )
            self.tracer.end_span(
                llm_span,
                metadata={"model": self.config.models.default_model, "finish": response.finish_reason},
            )

            issues = self.inference.validate_response(response)
            if issues:
                logger.warning("Inference issues: {}", issues)

            if response.content:
                response_text = response.content

            if not response.has_tool_calls:
                break

            if response.content:
                await _emit_progress(response.content)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content}
            assistant_msg["tool_calls"] = [tc.to_openai_format() for tc in response.tool_calls]
            messages.append(assistant_msg)

            tool_call_fmts = [tc.to_openai_format() for tc in response.tool_calls]
            session.add_message("assistant", response.content or "", tool_calls=tool_call_fmts)

            for i, tc in enumerate(response.tool_calls):
                tool_span = self.tracer.start_span(trace_id, f"tool_{iteration}_{i}", f"tool:{tc.name}", "tool_call")

                await _emit_progress(f"Using tool: {tc.name}", tool_hint=True)

                if self.inference.needs_confirmation(tc.name):
                    logger.info("Tool {} requires confirmation (skipping in auto mode)", tc.name)

                if not self.permissions.check_tool(tc.name, user_id=event.sender_id):
                    result = ToolResult(success=False, error=f"Permission denied for tool '{tc.name}'")
                    result_text = result.text
                    self.tracer.end_span(tool_span, metadata={"success": False, "denied": True})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result_text})
                    session.add_message("tool", result_text, tool_call_id=tc.id, name=tc.name)
                    total_tool_calls += 1
                    continue

                ctx = ToolExecutionContext(
                    execution_id=uuid.uuid4().hex[:12],
                    trace_id=trace_id,
                    session_key=event.session_key,
                    user_id=event.sender_id,
                    attempt_index=0,
                    idempotency_key=build_idempotency_key(trace_id, tc.name, i, tc.arguments),
                )
                result = await self.tools.execute(tc.name, tc.arguments, ctx)
                result_text = result.text
                if len(result_text) > self._MAX_TOOL_RESULT_CHARS:
                    result_text = result_text[:self._MAX_TOOL_RESULT_CHARS] + "\n...(truncated)"

                self.tracer.end_span(tool_span, metadata={"success": result.success})

                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result_text})
                session.add_message("tool", result_text, tool_call_id=tc.id, name=tc.name)

                total_tool_calls += 1
                self._tool_iters_since_skill_check += 1
                self._tool_iters_since_memory_check += 1
                if (
                    self._nudge_interval > 0
                    and self._tool_iters_since_skill_check >= self._nudge_interval
                    and self.tools.has("skill_manage")
                ):
                    should_review_skills = True
                    self._tool_iters_since_skill_check = 0
                if (
                    self._memory_nudge_interval > 0
                    and self._tool_iters_since_memory_check >= self._memory_nudge_interval
                    and self.tools.has("memory")
                ):
                    should_review_memory = True
                    self._tool_iters_since_memory_check = 0

        if response_text:
            response_text = strip_thinking(response_text)

        session.add_message("assistant", response_text)
        self.sessions.save(session)

        if self.consolidator.should_consolidate(session.message_count, session.last_consolidated):
            asyncio.create_task(self._consolidate(session))

        if should_review_skills and total_tool_calls > 0:
            asyncio.create_task(self._background_skill_review(messages))

        if should_review_memory and total_tool_calls > 0:
            asyncio.create_task(self._background_memory_review(messages))

        return response_text

    async def _background_skill_review(self, messages: list[dict[str, Any]]) -> None:
        try:
            from echo_agent.skills.reviewer import SkillReviewer
            reviewer = SkillReviewer(
                provider=self.provider,
                store=self.skill_store,
                model=self.config.models.default_model,
            )
            actions = await reviewer.review(messages)
            if actions:
                logger.info("Background skill review: {}", "; ".join(actions))
        except Exception as e:
            logger.warning("Background skill review failed: {}", e)

    async def _background_memory_review(self, messages: list[dict[str, Any]]) -> None:
        try:
            from echo_agent.memory.reviewer import MemoryReviewer
            reviewer = MemoryReviewer(
                provider=self.provider,
                store=self.memory,
                model=self.config.models.default_model,
            )
            actions = await reviewer.review(messages)
            if actions:
                logger.info("Background memory review: {}", "; ".join(actions))
        except Exception as e:
            logger.warning("Background memory review failed: {}", e)

    async def _consolidate(self, session: Session) -> None:
        try:
            chunk = session.messages[session.last_consolidated:]
            if await self.consolidator.consolidate_chunk(chunk):
                session.last_consolidated = len(session.messages)
                await self.sessions.save(session)        except Exception as e:
            logger.error("Consolidation failed for {}: {}", session.key, e)

    async def process_direct(self, content: str, session_key: str = "cli:direct") -> str:
        """Process a message directly (for CLI or testing)."""
        event = InboundEvent.text_message(
            channel="cli", sender_id="user", chat_id="direct", text=content,
            session_key_override=session_key,
        )
        return await self._process_event(event, uuid.uuid4().hex[:12]) or ""
