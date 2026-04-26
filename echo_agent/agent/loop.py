"""Agent loop — the core processing engine.

Receives events → builds context → calls LLM → executes tools → sends responses.
Integrates all subsystems: session, memory, tools, permissions, observability.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from echo_agent.models.provider import LLMProvider, LLMResponse, ToolCallRequest
from echo_agent.models.router import ModelRouter, RouteDecision
from echo_agent.observability.monitor import TraceLogger
from echo_agent.permissions.manager import (
    ApprovalManager, ApprovalStatus, CredentialManager, Effect, PermissionLevel, PermissionManager, PermissionRule,
)
from echo_agent.runtime_paths import bundled_skills_dir
from echo_agent.session.manager import Session, SessionManager
from echo_agent.skills.store import SkillStore
from echo_agent.utils.text import strip_thinking


@dataclass
class _ProcessResult:
    response_text: str = ""
    outbound_sent: bool = False


class _TokenStreamPublisher:
    _PARAGRAPH_RE = re.compile(r"\n\n")
    _SENTENCE_RE = re.compile(r"[。！？!?]")

    def __init__(
        self,
        bus: MessageBus,
        event: InboundEvent,
        *,
        enabled: bool,
        flush_chars: int,
        flush_interval_ms: int,
        paragraph_mode: bool = True,
        intro_text: str = "",
    ):
        self._bus = bus
        self._event = event
        self._enabled = enabled
        self._flush_chars = max(1, flush_chars)
        self._flush_interval = max(0.05, flush_interval_ms / 1000.0)
        self._paragraph_mode = paragraph_mode
        self._full_text = ""
        self._pending = ""
        self._last_flush = time.monotonic()
        self._sent_nonfinal = False
        self._intro_text = intro_text.strip()
        self._needs_intro_separator = bool(self._intro_text)

    async def start(self) -> None:
        if not self._enabled or not self._intro_text:
            return
        self._full_text = self._intro_text
        self._pending = self._intro_text

    async def on_delta(self, delta: str) -> None:
        if not self._enabled or not delta:
            return
        if self._needs_intro_separator:
            self._full_text += "\n\n"
            self._pending += "\n\n"
            self._needs_intro_separator = False
        self._full_text += delta
        self._pending += delta
        now = time.monotonic()

        if self._paragraph_mode:
            boundary = self._find_paragraph_boundary()
            if boundary > 0 and len(self._pending[:boundary]) >= self._flush_chars:
                await self._flush_up_to(boundary, is_final=False)
                return
            elapsed = now - self._last_flush
            if elapsed >= self._flush_interval:
                sentence_end = self._find_sentence_boundary()
                if sentence_end > 0 and len(self._pending[:sentence_end]) >= self._flush_chars:
                    await self._flush_up_to(sentence_end, is_final=False)
                elif sentence_end > 0 and elapsed >= self._flush_interval * 2:
                    await self._flush_up_to(sentence_end, is_final=False)
                elif elapsed >= self._flush_interval * 3:
                    await self._flush(is_final=False)
        else:
            if len(self._pending) >= self._flush_chars or now - self._last_flush >= self._flush_interval:
                await self._flush(is_final=False)

    def _find_paragraph_boundary(self) -> int:
        m = None
        for m in self._PARAGRAPH_RE.finditer(self._pending):
            pass
        return m.end() if m else 0

    def _find_sentence_boundary(self) -> int:
        m = None
        for m in self._SENTENCE_RE.finditer(self._pending):
            pass
        return m.end() if m else 0

    async def finalize(self, final_text: str) -> bool:
        if not self._enabled:
            return False

        if final_text.startswith(self._full_text):
            self._pending += final_text[len(self._full_text):]
            self._full_text = final_text
        elif not self._sent_nonfinal:
            self._full_text = final_text
            self._pending = final_text
        elif final_text != self._full_text:
            logger.debug("Stream final text diverged from streamed text for channel {}", self._event.channel)
            self._full_text = final_text

        if self._sent_nonfinal:
            if self._pending:
                await self._flush(is_final=True)
            else:
                await self._publish("", is_final=True)
            return True

        await self._publish(final_text, is_final=True)
        return True

    async def _flush(self, *, is_final: bool) -> None:
        text = self._pending
        self._pending = ""
        await self._publish(text, is_final=is_final)

    async def _flush_up_to(self, pos: int, *, is_final: bool) -> None:
        text = self._pending[:pos]
        self._pending = self._pending[pos:]
        await self._publish(text, is_final=is_final)

    async def _publish(self, text: str, *, is_final: bool) -> None:
        outbound = OutboundEvent.text_reply(
            channel=self._event.channel,
            chat_id=self._event.chat_id,
            text=text,
            reply_to_id=self._event.reply_to_id,
        )
        outbound.is_final = is_final
        outbound.message_kind = "final" if is_final else "streaming"
        outbound.metadata = dict(self._event.metadata)
        outbound.metadata["_inbound_event_id"] = self._event.event_id
        outbound.metadata["_token_stream"] = True
        await self._bus.publish_outbound(outbound)
        self._last_flush = time.monotonic()
        if not is_final and text:
            self._sent_nonfinal = True


def _resolve_builtin_skills_dir(workspace: Path, configured_path: str) -> Path | None:
    raw_path = Path(configured_path).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(workspace / raw_path)
        bundled = bundled_skills_dir()
        if bundled:
            candidates.append(bundled)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class AgentLoop:
    """Core processing engine that ties all subsystems together."""

    _MAX_TOOL_RESULT_CHARS = 16000

    def __init__(
        self,
        bus: MessageBus,
        config: Config,
        provider: LLMProvider,
        workspace: Path,
        router: ModelRouter | None = None,
        scheduler: Any = None,
        storage: Any = None,
        task_manager: Any = None,
        workflow_engine: Any = None,
    ):
        self.bus = bus
        self.config = config
        self.provider = provider
        self.router = router
        self.workspace = workspace
        try:
            provider_default_model = provider.get_default_model()
        except Exception:
            provider_default_model = ""
        self._default_model = config.models.default_model or provider_default_model or ""

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
            default_model=self._default_model,
        )
        try:
            from echo_agent.models.tokenizer import TokenCounter
            provider_name = getattr(config.models, "default_provider", "") or ""
            if not provider_name and config.models.providers:
                provider_name = config.models.providers[0].name
            if self._default_model:
                tc = TokenCounter.for_model(provider_name, self._default_model)
                if hasattr(self.compressor, 'set_token_counter'):
                    self.compressor.set_token_counter(tc)
        except Exception:
            pass
        self.permissions = PermissionManager(admin_users=config.permissions.admin_users)
        for rule_cfg in config.permissions.rules:
            self.permissions.add_rule(PermissionRule(
                level=PermissionLevel(rule_cfg.level),
                subject=rule_cfg.subject,
                action=rule_cfg.action,
                effect=Effect(rule_cfg.effect),
                scope=rule_cfg.scope,
                priority=rule_cfg.priority,
            ))
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
        self.credentials = CredentialManager(
            store_path=workspace / "data" / "credentials.json",
            encryption_key_env=config.credentials.encryption_key_env,
            require_encryption=config.credentials.require_encryption,
        )
        self.tracer = TraceLogger(logs_dir=workspace / config.storage.logs_dir)
        self.consolidator = MemoryConsolidator(
            memory_store=self.memory,
            llm_call=self.provider.chat_with_retry,
            context_window_tokens=config.session.context_window_tokens,
            consolidation_threshold=config.memory.consolidation_threshold,
        )

        self._working_memories: dict[str, Any] = {}
        self._hybrid_retriever = None
        self._prefetcher = None
        if config.memory.enabled:
            self._init_advanced_memory(config, storage)

        self.planner = None
        if config.planning.enabled:
            from echo_agent.agent.planning import AgentPlanner
            self.planner = AgentPlanner(
                llm_call=self.provider.chat_with_retry,
                default_strategy=config.planning.default_strategy,
                max_tree_depth=config.planning.max_tree_depth,
                reflection_enabled=config.planning.reflection_enabled,
            )

        self._telemetry = None
        if config.observability.otel_enabled:
            from echo_agent.observability.telemetry import TelemetryManager
            self._telemetry = TelemetryManager(
                service_name=config.observability.otel_service_name,
                otel_endpoint=config.observability.otel_endpoint,
                export_interval_ms=config.observability.otel_export_interval_ms,
            )
            self._telemetry.setup()
            if self._telemetry.available:
                self.tracer.set_otel_tracer(self._telemetry.get_tracer())
        self.mcp_manager: Any = None
        self.knowledge: Any = None
        self.multi_agent: Any = None
        if config.knowledge.enabled:
            from echo_agent.knowledge import KnowledgeIndex
            self.knowledge = KnowledgeIndex(
                workspace=workspace,
                docs_dir=config.knowledge.docs_dir,
                index_path=config.knowledge.index_path,
                chunk_size=config.knowledge.chunk_size,
                chunk_overlap=config.knowledge.chunk_overlap,
                allowed_extensions=config.knowledge.allowed_extensions,
            )
            self.knowledge.ensure_ready(auto_index=config.knowledge.auto_index)

        skills_dir = _resolve_builtin_skills_dir(workspace, config.skills.skills_dir)
        self.skill_store = SkillStore(
            user_dir=workspace / "data" / "skills",
            builtin_dir=skills_dir,
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
        self._setup_multi_agent()

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
            knowledge_index=self.knowledge,
        )
        for tool in all_tools:
            self.tools.register(tool)

    def _init_advanced_memory(self, config: Config, storage: Any) -> None:
        """初始化高级记忆子系统：四层记忆、向量索引、知识图谱、混合检索、预测预取。"""
        from echo_agent.memory.tiers import WorkingMemory, EpisodicManager, SemanticManager, ArchivalManager
        from echo_agent.memory.forgetting import ForgettingCurve
        from echo_agent.memory.retrieval import HybridRetriever

        forgetting = self.memory.forgetting_curve

        episodic = EpisodicManager(storage) if storage else None
        semantic = SemanticManager(self.memory)
        archival = ArchivalManager(storage) if storage else None

        self.consolidator.set_episodic_manager(episodic)
        self.consolidator.set_semantic_manager(semantic)
        self.consolidator.set_forgetting_curve(forgetting)
        self.consolidator.set_archival_manager(archival)

        vector_index = None
        embed_fn = None
        if config.memory.vector_enabled and storage:
            from echo_agent.memory.vectors import VectorIndex
            vector_index = VectorIndex(storage, dimensions=config.memory.vector_dimensions)
            self.memory.set_vector_index(vector_index)

        graph = None
        if config.memory.graph_enabled and storage:
            from echo_agent.memory.graph import MemoryGraph
            graph = MemoryGraph(storage)
            self.memory.set_graph(graph)

        if config.memory.contradiction_detection and storage:
            from echo_agent.memory.contradiction import ContradictionDetector
            detector = ContradictionDetector(storage, vector_index)
            self.consolidator.set_contradiction_detector(detector)

        entries_fn = lambda: list(self.memory._entries.values())
        self._hybrid_retriever = HybridRetriever(
            entries_fn=entries_fn,
            vector_index=vector_index,
            graph=graph,
            forgetting=forgetting,
            embed_fn=embed_fn,
        )
        self.memory.set_retriever(self._hybrid_retriever)

        if config.memory.prefetch_enabled:
            from echo_agent.memory.prefetch import PredictivePrefetch
            self._prefetcher = PredictivePrefetch(
                retriever=self._hybrid_retriever,
                llm_call=self.provider.chat_with_retry,
            )

    def _setup_multi_agent(self) -> None:
        if not self.config.multi_agent.enabled:
            return
        from echo_agent.agent.multi_agent import AgentRegistry, IntentRouter, MultiAgentRuntime
        from echo_agent.agent.tools.agents import AgentsListTool, AgentsRouteTool
        registry = AgentRegistry.from_config(self.config.multi_agent)
        router = IntentRouter(
            registry,
            default_agent=self.config.multi_agent.default_agent,
            route_threshold=self.config.multi_agent.route_threshold,
            multi_threshold=self.config.multi_agent.multi_threshold,
            max_parallel_agents=self.config.multi_agent.max_parallel_agents,
        )
        audit_path = Path(self.config.multi_agent.audit_path).expanduser()
        if not audit_path.is_absolute():
            audit_path = self.workspace / audit_path
        self.multi_agent = MultiAgentRuntime(
            registry=registry,
            router=router,
            provider=self.provider,
            model_router=self.router,
            tools=self.tools,
            audit_path=audit_path,
            max_iterations=self.config.multi_agent.max_iterations,
            synthesize_results=self.config.multi_agent.synthesize_results,
        )
        self.tools.register(AgentsListTool(self.multi_agent))
        self.tools.register(AgentsRouteTool(self.multi_agent))
        logger.info("Multi-agent dispatch enabled with {} profiles", len(registry.list()))

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
        self.mcp_manager = MCPManager(
            workspace=self.workspace,
            security_policy=self.config.tools.mcp_security_policy,
        )
        await self.mcp_manager.start_all(mcp_servers)
        await self.mcp_manager.discover_tools(self.tools)

    async def _on_inbound(self, event: InboundEvent) -> None:
        """入站事件处理入口，负责追踪、错误处理和响应发布。"""
        if not self._running:
            return
        trace_id = uuid.uuid4().hex[:12]
        span = self.tracer.start_span(trace_id, f"s_{trace_id}", "process_message", "input")
        try:
            result = await self._process_event(event, trace_id, publish_response=True)
            response_text = result.response_text
            if response_text and not result.outbound_sent:
                out = OutboundEvent.text_reply(
                    channel=event.channel, chat_id=event.chat_id, text=response_text, reply_to_id=event.reply_to_id,
                )
                out.metadata = dict(event.metadata)
                out.metadata["_inbound_event_id"] = event.event_id
                await self.bus.publish_outbound(out)
            self.tracer.end_span(span, metadata={"response_len": len(response_text or "")})
        except Exception as e:
            logger.error("Processing failed for event {}: {}", event.event_id, e)
            self.tracer.end_span(span, error=str(e))
            error_out = OutboundEvent.text_reply(
                channel=event.channel, chat_id=event.chat_id, text=f"Sorry, an error occurred: {e}", reply_to_id=event.reply_to_id,
            )
            error_out.metadata = dict(event.metadata)
            error_out.metadata["_inbound_event_id"] = event.event_id
            await self.bus.publish_outbound(error_out)
        finally:
            self.tracer.flush_trace(trace_id)

    async def _process_event(self, event: InboundEvent, trace_id: str, *, publish_response: bool = False) -> _ProcessResult:
        """处理单个入站事件的完整流水线。

        流程: 会话初始化 → 上下文构建 → LLM 调用 → 工具执行循环 → 响应输出。

        Args:
            event: 入站事件（用户消息）
            trace_id: 追踪标识符
            publish_response: 是否通过消息总线发布响应
        Returns:
            _ProcessResult: 包含响应文本和发送状态
        """
        session = await self.sessions.get_or_create(event.session_key)
        if event.session_key not in self._working_memories and self._prefetcher:
            from echo_agent.memory.tiers import WorkingMemory
            self._working_memories[event.session_key] = WorkingMemory(
                max_entries=self.config.memory.max_working_memory
            )
        command_response = await self._handle_approval_command(event)
        if command_response is not None:
            return _ProcessResult(response_text=command_response)

        should_introduce = self._should_introduce(session)
        intro_text = self._build_introduction(event) if should_introduce else ""
        stream_publisher = _TokenStreamPublisher(
            self.bus,
            event,
            enabled=publish_response and self._should_stream_channel(event.channel),
            flush_chars=self.config.channels.stream_flush_chars,
            flush_interval_ms=self.config.channels.stream_flush_interval_ms,
            paragraph_mode=self.config.channels.stream_paragraph_mode,
            intro_text=intro_text,
        )
        if publish_response:
            await stream_publisher.start()

        working_ctx = ""
        if self._prefetcher and event.session_key in self._working_memories:
            wm = self._working_memories[event.session_key]
            try:
                await self._prefetcher.prefetch(
                    recent_messages=session.get_history(5),
                    working_memory=wm,
                    session_key=event.session_key,
                )
                working_ctx = wm.get_context()
            except Exception as e:
                logger.debug("Prefetch failed: {}", e)

        if self._snapshot_enabled:
            if event.session_key not in self._memory_snapshots:
                self._memory_snapshots[event.session_key] = self.memory.get_snapshot(session_key=event.session_key)
            memory_ctx = build_memory_context(self.memory, snapshot=self._memory_snapshots[event.session_key], working_memory=working_ctx)
        else:
            memory_ctx = build_memory_context(self.memory, session_key=event.session_key, working_memory=working_ctx)

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

        retrieval_parts: list[str] = []
        if self.config.memory.enabled:
            scored = self.memory.search_scored(event.text, limit=5, session_key=event.session_key)
            if scored:
                retrieval_parts.append("Relevant memory:\n" + "\n".join(f"- {r.key}: {r.content}" for r, _ in scored))

        if self.knowledge:
            knowledge_results = self.knowledge.search(
                event.text,
                limit=self.config.knowledge.max_results,
                user_id=event.sender_id,
            )
            knowledge_context = self.knowledge.format_results(knowledge_results)
            if knowledge_context:
                retrieval_parts.append(knowledge_context)

        task_type = self._infer_task_type(event.text)
        dispatch_plan = None
        if self.multi_agent:
            dispatch_plan = self.multi_agent.plan(event.text, task_type=task_type)
            if self.config.multi_agent.mode == "assist" and dispatch_plan.candidates:
                selected = ", ".join(dispatch_plan.selected_agent_ids) or dispatch_plan.primary_agent_id
                retrieval_parts.append(
                    "Multi-agent routing suggestion:\n"
                    f"- strategy: {dispatch_plan.strategy}\n"
                    f"- selected: {selected}\n"
                    f"- rationale: {dispatch_plan.rationale}"
                )

        retrieval = "\n\n".join(retrieval_parts)

        messages = self.context.build_messages(
            history=history,
            current_message=event.text,
            channel=event.channel,
            chat_id=event.chat_id,
            system_prompt=system_prompt,
            retrieval_context=retrieval,
        )

        tool_defs = self.inference.filter_tools(self.tools.get_definitions())

        current_plan = None
        if self.planner and tool_defs:
            try:
                token_est = len(event.text) // 4
                current_plan = await self.planner.create_plan(
                    query=event.text, tools=tool_defs, context=retrieval, token_estimate=token_est,
                )
                if current_plan and current_plan.steps:
                    plan_context = current_plan.to_prompt()
                    messages[-1]["content"] = messages[-1]["content"] + f"\n\n[Plan]\n{plan_context}"
            except Exception as e:
                logger.debug("Planning failed, proceeding without plan: {}", e)

        async def _emit_progress(text: str, *, tool_hint: bool = False) -> None:
            out = OutboundEvent.text_reply(
                channel=event.channel, chat_id=event.chat_id, text=text, reply_to_id=event.reply_to_id,
            )
            out.is_final = False
            out.message_kind = "tool" if tool_hint else "progress"
            out.metadata = dict(event.metadata)
            out.metadata.update({"_progress": True, "_tool_hint": tool_hint, "_inbound_event_id": event.event_id})
            await self.bus.publish_outbound(out)

        response_text = ""
        should_review_skills = False
        should_review_memory = False
        total_tool_calls = 0

        if (
            self.multi_agent
            and self.config.multi_agent.mode == "auto"
            and dispatch_plan
            and dispatch_plan.should_dispatch
        ):
            selected = ", ".join(dispatch_plan.selected_agent_ids)
            await _emit_progress(f"Dispatching to specialist agent(s): {selected}", tool_hint=True)
            dispatch_span = self.tracer.start_span(trace_id, "multi_agent_dispatch", "multi_agent_dispatch", "agent_dispatch")

            async def _child_tool_executor(
                agent_id: str,
                tool_call: ToolCallRequest,
                index: int,
                child_messages: list[dict[str, Any]],
                allowed_tools: set[str],
            ) -> str:
                return await self._execute_child_tool_call(
                    agent_id=agent_id,
                    tool_call=tool_call,
                    index=index,
                    event=event,
                    trace_id=trace_id,
                    allowed_tools=allowed_tools,
                )

            dispatch_result = await self.multi_agent.dispatch(
                query=event.text,
                plan=dispatch_plan,
                base_messages=messages,
                retrieval_context=retrieval,
                tool_executor=_child_tool_executor,
                trace_id=trace_id,
            )
            self.tracer.end_span(
                dispatch_span,
                metadata={
                    "strategy": dispatch_plan.strategy,
                    "selected": dispatch_plan.selected_agent_ids,
                    "confidence": dispatch_plan.confidence,
                    "success": dispatch_result.success,
                    "duration_ms": dispatch_result.metadata.get("duration_ms", 0),
                },
            )
            response_text = dispatch_result.final_output
            if response_text:
                response_text = strip_thinking(response_text)
            if intro_text:
                response_text = f"{intro_text}\n\n{response_text}" if response_text else intro_text
            session.add_message(
                "assistant",
                response_text,
                dispatch={
                    "strategy": dispatch_plan.strategy,
                    "selected": dispatch_plan.selected_agent_ids,
                    "confidence": dispatch_plan.confidence,
                    "success": dispatch_result.success,
                },
            )
            await self.sessions.save(session)
            outbound_sent = False
            if publish_response:
                outbound_sent = await stream_publisher.finalize(response_text)
            return _ProcessResult(response_text=response_text or "", outbound_sent=outbound_sent)

        for iteration in range(self._max_iterations):
            llm_span = self.tracer.start_span(trace_id, f"llm_{iteration}", "llm_call", "llm_call")
            response, route_decision = await self._chat_stream_with_routing(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                on_delta=stream_publisher.on_delta if publish_response else None,
                task_type=task_type,
                content=event.text,
            )
            self.tracer.end_span(
                llm_span,
                metadata={
                    "model": route_decision.model,
                    "provider": route_decision.provider_name,
                    "route_reason": route_decision.reason,
                    "finish": response.finish_reason,
                },
            )

            if self._telemetry and self._telemetry.available and response.usage:
                from echo_agent.observability.spans import start_llm_span, record_llm_usage, end_llm_span
                otel_span = start_llm_span(self._telemetry.get_tracer(), route_decision.model, route_decision.provider_name)
                record_llm_usage(otel_span, response.usage, route_decision.model)
                end_llm_span(otel_span)

            issues = self.inference.validate_response(response)
            if issues:
                logger.warning("Inference issues: {}", issues)

            if response.content:
                response_text = response.content

            if not response.has_tool_calls:
                if current_plan and not current_plan.is_complete:
                    current_plan.is_complete = True
                break

            if current_plan and iteration < len(current_plan.steps):
                current_plan.mark_step_complete(iteration, response.content or "")

            if response.content:
                await _emit_progress(response.content)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content}
            assistant_msg["tool_calls"] = [tool_call.to_openai_format() for tool_call in response.tool_calls]
            messages.append(assistant_msg)

            tool_call_fmts = [tool_call.to_openai_format() for tool_call in response.tool_calls]
            session.add_message("assistant", response.content or "", tool_calls=tool_call_fmts)

            for tool_index, tool_call in enumerate(response.tool_calls):
                tool_span = self.tracer.start_span(trace_id, f"tool_{iteration}_{tool_index}", f"tool:{tool_call.name}", "tool_call")

                await _emit_progress(f"Using tool: {tool_call.name}", tool_hint=True)

                denial = self._check_permission_and_approval(tool_call.name, tool_call.arguments, event.sender_id)
                if denial:
                    self.tracer.end_span(tool_span, metadata={"success": False, "denied": True})
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.name, "content": denial.text})
                    session.add_message("tool", denial.text, tool_call_id=tool_call.id, name=tool_call.name)
                    total_tool_calls += 1
                    continue

                tool_exec_ctx = ToolExecutionContext(
                    execution_id=uuid.uuid4().hex[:12],
                    trace_id=trace_id,
                    session_key=event.session_key,
                    user_id=event.sender_id,
                    attempt_index=0,
                    idempotency_key=build_idempotency_key(trace_id, tool_call.name, tool_index, tool_call.arguments),
                    credentials=self.credentials.get_for_tool(tool_call.name),
                )
                result = await self.tools.execute(tool_call.name, tool_call.arguments, tool_exec_ctx)
                result_text = result.text
                if len(result_text) > self._MAX_TOOL_RESULT_CHARS:
                    result_text = result_text[:self._MAX_TOOL_RESULT_CHARS] + "\n...(truncated)"

                self.tracer.end_span(tool_span, metadata={"success": result.success})

                if self._telemetry and self._telemetry.available:
                    from echo_agent.observability.spans import start_tool_span, end_tool_span
                    otel_tool = start_tool_span(self._telemetry.get_tracer(), tool_call.name)
                    end_tool_span(otel_tool, error=None if result.success else result.error)

                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.name, "content": result_text})
                session.add_message("tool", result_text, tool_call_id=tool_call.id, name=tool_call.name)

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
        if intro_text:
            response_text = f"{intro_text}\n\n{response_text}" if response_text else intro_text

        session.add_message("assistant", response_text)
        await self.sessions.save(session)

        if self.consolidator.should_consolidate(session.message_count, session.last_consolidated):
            asyncio.create_task(self._consolidate(session))

        if should_review_skills and total_tool_calls > 0:
            asyncio.create_task(self._background_skill_review(messages))

        if should_review_memory and total_tool_calls > 0:
            asyncio.create_task(self._background_memory_review(messages, event.session_key))

        outbound_sent = False
        if publish_response:
            outbound_sent = await stream_publisher.finalize(response_text)

        return _ProcessResult(response_text=response_text or "", outbound_sent=outbound_sent)

    def _check_permission_and_approval(
        self, tool_name: str, arguments: dict[str, Any], sender_id: str,
    ) -> ToolResult | None:
        """检查工具的权限和审批状态。

        Returns:
            ToolResult: 如果被拒绝或需要等待审批，返回拒绝结果；否则返回 None 表示通过。
        """
        if not self.permissions.check_tool(tool_name, user_id=sender_id):
            return ToolResult(success=False, error=f"Permission denied for tool '{tool_name}'")

        if not (self.inference.needs_confirmation(tool_name) or self.approval.needs_approval(tool_name)):
            return None

        approval_req = self.approval.request_approval(
            tool_name, tool_name=tool_name, params=arguments, user_id=sender_id,
        )
        if approval_req.status == ApprovalStatus.DENIED:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' denied by approval policy: {approval_req.reason}",
            )
        if approval_req.status == ApprovalStatus.PENDING:
            return ToolResult(
                success=False,
                error=(
                    f"Approval required before executing '{tool_name}'. "
                    f"Request id: {approval_req.id}. "
                    f"An admin can reply `/approve {approval_req.id}` or `/deny {approval_req.id} <reason>`."
                ),
                metadata={"approval_request_id": approval_req.id},
            )
        return None

    async def _execute_child_tool_call(
        self,
        *,
        agent_id: str,
        tool_call: ToolCallRequest,
        index: int,
        event: InboundEvent,
        trace_id: str,
        allowed_tools: set[str],
    ) -> str:
        if tool_call.name not in allowed_tools:
            return f"Error: Tool '{tool_call.name}' is not allowed for agent '{agent_id}'"

        denial = self._check_permission_and_approval(tool_call.name, tool_call.arguments, event.sender_id)
        if denial:
            return f"Error: {denial.error or denial.text}"

        ctx = ToolExecutionContext(
            execution_id=uuid.uuid4().hex[:12],
            trace_id=trace_id,
            session_key=event.session_key,
            user_id=event.sender_id,
            attempt_index=0,
            idempotency_key=build_idempotency_key(trace_id, f"{agent_id}:{tool_call.name}", index, tool_call.arguments),
            parent_execution_id=f"multi_agent:{agent_id}",
            credentials=self.credentials.get_for_tool(tool_call.name),
        )
        result = await self.tools.execute(tool_call.name, tool_call.arguments, ctx)
        result_text = result.text
        if len(result_text) > self._MAX_TOOL_RESULT_CHARS:
            result_text = result_text[:self._MAX_TOOL_RESULT_CHARS] + "\n...(truncated)"
        return result_text

    async def _chat_stream_with_routing(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_delta: Any = None,
        task_type: str = "",
        content: str = "",
    ) -> tuple[LLMResponse, RouteDecision]:
        """带路由和降级的 LLM 调用。

        按路由决策选择模型，失败时沿降级链尝试备选模型。
        支持流式输出和工具调用。
        """
        if not self.router:
            response = await self.provider.chat_stream_with_retry(
                messages=messages,
                tools=tools,
                model=self._default_model or None,
                on_delta=on_delta,
            )
            return response, RouteDecision(
                provider_name="default",
                model=self._default_model,
                reason="router disabled",
            )

        emitted = False

        async def routed_delta(delta: str) -> None:
            nonlocal emitted
            emitted = True
            if on_delta:
                maybe = on_delta(delta)
                if asyncio.iscoroutine(maybe):
                    await maybe

        last_response: LLMResponse | None = None
        last_decision: RouteDecision | None = None
        for provider_name, provider, decision in self.router.route_candidates(task_type, content):
            response = await provider.chat_stream_with_retry(
                messages=messages,
                tools=tools,
                model=decision.model,
                on_delta=routed_delta if on_delta else None,
                max_tokens=decision.max_tokens,
                temperature=decision.temperature,
            )
            last_response = response
            last_decision = decision
            if response.finish_reason != "error":
                self.router.mark_success(provider_name)
                return response, decision
            self.router.mark_failure(provider_name, response.content or "LLM error")
            logger.warning("LLM provider '{}' failed for model '{}': {}", provider_name, decision.model, response.content)
            if emitted:
                return response, decision

        if last_response and last_decision:
            return last_response, last_decision
        response = await self.provider.chat_stream_with_retry(
            messages=messages,
            tools=tools,
            model=self._default_model or None,
            on_delta=on_delta,
        )
        return response, RouteDecision(provider_name="default", model=self._default_model, reason="router empty")

    def _infer_task_type(self, text: str) -> str:
        lower = text.lower()
        if any(marker in lower for marker in ("代码", "报错", "bug", "函数", "class ", "def ", "typescript", "python")):
            return "code"
        if any(marker in lower for marker in ("搜索", "最新", "今天", "新闻", "价格", "search", "latest")):
            return "research"
        if any(marker in lower for marker in ("规划", "方案", "架构", "设计", "plan", "architecture")):
            return "planning"
        return "chat"

    async def _handle_approval_command(self, event: InboundEvent) -> str | None:
        text = event.text.strip()
        if not text.startswith("/"):
            return None
        parts = text.split(maxsplit=2)
        command = parts[0].lower()
        if command not in {"/approvals", "/approve", "/deny"}:
            return None

        if command == "/approvals":
            pending = self.approval.get_pending()
            visible = [req for req in pending if self._can_decide_approval(event.sender_id, req)]
            if not visible:
                return "No pending approval requests."
            lines = ["Pending approval requests:"]
            for req in visible:
                lines.append(f"- {req.id}: {req.tool_name or req.action} requested by {req.user_id}")
            return "\n".join(lines)

        if len(parts) < 2:
            return f"Usage: `{command} <request_id>`"
        request_id = parts[1]
        req = self.approval.get(request_id)
        if not req:
            return f"Approval request not found: {request_id}"
        if not self._can_decide_approval(event.sender_id, req):
            return "You are not allowed to decide this approval request."

        if command == "/approve":
            ok = self.approval.approve(request_id, decided_by=event.sender_id)
            return f"Approval request {request_id} approved." if ok else f"Approval request not found: {request_id}"

        reason = parts[2] if len(parts) >= 3 else ""
        ok = self.approval.deny(request_id, reason=reason, decided_by=event.sender_id)
        return f"Approval request {request_id} denied." if ok else f"Approval request not found: {request_id}"

    def _can_decide_approval(self, user_id: str, request: Any) -> bool:
        if self.permissions.is_admin(user_id):
            return True
        if not self.config.permissions.admin_users:
            return not request.user_id or request.user_id == user_id
        return False

    def _should_introduce(self, session: Session) -> bool:
        if not self.config.session.introduction_enabled:
            return False
        return not any(msg.get("role") == "assistant" for msg in session.messages)

    def _build_introduction(self, event: InboundEvent) -> str:
        template = self.config.session.introduction_template.strip()
        if not template:
            if event.channel in {"wechat", "wecom", "weixin"}:
                template = "你好，我是 {agent_name}，很高兴为你服务。"
            else:
                template = "Hello, I'm {agent_name}. How can I help?"

        values = {
            "agent_name": self.context.agent_name,
            "channel": event.channel,
            "chat_id": event.chat_id,
            "session_key": event.session_key,
        }
        try:
            return template.format(**values).strip()
        except Exception:
            logger.warning("Invalid session introduction template, using raw text")
            return template

    def _should_stream_channel(self, channel: str) -> bool:
        if channel.startswith("gateway:"):
            return False
        return channel in set(self.config.channels.stream_channels)

    async def _background_skill_review(self, messages: list[dict[str, Any]]) -> None:
        try:
            from echo_agent.skills.reviewer import SkillReviewer
            reviewer = SkillReviewer(
                provider=self.provider,
                store=self.skill_store,
                model=self._default_model,
            )
            actions = await reviewer.review(messages)
            if actions:
                logger.info("Background skill review: {}", "; ".join(actions))
        except Exception as e:
            logger.warning("Background skill review failed: {}", e)

    async def _background_memory_review(self, messages: list[dict[str, Any]], session_key: str) -> None:
        try:
            from echo_agent.memory.reviewer import MemoryReviewer
            reviewer = MemoryReviewer(
                provider=self.provider,
                store=self.memory,
                model=self._default_model,
                session_key=session_key,
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
                await self.sessions.save(session)
            if self.config.memory.sleep_consolidation:
                try:
                    stats = await self.consolidator.sleep_consolidate(session.key, chunk)
                    if any(v > 0 for v in stats.values()):
                        logger.info("Sleep consolidation for {}: {}", session.key, stats)
                except Exception as e:
                    logger.warning("Sleep consolidation failed: {}", e)
        except Exception as e:
            logger.error("Consolidation failed for {}: {}", session.key, e)

    async def process_direct(self, content: str, session_key: str = "cli:direct") -> str:
        """Process a message directly (for CLI or testing)."""
        event = InboundEvent.text_message(
            channel="cli", sender_id="user", chat_id="direct", text=content,
            session_key_override=session_key,
        )
        result = await self._process_event(event, uuid.uuid4().hex[:12], publish_response=False)
        return result.response_text or ""
