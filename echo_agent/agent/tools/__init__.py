"""Tool discovery and registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from echo_agent.agent.tools.base import Tool
from echo_agent.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from echo_agent.agent.tools.message import MessageTool
from echo_agent.agent.tools.web import WebFetchTool, WebSearchTool
from echo_agent.bus.queue import MessageBus
from echo_agent.config.schema import Config
from echo_agent.models.provider import LLMProvider


def discover_tools(
    config: Config,
    workspace: Path,
    bus: MessageBus,
    provider: LLMProvider | None = None,
    scheduler: Any = None,
    session_manager: Any = None,
    skill_store: Any = None,
    memory_store: Any = None,
    task_manager: Any = None,
    workflow_engine: Any = None,
    knowledge_index: Any = None,
) -> list[Tool]:
    ws = str(workspace)
    restrict = config.tools.restrict_to_workspace
    tools: list[Tool] = []
    executor = None

    if config.tools.exec.enabled:
        from echo_agent.agent.executors.factory import create_executor
        from echo_agent.agent.tools.shell import ShellTool
        executor = create_executor(config.execution, workspace)
        tools.append(ShellTool(
            ws,
            allowed=config.tools.exec.allowed_commands,
            blocked=config.tools.exec.blocked_commands,
            max_output=config.tools.exec.max_output_chars,
            executor=executor,
        ))
    tools.append(ReadFileTool(ws, restrict))
    tools.append(WriteFileTool(ws, restrict))
    tools.append(EditFileTool(ws, restrict))
    tools.append(ListDirTool(ws, restrict))
    if config.tools.web.enabled:
        tools.append(WebFetchTool(proxy=config.tools.web.proxy))
        if config.tools.web.search_api_key or config.tools.web.search_provider == "searxng":
            tools.append(WebSearchTool(
                api_key=config.tools.web.search_api_key,
                provider=config.tools.web.search_provider,
                api_base=config.tools.web.search_api_base,
                proxy=config.tools.web.proxy,
                timeout_seconds=config.tools.web.timeout_seconds,
            ))
    tools.append(MessageTool(publish_fn=bus.publish_outbound))

    from echo_agent.agent.tools.search import SearchFilesTool
    tools.append(SearchFilesTool(ws, restrict))

    from echo_agent.agent.tools.patch import PatchTool
    tools.append(PatchTool(ws, restrict))

    from echo_agent.agent.tools.todo import TodoTool
    tools.append(TodoTool(store_dir=workspace / "data" / "todos"))

    if task_manager:
        from echo_agent.agent.tools.task import TaskTool
        tools.append(TaskTool(manager=task_manager))

    if workflow_engine:
        from echo_agent.agent.tools.workflow import WorkflowTool
        tools.append(WorkflowTool(engine=workflow_engine))

    from echo_agent.agent.tools.clarify import ClarifyTool
    tools.append(ClarifyTool(bus=bus))

    from echo_agent.agent.tools.notify import NotifyTool
    tools.append(NotifyTool(bus=bus))

    if config.tools.exec.enabled:
        from echo_agent.agent.tools.code_exec import CodeExecTool
        tools.append(CodeExecTool(
            ws,
            executor=executor,
            allowed_languages=config.tools.code_exec.allowed_languages,
            max_output=config.tools.exec.max_output_chars,
            timeout_seconds=config.tools.code_exec.timeout_seconds,
        ))

        from echo_agent.agent.tools.process import ProcessTool
        tools.append(ProcessTool(ws))

    if provider:
        from echo_agent.agent.tools.delegate import DelegateTool, SpawnTool
        tools.append(DelegateTool(provider=provider))
        tools.append(SpawnTool(provider=provider, bus=bus))

        from echo_agent.agent.tools.vision import VisionTool
        tools.append(VisionTool(provider=provider, workspace=ws))

    _try_register_image_gen(tools, config)
    _try_register_tts(tools, config, ws)

    if session_manager:
        from echo_agent.agent.tools.session_search import SessionSearchTool
        tools.append(SessionSearchTool(session_manager=session_manager))

    if scheduler:
        from echo_agent.agent.tools.cronjob import CronjobTool
        tools.append(CronjobTool(scheduler=scheduler))

    if skill_store:
        from echo_agent.agent.tools.skills import SkillsListTool, SkillViewTool, SkillManageTool
        from echo_agent.agent.tools.skill_install import SkillInstallTool
        tools.append(SkillsListTool(store=skill_store))
        tools.append(SkillViewTool(store=skill_store))
        tools.append(SkillManageTool(store=skill_store))
        tools.append(SkillInstallTool(store=skill_store))

    if memory_store:
        from echo_agent.agent.tools.memory import MemoryTool
        tools.append(MemoryTool(store=memory_store))

    if knowledge_index:
        from echo_agent.agent.tools.knowledge import KnowledgeIndexTool, KnowledgeSearchTool
        tools.append(KnowledgeSearchTool(index=knowledge_index, default_limit=config.knowledge.max_results))
        tools.append(KnowledgeIndexTool(index=knowledge_index))

    logger.info("Discovered {} tools", len(tools))
    return tools


def _try_register_image_gen(tools: list[Tool], config: Config) -> None:
    ig = getattr(config.tools, "image_gen", None)
    if not ig or not getattr(ig, "api_key", ""):
        return
    from echo_agent.agent.tools.image_gen import ImageGenTool
    tools.append(ImageGenTool(
        api_key=ig.api_key,
        api_base=getattr(ig, "api_base", ""),
        model=getattr(ig, "model", "dall-e-3"),
    ))


def _try_register_tts(tools: list[Tool], config: Config, ws: str) -> None:
    from echo_agent.agent.tools.tts import TTSTool
    tts_cfg = getattr(config.tools, "tts", None)
    openai_key = getattr(tts_cfg, "openai_api_key", "") if tts_cfg else ""
    tools.append(TTSTool(workspace=ws, openai_api_key=openai_key))
