from __future__ import annotations

import pytest

from echo_agent.agent.multi_agent import AgentRegistry, IntentRouter, MultiAgentRuntime
from echo_agent.agent.loop import AgentLoop
from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult
from echo_agent.agent.tools.registry import ToolRegistry
from echo_agent.bus.queue import MessageBus
from echo_agent.config.schema import AgentProfileConfig, MultiAgentConfig, Config, SessionConfig
from echo_agent.models.provider import LLMProvider, LLMResponse, ToolCallRequest


class EchoProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None, model=None, tool_choice=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools or [], "model": model})
        agent_id = ""
        system = messages[0]["content"] if messages else ""
        for line in system.splitlines():
            if line.startswith("Agent ID:"):
                agent_id = line.split(":", 1)[1].strip()
        return LLMResponse(content=f"{agent_id or 'supervisor'} handled")

    def get_default_model(self) -> str:
        return "fake-model"


class ToolCallingProvider(EchoProvider):
    async def chat(self, messages, tools=None, model=None, tool_choice=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools or [], "model": model})
        if not any(msg.get("role") == "tool" for msg in messages):
            return LLMResponse(
                tool_calls=[ToolCallRequest(id="tc1", name="exec", arguments={"command": "date"})],
                finish_reason="tool_calls",
            )
        return LLMResponse(content="tool done")


class DummyTool(Tool):
    name = "exec"
    description = "Dummy exec"
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    async def execute(self, params: dict, ctx: ToolExecutionContext | None = None) -> ToolResult:
        return ToolResult(output=f"ran {params['command']}")


def _registry() -> AgentRegistry:
    cfg = MultiAgentConfig(
        enabled=True,
        agents=[
            AgentProfileConfig(id="general", name="General", task_types=["chat"], capabilities=["chat"]),
            AgentProfileConfig(
                id="coder",
                name="Coder",
                task_types=["code"],
                capabilities=["code", "tests"],
                keywords=["代码", "bug", "pytest"],
                tools_allow=["read_file", "exec"],
                priority=30,
            ),
            AgentProfileConfig(
                id="researcher",
                name="Researcher",
                task_types=["research"],
                capabilities=["research", "web"],
                keywords=["搜索", "最新", "比较"],
                tools_allow=["web_search"],
                priority=20,
            ),
        ],
    )
    return AgentRegistry.from_config(cfg)


def test_intent_router_selects_coder_for_code_request() -> None:
    router = IntentRouter(_registry(), route_threshold=0.45)

    plan = router.build_plan("帮我修复这个 Python bug 并跑 pytest", task_type="chat", available_tools=["exec"])

    assert plan.should_dispatch
    assert plan.primary_agent_id == "coder"
    assert plan.task_type == "code"
    assert plan.confidence >= 0.45


def test_intent_router_parallel_when_request_has_two_strong_domains() -> None:
    router = IntentRouter(_registry(), route_threshold=0.45, multi_threshold=0.5, max_parallel_agents=2)

    plan = router.build_plan("同时搜索最新资料并比较代码实现方案", task_type="chat", available_tools=["exec", "web_search"])

    assert plan.strategy == "parallel"
    assert "coder" in plan.selected_agent_ids
    assert "researcher" in plan.selected_agent_ids


@pytest.mark.asyncio
async def test_multi_agent_runtime_filters_tools_and_runs_agent(tmp_path) -> None:
    registry = _registry()
    router = IntentRouter(registry)
    provider = EchoProvider()
    tools = ToolRegistry()
    tools.register(DummyTool())
    runtime = MultiAgentRuntime(
        registry=registry,
        router=router,
        provider=provider,
        model_router=None,
        tools=tools,
        audit_path=tmp_path / "audit.jsonl",
    )
    plan = router.build_plan("修复代码 bug", task_type="code", available_tools=tools.tool_names)

    async def executor(agent_id, tool_call, index, messages, allowed_tools):
        return "unused"

    result = await runtime.dispatch(
        query="修复代码 bug",
        plan=plan,
        base_messages=[{"role": "system", "content": "base"}, {"role": "user", "content": "修复代码 bug"}],
        tool_executor=executor,
    )

    assert result.final_output == "coder handled"
    assert provider.calls[0]["tools"][0]["function"]["name"] == "exec"
    assert (tmp_path / "audit.jsonl").exists()


@pytest.mark.asyncio
async def test_multi_agent_runtime_executes_tool_callback(tmp_path) -> None:
    registry = _registry()
    router = IntentRouter(registry)
    provider = ToolCallingProvider()
    tools = ToolRegistry()
    tools.register(DummyTool())
    runtime = MultiAgentRuntime(
        registry=registry,
        router=router,
        provider=provider,
        model_router=None,
        tools=tools,
        audit_path=tmp_path / "audit.jsonl",
    )
    plan = router.build_plan("运行诊断代码", task_type="code", available_tools=tools.tool_names)
    seen = []

    async def executor(agent_id, tool_call, index, messages, allowed_tools):
        seen.append((agent_id, tool_call.name, set(allowed_tools)))
        return "ran date"

    result = await runtime.dispatch(
        query="运行诊断代码",
        plan=plan,
        base_messages=[{"role": "system", "content": "base"}, {"role": "user", "content": "运行诊断代码"}],
        tool_executor=executor,
    )

    assert result.results[0].tool_calls == 1
    assert result.final_output == "tool done"
    assert seen == [("coder", "exec", {"exec"})]


@pytest.mark.asyncio
async def test_agent_loop_auto_dispatches_to_specialist(tmp_path) -> None:
    cfg = Config(
        workspace=str(tmp_path),
        session=SessionConfig(introduction_enabled=False),
        multi_agent=MultiAgentConfig(
            enabled=True,
            synthesize_results=False,
            agents=[
                AgentProfileConfig(id="general", name="General", task_types=["chat"], capabilities=["chat"]),
                AgentProfileConfig(
                    id="coder",
                    name="Coder",
                    task_types=["code"],
                    capabilities=["code"],
                    keywords=["Python", "bug"],
                    tools_allow=[],
                    priority=30,
                ),
            ],
        ),
    )
    agent = AgentLoop(
        bus=MessageBus(),
        config=cfg,
        provider=EchoProvider(),
        workspace=tmp_path,
    )

    response = await agent.process_direct("请修复这个 Python bug", session_key="test:auto-dispatch")

    assert response == "coder handled"
