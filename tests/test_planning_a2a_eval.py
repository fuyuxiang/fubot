"""Comprehensive tests for planning system, A2A protocol, and evaluation framework."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest

from echo_agent.models.provider import LLMResponse, ToolCallRequest

# ── Planning imports ──
from echo_agent.agent.planning.models import (
    Plan, PlanStep, StepAction, Feedback, StrategyType, StepStatus,
)
from echo_agent.agent.planning.tree import SearchNode, SearchTree
from echo_agent.agent.planning.strategies import (
    ReactStrategy, PlanExecuteStrategy, TreeOfThoughtStrategy, LATSStrategy,
)
from echo_agent.agent.planning.reflection import ReflectionModule
from echo_agent.agent.planning.planner import AgentPlanner

# ── A2A imports ──
from echo_agent.a2a.models import AgentCard, A2ATask, A2AMessage, TaskState, Artifact
from echo_agent.a2a.protocol import A2AProtocol
from echo_agent.a2a.client import A2AClient

# ── Evaluation imports ──
from echo_agent.evaluation.dataset import EvalCase, EvalDataset
from echo_agent.evaluation.metrics import (
    exact_match, contains_all, tool_usage_correctness,
    iteration_efficiency, response_quality, MetricResult,
)
from echo_agent.evaluation.runner import CaseResult, EvalReport
from echo_agent.evaluation.reporter import EvalReporter


# ── Mock LLM helper ──

async def mock_llm_call(**kwargs):
    tools = kwargs.get("tools", [])
    if tools:
        name = tools[0]["function"]["name"]
        if name == "create_plan":
            return LLMResponse(tool_calls=[ToolCallRequest(id="1", name="create_plan", arguments={
                "goal": "test goal", "steps": [{"description": "step 1"}, {"description": "step 2"}]
            })])
        if name == "next_action":
            return LLMResponse(tool_calls=[ToolCallRequest(id="1", name="next_action", arguments={
                "action": "respond", "reasoning": "done"
            })])
        if name == "critique":
            return LLMResponse(tool_calls=[ToolCallRequest(id="1", name="critique", arguments={
                "score": 0.8, "should_replan": False, "critique": "good"
            })])
        if name == "predict_topics":
            return LLMResponse(tool_calls=[ToolCallRequest(id="1", name="predict_topics", arguments={
                "topics": ["python", "testing"]
            })])
        return LLMResponse(tool_calls=[ToolCallRequest(id="1", name=name, arguments={})])
    return LLMResponse(content="mock response")


# ═══════════════════════════════════════════════════════════════════
# 1. Planning Models
# ═══════════════════════════════════════════════════════════════════

class TestPlanModels:
    def test_plan_next_step_returns_first_pending(self):
        plan = Plan(strategy=StrategyType.REACT, steps=[
            PlanStep(index=0, description="a", status=StepStatus.COMPLETED),
            PlanStep(index=1, description="b", status=StepStatus.PENDING),
        ])
        assert plan.next_step().index == 1

    def test_plan_next_step_none_when_all_done(self):
        plan = Plan(strategy=StrategyType.REACT, steps=[
            PlanStep(index=0, description="a", status=StepStatus.COMPLETED),
        ])
        assert plan.next_step() is None
    def test_mark_step_complete(self):
        plan = Plan(strategy=StrategyType.REACT, steps=[
            PlanStep(index=0, description="a"),
            PlanStep(index=1, description="b"),
        ])
        plan.mark_step_complete(0, "done")
        assert plan.steps[0].status == StepStatus.COMPLETED
        assert plan.steps[0].result == "done"
        assert plan.current_step == 1
        assert not plan.is_complete

    def test_mark_all_steps_complete_sets_is_complete(self):
        plan = Plan(strategy=StrategyType.REACT, steps=[
            PlanStep(index=0, description="a"),
        ])
        plan.mark_step_complete(0, "ok")
        assert plan.is_complete

    def test_mark_step_failed(self):
        plan = Plan(strategy=StrategyType.REACT, steps=[PlanStep(index=0, description="a")])
        plan.mark_step_failed(0, "error!")
        assert plan.steps[0].status == StepStatus.FAILED
        assert plan.steps[0].result == "error!"

    def test_to_prompt_contains_goal_and_steps(self):
        plan = Plan(strategy=StrategyType.REACT, goal="my goal", steps=[
            PlanStep(index=0, description="step zero", status=StepStatus.COMPLETED, result="ok"),
            PlanStep(index=1, description="step one", status=StepStatus.PENDING),
        ])
        prompt = plan.to_prompt()
        assert "my goal" in prompt
        assert "step zero" in prompt
        assert "step one" in prompt
        assert "ok" in prompt

    def test_step_action_defaults(self):
        action = StepAction(action="respond", reasoning="test")
        assert action.tool_name == ""
        assert action.tool_args == {}

    def test_feedback_defaults(self):
        fb = Feedback()
        assert fb.should_replan is False
        assert fb.score == 0.0
        assert fb.suggestions == []


# ═══════════════════════════════════════════════════════════════════
# 2. Search Tree
# ═══════════════════════════════════════════════════════════════════

class TestSearchTree:
    def test_ucb1_unvisited_returns_inf(self):
        node = SearchNode(id="n", state="s", visits=0)
        assert node.ucb1() == float("inf")

    def test_ucb1_visited_node(self):
        parent = SearchNode(id="p", state="s", visits=10, score=5.0)
        child = SearchNode(id="c", state="s", visits=3, score=2.0, parent=parent)
        ucb = child.ucb1()
        exploit = 2.0 / 3
        explore = 1.41 * math.sqrt(math.log(10) / 3)
        assert abs(ucb - (exploit + explore)) < 1e-6

    def test_best_child_picks_highest_ucb(self):
        parent = SearchNode(id="p", state="s", visits=10, score=5.0)
        c1 = SearchNode(id="c1", state="s", visits=5, score=1.0, parent=parent)
        c2 = SearchNode(id="c2", state="s", visits=1, score=0.5, parent=parent)
        parent.children = [c1, c2]
        assert parent.best_child().id == "c2"  # c2 has fewer visits -> higher UCB

    def test_best_child_empty(self):
        node = SearchNode(id="n", state="s")
        assert node.best_child() is None

    def test_backpropagate(self):
        root = SearchNode(id="r", state="s")
        child = SearchNode(id="c", state="s", parent=root)
        root.children.append(child)
        child.backpropagate(1.0)
        assert child.visits == 1
        assert child.score == 1.0
        assert root.visits == 1
        assert root.score == 1.0
    def test_search_tree_select_returns_leaf(self):
        tree = SearchTree("root")
        child = tree.expand(tree.root, "act1", "s1", score=1.0)
        child.visits = 1
        tree.root.visits = 1
        selected = tree.select()
        assert selected.id == child.id  # leaf node

    def test_search_tree_expand_increments_size(self):
        tree = SearchTree("root")
        assert tree.size == 1
        tree.expand(tree.root, "a", "s")
        assert tree.size == 2

    def test_search_tree_best_path(self):
        tree = SearchTree("root")
        c1 = tree.expand(tree.root, "a1", "s1", score=1.0)
        c1.visits = 1
        c2 = tree.expand(tree.root, "a2", "s2", score=5.0)
        c2.visits = 1
        path = tree.best_path()
        assert len(path) == 1
        assert path[0].id == c2.id


# ═══════════════════════════════════════════════════════════════════
# 3. Planning Strategies
# ═══════════════════════════════════════════════════════════════════

class TestStrategies:
    @pytest.mark.asyncio
    async def test_react_plan_single_step(self):
        strategy = ReactStrategy(mock_llm_call)
        plan = await strategy.plan("hello", [], "")
        assert plan.strategy == StrategyType.REACT
        assert len(plan.steps) == 1

    @pytest.mark.asyncio
    async def test_react_step_returns_action(self):
        strategy = ReactStrategy(mock_llm_call)
        plan = Plan(strategy=StrategyType.REACT, goal="test", steps=[PlanStep(index=0, description="do")])
        action = await strategy.step(plan, 0, "")
        assert action.action == "respond"

    @pytest.mark.asyncio
    async def test_plan_execute_creates_multi_step(self):
        strategy = PlanExecuteStrategy(mock_llm_call)
        plan = await strategy.plan("complex task", [{"function": {"name": "tool1"}}], "ctx")
        assert plan.strategy == StrategyType.PLAN_EXECUTE
        assert len(plan.steps) == 2
        assert plan.steps[0].description == "step 1"

    @pytest.mark.asyncio
    async def test_plan_execute_step_returns_execute_tool(self):
        strategy = PlanExecuteStrategy(mock_llm_call)
        plan = Plan(strategy=StrategyType.PLAN_EXECUTE, steps=[
            PlanStep(index=0, description="do it", tool_hint="my_tool"),
        ])
        action = await strategy.step(plan, 0, "")
        assert action.action == "execute_tool"
        assert action.tool_name == "my_tool"

    @pytest.mark.asyncio
    async def test_tree_of_thought_picks_shortest(self):
        strategy = TreeOfThoughtStrategy(mock_llm_call)
        plan = await strategy.plan("task", [], "")
        assert plan.strategy == StrategyType.TREE_OF_THOUGHT
        assert len(plan.steps) == 2  # mock always returns 2-step plan

    @pytest.mark.asyncio
    async def test_lats_step_backtrack_on_failure(self):
        strategy = LATSStrategy(mock_llm_call)
        plan = Plan(strategy=StrategyType.LATS, steps=[
            PlanStep(index=0, description="a", status=StepStatus.FAILED, result="boom"),
            PlanStep(index=1, description="b"),
        ])
        action = await strategy.step(plan, 1, "")
        assert action.action == "replan"
        assert "boom" in action.reasoning

# ═══════════════════════════════════════════════════════════════════
# 4. Reflection
# ═══════════════════════════════════════════════════════════════════

class TestReflection:
    @pytest.mark.asyncio
    async def test_critique_returns_feedback(self):
        module = ReflectionModule(mock_llm_call)
        plan = Plan(strategy=StrategyType.REACT, goal="g", steps=[PlanStep(index=0, description="s")])
        fb = await module.critique(plan, ["result1"])
        assert fb.score == 0.8
        assert fb.should_replan is False
        assert fb.critique == "good"

    @pytest.mark.asyncio
    async def test_critique_fallback_on_error(self):
        async def failing_llm(**kw):
            raise RuntimeError("boom")
        module = ReflectionModule(failing_llm)
        plan = Plan(strategy=StrategyType.REACT, goal="g", steps=[])
        fb = await module.critique(plan, [])
        assert fb.score == 0.5
        assert "unavailable" in fb.critique.lower()


# ═══════════════════════════════════════════════════════════════════
# 5. AgentPlanner
# ═══════════════════════════════════════════════════════════════════

class TestAgentPlanner:
    def test_select_strategy_simple_returns_react(self):
        planner = AgentPlanner(mock_llm_call)
        assert planner.select_strategy("hi", tool_count=1, token_estimate=100) == StrategyType.REACT

    def test_select_strategy_complex_returns_plan_execute(self):
        planner = AgentPlanner(mock_llm_call)
        assert planner.select_strategy("complex", tool_count=6, token_estimate=3000) == StrategyType.PLAN_EXECUTE

    def test_select_strategy_forced(self):
        planner = AgentPlanner(mock_llm_call, default_strategy="lats")
        assert planner.select_strategy("x", 0, 0) == StrategyType.LATS

    @pytest.mark.asyncio
    async def test_create_plan_with_mock(self):
        planner = AgentPlanner(mock_llm_call)
        plan = await planner.create_plan("hello", [], token_estimate=100)
        assert plan.strategy == StrategyType.REACT

    @pytest.mark.asyncio
    async def test_reflect_disabled(self):
        planner = AgentPlanner(mock_llm_call, reflection_enabled=False)
        fb = await planner.reflect(Plan(strategy=StrategyType.REACT, steps=[]), [])
        assert fb.score == 0.5


# ═══════════════════════════════════════════════════════════════════
# 6. A2A Models
# ═══════════════════════════════════════════════════════════════════

class TestA2AModels:
    def test_agent_card_serialization(self):
        card = AgentCard(name="test", skills=["code"])
        d = card.to_dict()
        assert d["name"] == "test"
        assert d["capabilities"]["streaming"] is True
        assert len(d["skills"]) == 1
        assert d["skills"][0]["id"] == "code"

    def test_a2a_task_lifecycle(self):
        task = A2ATask()
        assert task.state == TaskState.SUBMITTED
        task.state = TaskState.WORKING
        assert task.state == TaskState.WORKING
        d = task.to_dict()
        assert d["state"] == "working"

    def test_a2a_task_from_dict(self):
        task = A2ATask.from_dict({"id": "abc", "state": "completed", "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
        ]})
        assert task.id == "abc"
        assert task.state == TaskState.COMPLETED
        assert len(task.messages) == 1

    def test_a2a_message_text_helper(self):
        msg = A2AMessage.text("user", "hello world")
        assert msg.role == "user"
        assert msg.text_content == "hello world"
        d = msg.to_dict()
        assert d["parts"][0]["type"] == "text"

    def test_task_state_transitions(self):
        assert TaskState.SUBMITTED.value == "submitted"
        assert TaskState.CANCELED.value == "canceled"
        assert TaskState("failed") == TaskState.FAILED

    def test_artifact_to_dict(self):
        a = Artifact(name="out.txt", data="hello")
        d = a.to_dict()
        assert d["name"] == "out.txt"
        assert d["contentType"] == "text/plain"

# ═══════════════════════════════════════════════════════════════════
# 7. A2A Protocol
# ═══════════════════════════════════════════════════════════════════

class TestA2AProtocol:
    @staticmethod
    async def _echo_process(task: A2ATask) -> A2ATask:
        task.state = TaskState.COMPLETED
        task.messages.append(A2AMessage.text("agent", "echo"))
        return task

    @pytest.mark.asyncio
    async def test_handle_send_new_task(self):
        proto = A2AProtocol(self._echo_process)
        resp = await proto.handle({
            "jsonrpc": "2.0", "id": "1", "method": "tasks/send",
            "params": {"message": {"role": "user", "parts": [{"type": "text", "text": "hi"}]}},
        })
        assert "result" in resp
        assert resp["result"]["state"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_send_existing_task(self):
        proto = A2AProtocol(self._echo_process)
        r1 = await proto.handle({
            "jsonrpc": "2.0", "id": "1", "method": "tasks/send",
            "params": {"id": "t1", "message": {"role": "user", "parts": [{"type": "text", "text": "first"}]}},
        })
        r2 = await proto.handle({
            "jsonrpc": "2.0", "id": "2", "method": "tasks/send",
            "params": {"id": "t1", "message": {"role": "user", "parts": [{"type": "text", "text": "second"}]}},
        })
        assert r2["result"]["id"] == "t1"
        assert len(r2["result"]["messages"]) >= 3  # first + agent echo + second + agent echo

    @pytest.mark.asyncio
    async def test_handle_get(self):
        proto = A2AProtocol(self._echo_process)
        await proto.handle({
            "jsonrpc": "2.0", "id": "1", "method": "tasks/send",
            "params": {"id": "t2", "message": {"role": "user", "parts": [{"type": "text", "text": "x"}]}},
        })
        resp = await proto.handle({"jsonrpc": "2.0", "id": "2", "method": "tasks/get", "params": {"id": "t2"}})
        assert resp["result"]["id"] == "t2"

    @pytest.mark.asyncio
    async def test_handle_cancel(self):
        proto = A2AProtocol(self._echo_process)
        await proto.handle({
            "jsonrpc": "2.0", "id": "1", "method": "tasks/send",
            "params": {"id": "t3", "message": {"role": "user", "parts": [{"type": "text", "text": "x"}]}},
        })
        resp = await proto.handle({"jsonrpc": "2.0", "id": "2", "method": "tasks/cancel", "params": {"id": "t3"}})
        assert resp["result"]["state"] == "canceled"

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self):
        proto = A2AProtocol(self._echo_process)
        resp = await proto.handle({"jsonrpc": "2.0", "id": "1", "method": "nope", "params": {}})
        assert "error" in resp
        assert resp["error"]["code"] == -32601


# ═══════════════════════════════════════════════════════════════════
# 8. A2A Client
# ═══════════════════════════════════════════════════════════════════

class TestA2AClient:
    def test_client_instantiation(self):
        client = A2AClient(timeout=30)
        assert client._timeout == 30

    def test_client_default_timeout(self):
        client = A2AClient()
        assert client._timeout == 60

# ═══════════════════════════════════════════════════════════════════
# 9. Evaluation Dataset
# ═══════════════════════════════════════════════════════════════════

class TestEvalDataset:
    def test_eval_case_from_dict(self):
        case = EvalCase.from_dict({
            "id": "c1", "input": "hello", "expected_tools": ["search"],
            "tags": ["basic"], "max_iterations": 5,
        })
        assert case.id == "c1"
        assert case.expected_tools == ["search"]
        assert case.max_iterations == 5

    def test_eval_dataset_from_yaml(self, tmp_path):
        yaml_content = """
cases:
  - id: t1
    input: "test input"
    tags: [smoke]
  - id: t2
    input: "another"
    tags: [regression]
"""
        p = tmp_path / "eval.yaml"
        p.write_text(yaml_content)
        ds = EvalDataset.from_yaml(p)
        assert len(ds) == 2
        assert ds.cases[0].id == "t1"

    def test_filter_by_tag(self, tmp_path):
        yaml_content = """
cases:
  - id: t1
    input: "a"
    tags: [smoke]
  - id: t2
    input: "b"
    tags: [regression]
  - id: t3
    input: "c"
    tags: [smoke, regression]
"""
        p = tmp_path / "eval.yaml"
        p.write_text(yaml_content)
        ds = EvalDataset.from_yaml(p)
        smoke = ds.filter_by_tag("smoke")
        assert len(smoke) == 2
        assert {c.id for c in smoke} == {"t1", "t3"}


# ═══════════════════════════════════════════════════════════════════
# 10. Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_exact_match_pass(self):
        r = exact_match("Hello World", "  hello world  ")
        assert r.passed is True
        assert r.score == 1.0

    def test_exact_match_fail(self):
        r = exact_match("foo", "bar")
        assert r.passed is False
        assert r.score == 0.0

    def test_contains_all_full(self):
        r = contains_all(["python", "test"], "This is a Python test case")
        assert r.passed is True
        assert r.score == 1.0

    def test_contains_all_partial(self):
        r = contains_all(["python", "java"], "I love python")
        assert r.passed is False
        assert r.score == 0.5

    def test_contains_all_empty(self):
        r = contains_all([], "anything")
        assert r.passed is True

    def test_tool_usage_perfect(self):
        r = tool_usage_correctness(["search", "calc"], ["calc", "search"])
        assert r.passed is True
        assert r.score == 1.0

    def test_tool_usage_partial(self):
        r = tool_usage_correctness(["search", "calc"], ["search", "other"])
        assert r.passed is False
        assert 0.0 < r.score < 1.0

    def test_tool_usage_missing(self):
        r = tool_usage_correctness(["search"], [])
        assert r.passed is False
        assert r.score == 0.0

    def test_iteration_efficiency_within(self):
        r = iteration_efficiency(3, 10)
        assert r.passed is True
        assert r.score > 0.0

    def test_iteration_efficiency_over(self):
        r = iteration_efficiency(15, 10)
        assert r.passed is False

    def test_response_quality(self):
        r = response_quality("the quick brown fox", "the quick brown fox jumps")
        assert r.passed is True
        assert r.score > 0.5

# ═══════════════════════════════════════════════════════════════════
# 11. Reporter
# ═══════════════════════════════════════════════════════════════════

def _sample_report() -> EvalReport:
    r1 = CaseResult(case_id="c1", passed=True, response="ok", duration_ms=100,
                     metrics=[MetricResult(name="m", score=1.0, passed=True)])
    r2 = CaseResult(case_id="c2", passed=False, response="bad", duration_ms=200, error="timeout",
                     metrics=[MetricResult(name="m", score=0.3, passed=False)])
    return EvalReport(results=[r1, r2], total_cases=2, passed_cases=1, duration_ms=300)


class TestReporter:
    def test_to_json_valid(self):
        reporter = EvalReporter()
        j = reporter.to_json(_sample_report())
        data = json.loads(j)
        assert "summary" in data
        assert len(data["results"]) == 2

    def test_to_table_contains_pass_fail(self):
        reporter = EvalReporter()
        table = reporter.to_table(_sample_report())
        assert "PASS" in table
        assert "FAIL" in table


# ═══════════════════════════════════════════════════════════════════
# 12. Runner data classes
# ═══════════════════════════════════════════════════════════════════

class TestRunnerModels:
    def test_case_result_score(self):
        cr = CaseResult(metrics=[
            MetricResult(name="a", score=1.0, passed=True),
            MetricResult(name="b", score=0.5, passed=False),
        ])
        assert abs(cr.score - 0.75) < 1e-6

    def test_case_result_score_empty(self):
        cr = CaseResult()
        assert cr.score == 0.0

    def test_eval_report_pass_rate(self):
        report = EvalReport(total_cases=4, passed_cases=3)
        assert report.pass_rate == 0.75

    def test_eval_report_avg_score(self):
        r1 = CaseResult(metrics=[MetricResult(name="m", score=1.0, passed=True)])
        r2 = CaseResult(metrics=[MetricResult(name="m", score=0.5, passed=False)])
        report = EvalReport(results=[r1, r2], total_cases=2, passed_cases=1)
        assert abs(report.avg_score - 0.75) < 1e-6

    def test_eval_report_summary(self):
        report = EvalReport(total_cases=10, passed_cases=7, duration_ms=1234.5)
        s = report.summary()
        assert s["total"] == 10
        assert s["passed"] == 7
        assert s["duration_ms"] == 1234.5
