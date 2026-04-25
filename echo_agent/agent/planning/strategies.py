from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable
from loguru import logger
from echo_agent.agent.planning.models import Plan, PlanStep, StepAction, StrategyType, StepStatus

_PLAN_TOOL = [{
    "type": "function",
    "function": {
        "name": "create_plan",
        "description": "Create an execution plan for the task.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "steps": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "tool_hint": {"type": "string"},
                    },
                    "required": ["description"]
                }},
            },
            "required": ["goal", "steps"]
        }
    }
}]

_NEXT_ACTION_TOOL = [{
    "type": "function",
    "function": {
        "name": "next_action",
        "description": "Decide the next action based on current plan state.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["execute_tool", "respond", "replan", "backtrack"]},
                "tool_name": {"type": "string"},
                "tool_args": {"type": "object"},
                "reasoning": {"type": "string"},
            },
            "required": ["action", "reasoning"]
        }
    }
}]


class PlanningStrategy(ABC):
    def __init__(self, llm_call: Callable[..., Awaitable[Any]]):
        self._llm_call = llm_call

    @abstractmethod
    async def plan(self, query: str, tools: list[dict], context: str) -> Plan: ...

    @abstractmethod
    async def step(self, plan: Plan, step_index: int, result: str) -> StepAction: ...

class ReactStrategy(PlanningStrategy):
    """ReAct 策略：交替推理与行动，适用于简单直接的任务。"""

    async def plan(self, query: str, tools: list[dict], context: str) -> Plan:
        return Plan(
            strategy=StrategyType.REACT,
            goal=query,
            steps=[PlanStep(index=0, description="Reason and act iteratively")],
        )

    async def step(self, plan: Plan, step_index: int, result: str) -> StepAction:
        try:
            response = await self._llm_call(
                messages=[
                    {"role": "system", "content": "You are a ReAct agent. Decide the next action. Call next_action."},
                    {"role": "user", "content": f"Goal: {plan.goal}\nPrevious result: {result or 'none'}\nDecide next action."},
                ],
                tools=_NEXT_ACTION_TOOL,
                tool_choice={"type": "function", "function": {"name": "next_action"}},
            )
            if response.tool_calls:
                tool_arguments = response.tool_calls[0].arguments
                if isinstance(tool_arguments, str):
                    tool_arguments = json.loads(tool_arguments)
                return StepAction(**{k: v for k, v in tool_arguments.items() if k in StepAction.__dataclass_fields__})
        except Exception as e:
            logger.warning("ReAct step failed: {}", e)
        return StepAction(action="respond", reasoning="Fallback to direct response")


class PlanExecuteStrategy(PlanningStrategy):
    """计划-执行策略：先生成完整计划，再逐步执行，适用于多步骤复杂任务。"""

    async def plan(self, query: str, tools: list[dict], context: str) -> Plan:
        tool_names = [t.get("function", t).get("name", "") for t in tools]
        try:
            response = await self._llm_call(
                messages=[
                    {"role": "system", "content": f"Create a step-by-step plan. Available tools: {', '.join(tool_names)}. Call create_plan."},
                    {"role": "user", "content": f"Task: {query}\nContext: {context[:1000] if context else 'none'}"},
                ],
                tools=_PLAN_TOOL,
                tool_choice={"type": "function", "function": {"name": "create_plan"}},
            )
            if response.tool_calls:
                tool_arguments = response.tool_calls[0].arguments
                if isinstance(tool_arguments, str):
                    tool_arguments = json.loads(tool_arguments)
                steps = [PlanStep(index=i, description=s["description"], tool_hint=s.get("tool_hint", ""))
                         for i, s in enumerate(tool_arguments.get("steps", []))]
                return Plan(strategy=StrategyType.PLAN_EXECUTE, goal=tool_arguments.get("goal", query), steps=steps)
        except Exception as e:
            logger.warning("Plan creation failed: {}", e)
        return Plan(strategy=StrategyType.PLAN_EXECUTE, goal=query,
                     steps=[PlanStep(index=0, description="Execute task directly")])

    async def step(self, plan: Plan, step_index: int, result: str) -> StepAction:
        current = plan.steps[step_index] if step_index < len(plan.steps) else None
        if not current:
            return StepAction(action="respond", reasoning="All steps completed")
        return StepAction(
            action="execute_tool",
            tool_name=current.tool_hint,
            reasoning=current.description,
        )


class TreeOfThoughtStrategy(PlanningStrategy):
    """思维树策略：探索多条推理路径，选择最优方案。"""

    async def plan(self, query: str, tools: list[dict], context: str) -> Plan:
        # Generate 3 candidate plans, score them, pick best
        candidates: list[Plan] = []
        for candidate_index in range(3):
            try:
                response = await self._llm_call(
                    messages=[
                        {"role": "system", "content": f"Create approach #{candidate_index+1} (of 3 different approaches). Call create_plan."},
                        {"role": "user", "content": f"Task: {query}"},
                    ],
                    tools=_PLAN_TOOL,
                    tool_choice={"type": "function", "function": {"name": "create_plan"}},
                )
                if response.tool_calls:
                    tool_arguments = response.tool_calls[0].arguments
                    if isinstance(tool_arguments, str):
                        tool_arguments = json.loads(tool_arguments)
                    steps = [PlanStep(index=step_index, description=step_dict["description"], tool_hint=step_dict.get("tool_hint", ""))
                             for step_index, step_dict in enumerate(tool_arguments.get("steps", []))]
                    candidates.append(Plan(strategy=StrategyType.TREE_OF_THOUGHT, goal=tool_arguments.get("goal", query), steps=steps))
            except Exception as exc:
                logger.warning("思维树候选方案 #{} 生成失败: {}", candidate_index + 1, exc)
                continue
        if not candidates:
            return Plan(strategy=StrategyType.TREE_OF_THOUGHT, goal=query,
                         steps=[PlanStep(index=0, description="Execute directly")])
        # Pick the plan with fewest steps (simplest viable approach)
        return min(candidates, key=lambda p: len(p.steps))

    async def step(self, plan: Plan, step_index: int, result: str) -> StepAction:
        current = plan.steps[step_index] if step_index < len(plan.steps) else None
        if not current:
            return StepAction(action="respond", reasoning="All paths explored")
        return StepAction(action="execute_tool", tool_name=current.tool_hint, reasoning=current.description)


class LATSStrategy(PlanningStrategy):
    """LATS 策略：基于蒙特卡洛树搜索的语言代理，支持回溯，适用于高风险复杂任务。"""

    async def plan(self, query: str, tools: list[dict], context: str) -> Plan:
        # Start with a plan, will backtrack on failures
        delegate = PlanExecuteStrategy(self._llm_call)
        plan = await delegate.plan(query, tools, context)
        plan.strategy = StrategyType.LATS
        return plan

    async def step(self, plan: Plan, step_index: int, result: str) -> StepAction:
        # Check if previous step failed -- if so, backtrack
        if step_index > 0 and plan.steps[step_index - 1].status == StepStatus.FAILED:
            return StepAction(action="replan", reasoning=f"Step {step_index-1} failed: {plan.steps[step_index-1].result}")
        current = plan.steps[step_index] if step_index < len(plan.steps) else None
        if not current:
            return StepAction(action="respond", reasoning="Search complete")
        return StepAction(action="execute_tool", tool_name=current.tool_hint, reasoning=current.description)
