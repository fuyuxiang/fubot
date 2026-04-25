from __future__ import annotations
from typing import Any, Callable, Awaitable
from loguru import logger
from echo_agent.agent.planning.models import Plan, PlanStep, StepAction, StrategyType, Feedback
from echo_agent.agent.planning.strategies import (
    PlanningStrategy, ReactStrategy, PlanExecuteStrategy,
    TreeOfThoughtStrategy, LATSStrategy,
)
from echo_agent.agent.planning.reflection import ReflectionModule


class AgentPlanner:
    def __init__(
        self,
        llm_call: Callable[..., Awaitable[Any]],
        default_strategy: str = "auto",
        max_tree_depth: int = 5,
        reflection_enabled: bool = True,
    ):
        self._llm_call = llm_call
        self._default_strategy = default_strategy
        self._max_tree_depth = max_tree_depth
        self._reflection = ReflectionModule(llm_call) if reflection_enabled else None
        self._strategies: dict[StrategyType, PlanningStrategy] = {
            StrategyType.REACT: ReactStrategy(llm_call),
            StrategyType.PLAN_EXECUTE: PlanExecuteStrategy(llm_call),
            StrategyType.TREE_OF_THOUGHT: TreeOfThoughtStrategy(llm_call),
            StrategyType.LATS: LATSStrategy(llm_call),
        }

    def select_strategy(self, query: str, tool_count: int, token_estimate: int) -> StrategyType:
        if self._default_strategy != "auto":
            try:
                return StrategyType(self._default_strategy)
            except ValueError:
                pass
        if token_estimate < 500 and tool_count <= 2:
            return StrategyType.REACT
        if tool_count > 5 or token_estimate > 2000:
            return StrategyType.PLAN_EXECUTE
        return StrategyType.REACT

    async def create_plan(self, query: str, tools: list[dict], context: str = "", token_estimate: int = 0) -> Plan:
        strategy_type = self.select_strategy(query, len(tools), token_estimate)
        strategy = self._strategies[strategy_type]
        logger.debug("Planning with strategy: {}", strategy_type.value)
        plan = await strategy.plan(query, tools, context)
        return plan

    async def execute_step(self, plan: Plan, step_index: int, result: str = "") -> StepAction:
        strategy = self._strategies.get(plan.strategy, self._strategies[StrategyType.REACT])
        return await strategy.step(plan, step_index, result)

    async def reflect(self, plan: Plan, results: list[str]) -> Feedback:
        if not self._reflection:
            return Feedback(score=0.5)
        return await self._reflection.critique(plan, results)
