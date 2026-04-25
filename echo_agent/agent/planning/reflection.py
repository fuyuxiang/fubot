from __future__ import annotations
import json
from typing import Any, Callable, Awaitable
from loguru import logger
from echo_agent.agent.planning.models import Plan, Feedback

_CRITIQUE_TOOL = [{
    "type": "function",
    "function": {
        "name": "critique",
        "description": "Evaluate the plan execution and provide feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "0.0-1.0 quality score"},
                "should_replan": {"type": "boolean"},
                "critique": {"type": "string"},
                "suggestions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["score", "should_replan", "critique"]
        }
    }
}]


class ReflectionModule:
    def __init__(self, llm_call: Callable[..., Awaitable[Any]]):
        self._llm_call = llm_call

    async def critique(self, plan: Plan, results: list[str]) -> Feedback:
        plan_summary = plan.to_prompt()
        results_text = "\n".join(f"Step {i}: {r[:300]}" for i, r in enumerate(results))
        try:
            response = await self._llm_call(
                messages=[
                    {"role": "system", "content": "Evaluate this plan execution. Call critique."},
                    {"role": "user", "content": f"Plan:\n{plan_summary}\n\nResults:\n{results_text}"},
                ],
                tools=_CRITIQUE_TOOL,
                tool_choice={"type": "function", "function": {"name": "critique"}},
            )
            if response.tool_calls:
                args = response.tool_calls[0].arguments
                if isinstance(args, str):
                    args = json.loads(args)
                return Feedback(
                    should_replan=args.get("should_replan", False),
                    critique=args.get("critique", ""),
                    suggestions=args.get("suggestions", []),
                    score=args.get("score", 0.5),
                )
        except Exception as e:
            logger.warning("Reflection failed: {}", e)
        return Feedback(score=0.5, critique="Reflection unavailable")
