"""Evaluation runner — executes test cases against the agent."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from loguru import logger

from echo_agent.evaluation.dataset import EvalCase, EvalDataset
from echo_agent.evaluation.metrics import (
    MetricResult, contains_all, tool_usage_correctness,
    iteration_efficiency, response_quality,
)

if TYPE_CHECKING:
    from echo_agent.agent.loop import AgentLoop


@dataclass
class CaseResult:
    case_id: str = ""
    passed: bool = False
    response: str = ""
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0
    duration_ms: float = 0.0
    metrics: list[MetricResult] = field(default_factory=list)
    error: str = ""

    @property
    def score(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.score for m in self.metrics) / len(self.metrics)


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)
    total_cases: int = 0
    passed_cases: int = 0
    duration_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / max(self.total_cases, 1)

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total_cases,
            "passed": self.passed_cases,
            "pass_rate": round(self.pass_rate, 3),
            "avg_score": round(self.avg_score, 3),
            "duration_ms": round(self.duration_ms, 1),
        }


class EvalRunner:
    def __init__(self, agent_loop: AgentLoop, parallel: int = 3, timeout: int = 120):
        self._loop = agent_loop
        self._parallel = parallel
        self._timeout = timeout

    async def run_case(self, case: EvalCase) -> CaseResult:
        start = time.monotonic()
        result = CaseResult(case_id=case.id)
        try:
            from echo_agent.bus.events import InboundEvent
            import uuid
            event = InboundEvent.text_message(
                channel="eval", sender_id="eval-runner",
                chat_id=f"eval:{case.id}", text=case.input,
            )
            proc_result = await asyncio.wait_for(
                self._loop._process_event(event, trace_id=uuid.uuid4().hex[:12]),
                timeout=self._timeout,
            )
            result.response = proc_result.response_text or ""
        except asyncio.TimeoutError:
            result.error = "Timeout"
        except Exception as e:
            result.error = str(e)

        result.duration_ms = (time.monotonic() - start) * 1000

        # Score
        result.metrics.append(contains_all(case.expected_contains, result.response))
        result.metrics.append(tool_usage_correctness(case.expected_tools, result.tools_used))
        result.metrics.append(iteration_efficiency(result.iterations, case.max_iterations))
        if case.expected_output:
            result.metrics.append(response_quality(case.expected_output, result.response))
        result.passed = all(m.passed for m in result.metrics) and not result.error
        return result

    async def run_dataset(self, dataset: EvalDataset) -> EvalReport:
        start = time.monotonic()
        semaphore = asyncio.Semaphore(self._parallel)

        async def run_with_limit(case: EvalCase) -> CaseResult:
            async with semaphore:
                return await self.run_case(case)

        tasks = [run_with_limit(c) for c in dataset.cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        report = EvalReport(total_cases=len(dataset.cases))
        for r in results:
            if isinstance(r, CaseResult):
                report.results.append(r)
                if r.passed:
                    report.passed_cases += 1
            else:
                report.results.append(CaseResult(error=str(r)))
        report.duration_ms = (time.monotonic() - start) * 1000
        logger.info("Eval complete: {}/{} passed, avg score {:.3f}", report.passed_cases, report.total_cases, report.avg_score)
        return report
