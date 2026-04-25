"""Evaluation reporter — output formatting."""

from __future__ import annotations

import json
from typing import Any

from echo_agent.evaluation.runner import EvalReport, CaseResult


class EvalReporter:
    def to_json(self, report: EvalReport) -> str:
        return json.dumps({
            "summary": report.summary(),
            "results": [self._case_to_dict(r) for r in report.results],
        }, indent=2, ensure_ascii=False)

    def to_table(self, report: EvalReport) -> str:
        lines = [
            f"{'ID':<20} {'Pass':<6} {'Score':<8} {'Time(ms)':<10} {'Error'}",
            "-" * 70,
        ]
        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"{r.case_id:<20} {status:<6} {r.score:<8.3f} {r.duration_ms:<10.0f} {r.error[:30]}")
        lines.append("-" * 70)
        lines.append(f"Total: {report.total_cases} | Passed: {report.passed_cases} | Rate: {report.pass_rate:.1%} | Avg Score: {report.avg_score:.3f}")
        return "\n".join(lines)

    @staticmethod
    def _case_to_dict(result: CaseResult) -> dict[str, Any]:
        return {
            "id": result.case_id,
            "passed": result.passed,
            "score": round(result.score, 3),
            "response_preview": result.response[:200],
            "tools_used": result.tools_used,
            "iterations": result.iterations,
            "duration_ms": round(result.duration_ms, 1),
            "error": result.error,
            "metrics": [{"name": m.name, "score": m.score, "passed": m.passed} for m in result.metrics],
        }
