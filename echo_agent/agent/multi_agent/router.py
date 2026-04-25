"""Deterministic intent router for precise multi-agent dispatch."""

from __future__ import annotations

import re
from collections import Counter

from echo_agent.agent.multi_agent.models import AgentProfile, DispatchCandidate, DispatchPlan
from echo_agent.agent.multi_agent.registry import AgentRegistry


_WORD_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")

_INTENT_HINTS: dict[str, tuple[str, ...]] = {
    "code": ("代码", "报错", "bug", "测试", "实现", "修复", "函数", "class", "def", "python", "typescript", "pytest"),
    "research": ("搜索", "最新", "调研", "资料", "来源", "比较", "search", "latest", "news", "compare"),
    "knowledge": ("知识库", "内部", "文档", "制度", "资料库", "handbook", "policy", "docs", "knowledge"),
    "planning": ("规划", "方案", "架构", "设计", "拆解", "路线", "plan", "roadmap", "architecture"),
    "operations": ("运行", "执行", "部署", "诊断", "状态", "日志", "run", "execute", "deploy", "status", "health"),
}

_PARALLEL_HINTS = ("同时", "分别", "并行", "对比", "比较", "各自", "multi", "parallel", "compare", "both")


class IntentRouter:
    """Scores specialist agents by task type, keywords, capabilities, and tool hints."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        default_agent: str = "general",
        route_threshold: float = 0.45,
        multi_threshold: float = 0.62,
        max_parallel_agents: int = 3,
    ):
        self._registry = registry
        self._default_agent = default_agent
        self._route_threshold = route_threshold
        self._multi_threshold = multi_threshold
        self._max_parallel = max(1, max_parallel_agents)

    def infer_task_type(self, text: str, fallback: str = "chat") -> str:
        lower = text.lower()
        scores: Counter[str] = Counter()
        for task_type, hints in _INTENT_HINTS.items():
            for hint in hints:
                if hint.lower() in lower:
                    scores[task_type] += 1
        if not scores:
            return fallback
        return scores.most_common(1)[0][0]

    def build_plan(self, text: str, *, task_type: str = "chat", available_tools: list[str] | None = None) -> DispatchPlan:
        inferred = task_type if task_type and task_type != "chat" else self.infer_task_type(text, fallback=task_type or "chat")
        tokens = self._tokens(text)
        available = set(available_tools or [])
        candidates = [self._score(profile, tokens, text, inferred, available) for profile in self._registry.list()]
        candidates.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        top = candidates[0] if candidates else DispatchCandidate(self._default_agent, 0.0, 0.0, ["no agents configured"])

        selected = [top.agent_id]
        strategy = "single"
        if top.confidence < self._route_threshold or top.agent_id == self._default_agent:
            return DispatchPlan(
                query=text,
                task_type=inferred,
                strategy="main",
                primary_agent_id=self._default_agent,
                candidates=candidates,
                selected_agent_ids=[],
                confidence=top.confidence,
                rationale=f"fallback to main agent: confidence={top.confidence:.2f}",
            )

        if self._should_parallel(text, candidates):
            selected = [
                c.agent_id
                for c in candidates
                if self._parallel_candidate_ok(c, text) and c.agent_id != self._default_agent
            ][:self._max_parallel]
            if len(selected) > 1:
                strategy = "parallel"

        return DispatchPlan(
            query=text,
            task_type=inferred,
            strategy=strategy,
            primary_agent_id=selected[0],
            candidates=candidates,
            selected_agent_ids=selected,
            confidence=top.confidence,
            rationale="; ".join(top.reasons[:5]) or "highest scored specialist",
        )

    def _score(
        self,
        profile: AgentProfile,
        tokens: set[str],
        text: str,
        task_type: str,
        available_tools: set[str],
    ) -> DispatchCandidate:
        reasons: list[str] = []
        score = float(profile.priority) / 100.0
        profile_terms = {
            *(item.lower() for item in profile.capabilities),
            *(item.lower() for item in profile.keywords),
            *(item.lower() for item in profile.task_types),
        }
        if task_type and task_type in {item.lower() for item in profile.task_types}:
            score += 3.0
            reasons.append(f"task_type={task_type}")
        if task_type and task_type in {item.lower() for item in profile.capabilities}:
            score += 2.0
            reasons.append(f"capability={task_type}")

        lower = text.lower()
        keyword_hits = [kw for kw in profile.keywords if kw and kw.lower() in lower]
        capability_hits = sorted(tokens.intersection(profile_terms))
        if keyword_hits:
            score += min(3.0, len(keyword_hits) * 0.7)
            reasons.append("keywords=" + ",".join(keyword_hits[:5]))
        if capability_hits:
            score += min(2.0, len(capability_hits) * 0.35)
            reasons.append("terms=" + ",".join(capability_hits[:5]))

        if profile.tools_allow and available_tools:
            coverage = len(set(profile.tools_allow).intersection(available_tools)) / max(1, len(profile.tools_allow))
            score += coverage * 0.5
            if coverage:
                reasons.append(f"tool_coverage={coverage:.2f}")

        if profile.is_general:
            score += 0.25
            if task_type == "chat":
                score += 1.5
                reasons.append("general chat")

        confidence = min(1.0, score / 6.0)
        return DispatchCandidate(profile.id, round(score, 4), round(confidence, 4), reasons)

    def _should_parallel(self, text: str, candidates: list[DispatchCandidate]) -> bool:
        lower = text.lower()
        if any(hint in lower for hint in _PARALLEL_HINTS):
            return True
        strong = [c for c in candidates if c.confidence >= self._multi_threshold and c.agent_id != self._default_agent]
        if len(strong) >= 2 and strong[0].score - strong[1].score <= 1.0:
            return True
        return False

    def _parallel_candidate_ok(self, candidate: DispatchCandidate, text: str) -> bool:
        if candidate.confidence >= self._multi_threshold:
            return True
        lower = text.lower()
        has_parallel_hint = any(hint in lower for hint in _PARALLEL_HINTS)
        has_domain_reason = any(
            reason.startswith(("task_type=", "capability=", "keywords=", "terms="))
            for reason in candidate.reasons
        )
        return has_parallel_hint and has_domain_reason and candidate.score >= 1.5

    def _tokens(self, text: str) -> set[str]:
        lower = text.lower()
        tokens = {item for item in _WORD_RE.findall(lower) if item.strip()}
        cjk = [ch for ch in lower if "\u4e00" <= ch <= "\u9fff"]
        tokens.update("".join(cjk[i:i + 2]) for i in range(max(0, len(cjk) - 1)))
        return tokens
