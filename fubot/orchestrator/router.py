"""Task classification and executor/model selection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

from fubot.config.schema import AgentProfile, Config
from fubot.orchestrator.models import AgentRouteDecision, utc_now

_TASK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("debugging", ("traceback", "stack trace", "报错", "error", "exception", "bug", "debug")),
    ("review", ("review", "code review", "审查")),
    ("testing", ("test", "pytest", "unit test", "集成测试")),
    ("refactor", ("refactor", "重构", "cleanup")),
    ("coding", ("implement", "write code", "build", "feature", "代码", "开发")),
    ("research", ("research", "investigate", "analyze", "调研", "分析")),
    ("search", ("search", "look up", "find online", "查一下")),
    ("writing", ("write", "draft", "rewrite", "润色")),
    ("ops", ("deploy", "docker", "service", "ops", "运维")),
    ("scheduling", ("schedule", "cron", "remind", "定时")),
    ("communication", ("reply", "email", "message", "沟通")),
)


class RoutePlanner:
    """Heuristic planner that stays config-driven and auditable."""

    def __init__(self, config: Config, health_cache: dict[str, Any] | None = None):
        self.config = config
        self._health_cache = health_cache or {}
        self._executor_load = defaultdict(int)
        self._executor_failures = defaultdict(int)

    def classify(self, content: str, media: list[str] | None = None) -> str:
        if media:
            return "multimodal"
        lower = content.lower()
        for task_type, patterns in _TASK_PATTERNS:
            if any(pattern in lower for pattern in patterns):
                return task_type
        return "communication"

    def choose_executors(self, task_type: str, content: str) -> list[AgentProfile]:
        overrides = self.config.orchestration.routing.task_executor_overrides.get(task_type, [])
        profiles = [self.config.get_profile(profile_id) for profile_id in overrides]
        chosen = [profile for profile in profiles if profile is not None]
        if not chosen:
            chosen = self.config.get_executor_profiles()[:1]
        if task_type not in self.config.orchestration.routing.prefer_parallel_for or len(content) < 32:
            return chosen[:1]
        limit = max(1, self.config.orchestration.routing.max_parallel_executors)
        return chosen[:limit]

    def choose_model(self, profile: AgentProfile, task_type: str, content: str) -> AgentRouteDecision:
        candidates = [profile.default_model, *profile.candidate_models, self.config.agents.defaults.model]
        model = ""
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            if profile.allowed_models and candidate not in profile.allowed_models:
                continue
            seen.add(candidate)
            model = candidate
            break
        if not model:
            model = self.config.agents.defaults.model

        provider = self._choose_provider(profile, model)
        reason_bits = [
            f"task={task_type}",
            f"role={profile.role}",
            f"tool_allowlist={len(profile.tool_allowlist)}",
        ]
        if profile.preferred_providers:
            reason_bits.append(f"preferred={','.join(profile.preferred_providers)}")
        return AgentRouteDecision(
            agent_id=profile.id,
            agent_name=profile.name or profile.id,
            agent_role=profile.role,
            task_type=task_type,
            model=model,
            provider=provider,
            reason="; ".join(reason_bits),
            fallback_chain=[model],
            health_score=self.health_score(provider or "default"),
            current_load=self._executor_load[profile.id],
        )

    def _choose_provider(self, profile: AgentProfile, model: str) -> str | None:
        providers = [
            *profile.preferred_providers,
            *(profile.allowed_providers or []),
        ]
        detected = self.config.get_provider_name(model)
        if detected:
            providers.append(detected)
        providers = [provider for provider in providers if provider]
        seen: set[str] = set()
        for provider in providers:
            if provider in seen:
                continue
            seen.add(provider)
            if self.health_score(provider) > 0:
                return provider
        return detected

    def health_score(self, provider_name: str) -> float:
        if not provider_name:
            return 1.0
        item = self._health_cache.get(provider_name)
        if not isinstance(item, dict):
            return 1.0
        failures = int(item.get("failures", 0))
        cooldown_until = item.get("cooldown_until", "")
        if cooldown_until and cooldown_until > utc_now():
            return 0.0
        return max(0.0, 1.0 - failures * 0.2)

    def mark_success(self, profile_id: str, provider_name: str | None) -> None:
        self._executor_failures[profile_id] = 0
        if provider_name:
            self._health_cache[provider_name] = {"failures": 0, "updated_at": utc_now()}

    def mark_failure(self, profile_id: str, provider_name: str | None) -> None:
        self._executor_failures[profile_id] += 1
        if provider_name:
            current = self._health_cache.get(provider_name, {})
            failures = int(current.get("failures", 0)) + 1
            cooldown = ""
            if failures >= 3:
                cooldown = utc_now()
            self._health_cache[provider_name] = {
                "failures": failures,
                "updated_at": utc_now(),
                "cooldown_until": cooldown,
            }

    def begin_load(self, profile_id: str) -> None:
        self._executor_load[profile_id] += 1

    def end_load(self, profile_id: str) -> None:
        self._executor_load[profile_id] = max(0, self._executor_load[profile_id] - 1)

    def export_health(self) -> dict[str, Any]:
        return dict(self._health_cache)

    @staticmethod
    def decision_to_dict(decision: AgentRouteDecision) -> dict[str, Any]:
        return asdict(decision)
