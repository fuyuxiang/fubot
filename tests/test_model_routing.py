from __future__ import annotations

from fubot.config.schema import Config
from fubot.orchestrator.router import RoutePlanner


def test_route_planner_classifies_debugging_and_multimodal() -> None:
    planner = RoutePlanner(Config())

    assert planner.classify("Please debug this traceback") == "debugging"
    assert planner.classify("What is in this image?", media=["/tmp/a.png"]) == "multimodal"


def test_route_planner_uses_parallel_profiles_for_coding() -> None:
    config = Config()
    config.orchestration.routing.max_parallel_executors = 2
    planner = RoutePlanner(config)

    profiles = planner.choose_executors("coding", "implement a new feature with tests and refactor the module")

    assert [profile.id for profile in profiles] == ["builder", "verifier"]


def test_route_planner_prefers_healthy_provider_after_cooldown() -> None:
    config = Config()
    profile = config.orchestration.executors["researcher"]
    profile.preferred_providers = ["dashscope", "openrouter"]
    planner = RoutePlanner(
        config,
        {
            "dashscope": {
                "failures": 5,
                "updated_at": "2026-03-14T00:00:00+00:00",
                "cooldown_until": "9999-01-01T00:00:00+00:00",
            }
        },
    )

    decision = planner.choose_model(profile, "research", "analyze the problem")

    assert decision.provider == "openrouter"
    assert decision.model == config.agents.defaults.model


def test_route_planner_records_failures_in_health_cache() -> None:
    planner = RoutePlanner(Config())

    planner.mark_failure("builder", "dashscope")
    planner.mark_failure("builder", "dashscope")
    planner.mark_failure("builder", "dashscope")

    health = planner.export_health()
    assert health["dashscope"]["failures"] == 3
    assert "cooldown_until" in health["dashscope"]
