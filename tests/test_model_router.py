from __future__ import annotations

import pytest

from echo_agent.config.schema import ModelRouteConfig, ModelsConfig, ProviderConfig
from echo_agent.models.provider import LLMProvider, LLMResponse
from echo_agent.models.providers import validate_provider_config
from echo_agent.models.router import ModelRouter


class FakeProvider(LLMProvider):
    def __init__(self, model: str):
        super().__init__()
        self._model = model

    async def chat(self, messages, tools=None, model=None, tool_choice=None, **kwargs):
        return LLMResponse(content=f"model={model or self._model}")

    def get_default_model(self) -> str:
        return self._model


@pytest.mark.asyncio
async def test_router_returns_task_route_and_fallback_after_failure() -> None:
    config = ModelsConfig(
        default_model="fast-model",
        providers=[
            ProviderConfig(name="fast", models=["fast-model"]),
            ProviderConfig(name="deep", models=["deep-model"]),
        ],
        routes=[
            ModelRouteConfig(
                provider="fast",
                model="fast-model",
                task_types=["code"],
                fallback_models=["deep-model"],
            )
        ],
    )
    router = ModelRouter(config, cooldown_seconds=1)
    router.register_provider("fast", FakeProvider("fast-model"))
    router.register_provider("deep", FakeProvider("deep-model"))

    first_name, _, first_decision = router.route_provider_with_fallback(task_type="code")
    assert first_name == "fast"
    assert first_decision.model == "fast-model"

    router.mark_failure("fast", "one")
    router.mark_failure("fast", "two")
    router.mark_failure("fast", "three")

    second_name, _, second_decision = router.route_provider_with_fallback(task_type="code")
    assert second_name == "deep"
    assert second_decision.model == "deep-model"


def test_provider_validation_requires_explicit_model() -> None:
    with pytest.raises(ValueError, match="requires an explicit model"):
        validate_provider_config(
            ProviderConfig(name="openai", api_key="key"),
            default_model="",
        )


def test_provider_validation_allows_local_openai_compatible_without_key() -> None:
    validate_provider_config(
        ProviderConfig(name="openai", api_base="http://127.0.0.1:11434/v1"),
        default_model="local-model",
    )


def test_provider_validation_requires_key_for_remote_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires api_key"):
        validate_provider_config(
            ProviderConfig(name="anthropic", models=["claude-model"]),
            default_model="",
        )
