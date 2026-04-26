"""OpenRouter provider — OpenAI-compatible with provider routing."""

from __future__ import annotations

from typing import Any

from echo_agent.models.providers.openai_provider import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str = "", api_base: str = "", default_model: str = "", **kwargs: Any):
        extra_headers = kwargs.pop("extra_headers", {})
        extra_headers.setdefault("HTTP-Referer", kwargs.pop("referer", "https://github.com/echo-agent"))
        extra_headers.setdefault("X-Title", kwargs.pop("title", "Echo Agent"))
        super().__init__(
            api_key=api_key,
            api_base=api_base or self._BASE_URL,
            default_model=default_model,
            extra_headers=extra_headers,
            **kwargs,
        )
        self._provider_prefs: dict[str, Any] = kwargs.get("provider_preferences", {})

    def _build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        tool_choice: str | dict | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params = super()._build_params(messages, tools, model, tool_choice, **kwargs)
        if self._provider_prefs:
            params.setdefault("extra_body", {})["provider"] = self._provider_prefs
        return params
