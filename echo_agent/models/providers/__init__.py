"""Provider factory — creates LLMProvider instances from config."""

from __future__ import annotations

from typing import Any

from loguru import logger

from echo_agent.config.schema import ProviderConfig
from echo_agent.models.credential_pool import CredentialPool
from echo_agent.models.provider import LLMProvider, LLMResponse, StreamDeltaCallback
from echo_agent.models.rate_limiter import RateLimitedProvider, TokenBucketLimiter

_PROVIDER_MAP: dict[str, str] = {
    "openai": "echo_agent.models.providers.openai_provider.OpenAIProvider",
    "anthropic": "echo_agent.models.providers.anthropic_provider.AnthropicProvider",
    "bedrock": "echo_agent.models.providers.bedrock_provider.BedrockProvider",
    "aws": "echo_agent.models.providers.bedrock_provider.BedrockProvider",
    "gemini": "echo_agent.models.providers.gemini_provider.GeminiProvider",
    "google": "echo_agent.models.providers.gemini_provider.GeminiProvider",
    "openrouter": "echo_agent.models.providers.openrouter_provider.OpenRouterProvider",
}


def _import_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_provider(config: ProviderConfig) -> LLMProvider:
    name = config.name.lower().strip()
    dotted = _PROVIDER_MAP.get(name)

    if dotted:
        cls = _import_class(dotted)
    else:
        from echo_agent.models.providers.openai_provider import OpenAIProvider
        cls = OpenAIProvider
        logger.info("Unknown provider '{}', using OpenAI-compatible mode", name)

    kwargs: dict[str, Any] = {}
    if config.extra_headers:
        kwargs["extra_headers"] = config.extra_headers
    if config.models:
        kwargs["default_model"] = config.models[0]

    pool: CredentialPool | None = None
    if config.credential_pool:
        pool = CredentialPool(config.credential_pool)
        api_key = pool.get_next()
    else:
        api_key = config.api_key

    provider = cls(api_key=api_key, api_base=config.api_base, **kwargs)

    if pool:
        provider = _PooledProvider(provider, pool, cls, config)

    if config.rate_limit_rpm > 0:
        limiter = TokenBucketLimiter(tokens_per_minute=config.rate_limit_rpm)
        provider = RateLimitedProvider(provider, limiter)

    return provider


class _PooledProvider(LLMProvider):
    """Wraps a provider with credential rotation on errors."""

    def __init__(self, inner: LLMProvider, pool: CredentialPool, cls: type, config: ProviderConfig):
        super().__init__()
        self._inner = inner
        self._pool = pool
        self._cls = cls
        self._config = config
        self.generation = inner.generation

    async def chat(self, messages, tools=None, model=None, tool_choice=None, **kwargs):
        resp = await self._inner.chat(messages, tools, model, tool_choice, **kwargs)
        if resp.finish_reason == "error" and self._pool.size > 1:
            self._pool.report_error(self._inner.api_key)
            next_key = self._pool.get_next()
            self._inner.api_key = next_key
            self._inner._client = self._inner._build_client()
            logger.info("Rotated to next credential in pool")
            resp = await self._inner.chat(messages, tools, model, tool_choice, **kwargs)
            if resp.finish_reason != "error":
                self._pool.report_success(next_key)
        else:
            self._pool.report_success(self._inner.api_key)
        return resp

    async def chat_stream(
        self,
        messages,
        tools=None,
        model=None,
        tool_choice=None,
        on_delta: StreamDeltaCallback | None = None,
        **kwargs,
    ):
        emitted = False

        async def wrapped(delta: str) -> None:
            nonlocal emitted
            emitted = True
            if on_delta:
                await on_delta(delta)

        resp = await self._inner.chat_stream(
            messages,
            tools,
            model,
            tool_choice,
            on_delta=wrapped,
            **kwargs,
        )
        if resp.finish_reason == "error" and self._pool.size > 1 and not emitted:
            self._pool.report_error(self._inner.api_key)
            next_key = self._pool.get_next()
            self._inner.api_key = next_key
            self._inner._client = self._inner._build_client()
            logger.info("Rotated to next credential in pool")
            resp = await self._inner.chat_stream(
                messages,
                tools,
                model,
                tool_choice,
                on_delta=wrapped,
                **kwargs,
            )
            if resp.finish_reason != "error":
                self._pool.report_success(next_key)
        else:
            self._pool.report_success(self._inner.api_key)
        return resp

    def get_default_model(self) -> str:
        return self._inner.get_default_model()
