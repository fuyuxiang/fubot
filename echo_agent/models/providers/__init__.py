"""Provider factory — creates LLMProvider instances from config."""

from __future__ import annotations

import os
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

_API_KEY_ENV: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
}

_BEDROCK_PROVIDERS = {"bedrock", "aws"}


def _import_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _env_api_key(provider_name: str) -> str:
    for env_name in _API_KEY_ENV.get(provider_name, ()):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return ""


def _has_aws_credentials() -> bool:
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
        or os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
    )


def _allows_keyless_openai_compatible(provider_name: str, config: ProviderConfig) -> bool:
    if not config.api_base:
        return False
    return provider_name == "openai" or provider_name not in _PROVIDER_MAP


def validate_provider_config(config: ProviderConfig, *, default_model: str = "") -> None:
    provider_name = config.name.lower().strip()
    if not provider_name:
        raise ValueError("provider name is required")

    if provider_name not in _PROVIDER_MAP and not config.api_base:
        raise ValueError(
            f"provider '{config.name}' is OpenAI-compatible by default and requires api_base"
        )

    if not config.models and not default_model:
        raise ValueError(
            f"provider '{config.name}' requires an explicit model; set models.defaultModel "
            "or models.providers[].models"
        )

    if provider_name in _BEDROCK_PROVIDERS:
        if config.api_key or _has_aws_credentials():
            return
        # boto3/AnthropicBedrock can still resolve instance/task role credentials at call time.
        return

    if config.api_key or config.credential_pool or _env_api_key(provider_name):
        return

    if _allows_keyless_openai_compatible(provider_name, config):
        return

    env_hint = ", ".join(_API_KEY_ENV.get(provider_name, ()))
    hint = f" or set {env_hint}" if env_hint else ""
    raise ValueError(f"provider '{config.name}' requires api_key{hint}")


def create_provider(config: ProviderConfig, *, default_model: str = "") -> LLMProvider:
    name = config.name.lower().strip()
    default_model = default_model.strip()
    validate_provider_config(config, default_model=default_model)
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
    configured_default = default_model or (config.models[0] if config.models else "")
    if configured_default:
        kwargs["default_model"] = configured_default

    pool: CredentialPool | None = None
    if config.credential_pool:
        pool = CredentialPool(config.credential_pool)
        api_key = pool.get_next()
    else:
        api_key = config.api_key or _env_api_key(name)

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
