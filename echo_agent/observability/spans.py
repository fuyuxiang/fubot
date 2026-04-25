"""GenAI semantic convention helpers for OpenTelemetry spans."""

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


def start_llm_span(tracer: Any, model: str, provider: str, operation: str = "chat") -> Any:
    if not _HAS_OTEL or tracer is None:
        return None
    span = tracer.start_span(
        f"gen_ai.{operation}",
        attributes={
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
            "gen_ai.operation.name": operation,
        },
    )
    return span


def record_llm_usage(span: Any, usage: dict[str, Any], model: str = "") -> None:
    if span is None or not _HAS_OTEL:
        return
    if model:
        span.set_attribute("gen_ai.response.model", model)
    span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
    span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    if cache_read:
        span.set_attribute("gen_ai.usage.cache_read_input_tokens", cache_read)
    if cache_create:
        span.set_attribute("gen_ai.usage.cache_creation_input_tokens", cache_create)


def end_llm_span(span: Any, error: str | None = None) -> None:
    if span is None or not _HAS_OTEL:
        return
    if error:
        span.set_status(StatusCode.ERROR, error)
    else:
        span.set_status(StatusCode.OK)
    span.end()


def start_tool_span(tracer: Any, tool_name: str) -> Any:
    if not _HAS_OTEL or tracer is None:
        return None
    return tracer.start_span(
        f"tool.{tool_name}",
        attributes={"tool.name": tool_name},
    )


def end_tool_span(span: Any, error: str | None = None) -> None:
    end_llm_span(span, error)


def start_agent_span(tracer: Any, iteration: int, strategy: str = "") -> Any:
    if not _HAS_OTEL or tracer is None:
        return None
    attrs: dict[str, Any] = {"gen_ai.agent.iteration": iteration}
    if strategy:
        attrs["gen_ai.agent.strategy"] = strategy
    return tracer.start_span("agent.iteration", attributes=attrs)
