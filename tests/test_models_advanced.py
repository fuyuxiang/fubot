"""Comprehensive tests for credential_pool, inference, and rate_limiter modules."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from echo_agent.models.credential_pool import CredentialPool
from echo_agent.models.inference import InferenceConstraints, InferenceController
from echo_agent.models.provider import LLMResponse, ToolCallRequest
from echo_agent.models.rate_limiter import RateLimitedProvider, TokenBucketLimiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _make_response(
    content: str | None = None,
    tool_names: list[str] | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    tool_calls = []
    if tool_names:
        for n in tool_names:
            tool_calls.append(ToolCallRequest(id=f"call_{n}", name=n, arguments={}))
    return LLMResponse(content=content, tool_calls=tool_calls, finish_reason=finish_reason)


def _make_provider_mock() -> MagicMock:
    provider = MagicMock()
    provider.api_key = "test"
    provider.api_base = "http://test"
    provider.generation = MagicMock()
    provider.chat = AsyncMock(return_value=_make_response(content="ok"))
    provider.chat_stream = AsyncMock(return_value=_make_response(content="streamed"))
    provider.get_default_model = MagicMock(return_value="test-model")
    return provider


# ===========================================================================
# TestCredentialPool
# ===========================================================================

class TestCredentialPool:

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="at least one key"):
            CredentialPool([])

    def test_size_property(self):
        pool = CredentialPool(["a", "b", "c"])
        assert pool.size == 3

    def test_round_robin_cycles(self):
        pool = CredentialPool(["a", "b", "c"])
        results = [pool.get_next() for _ in range(6)]
        assert results == ["a", "b", "c", "a", "b", "c"]

    def test_report_error_exhausts_at_three(self):
        pool = CredentialPool(["a", "b"])
        pool.report_error("a")
        pool.report_error("a")
        # Two errors — key still usable
        assert pool.get_next() == "a"
        pool.report_error("a")
        # Third error — key exhausted, should skip to "b"
        assert pool.get_next() == "b"

    def test_exhausted_key_skipped(self):
        pool = CredentialPool(["a", "b", "c"])
        for _ in range(3):
            pool.report_error("b")
        results = [pool.get_next() for _ in range(4)]
        assert "b" not in results

    def test_all_exhausted_resets(self):
        pool = CredentialPool(["a", "b"])
        for key in ["a", "b"]:
            for _ in range(3):
                pool.report_error(key)
        # All exhausted — should reset and return first key
        result = pool.get_next()
        assert result == "a"

    def test_report_success_clears_errors(self):
        pool = CredentialPool(["a", "b"])
        pool.report_error("a")
        pool.report_error("a")
        pool.report_success("a")
        # Error count reset — three more errors needed to exhaust
        pool.report_error("a")
        pool.report_error("a")
        assert pool.get_next() == "a"


# ===========================================================================
# TestInferenceConstraints
# ===========================================================================

class TestInferenceConstraints:

    def test_defaults(self):
        c = InferenceConstraints()
        assert c.allowed_tools is None
        assert c.blocked_tools is None
        assert c.output_format is None
        assert c.max_output_tokens == 4096
        assert c.require_tool_call is False
        assert c.require_confirmation_for == []


# ===========================================================================
# TestInferenceController
# ===========================================================================

class TestInferenceController:

    def _controller(self, **kwargs) -> InferenceController:
        ctrl = InferenceController()
        ctrl.set_constraints(InferenceConstraints(**kwargs))
        return ctrl

    # -- filter_tools -------------------------------------------------------

    def test_filter_tools_allowed(self):
        ctrl = self._controller(allowed_tools=["search", "read"])
        tools = [_make_tool("search"), _make_tool("write"), _make_tool("read")]
        result = ctrl.filter_tools(tools)
        names = [t["function"]["name"] for t in result]
        assert names == ["search", "read"]

    def test_filter_tools_blocked(self):
        ctrl = self._controller(blocked_tools=["delete"])
        tools = [_make_tool("search"), _make_tool("delete")]
        result = ctrl.filter_tools(tools)
        names = [t["function"]["name"] for t in result]
        assert names == ["search"]

    def test_filter_tools_allowed_and_blocked(self):
        ctrl = self._controller(allowed_tools=["a", "b", "c"], blocked_tools=["b"])
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c"), _make_tool("d")]
        result = ctrl.filter_tools(tools)
        names = [t["function"]["name"] for t in result]
        assert names == ["a", "c"]

    # -- validate_response --------------------------------------------------

    def test_validate_require_tool_call_missing(self):
        ctrl = self._controller(require_tool_call=True)
        resp = _make_response(content="just text")
        issues = ctrl.validate_response(resp)
        assert any("Expected tool call" in i for i in issues)

    def test_validate_blocked_tool_used(self):
        ctrl = self._controller(blocked_tools=["danger"])
        resp = _make_response(tool_names=["danger"])
        issues = ctrl.validate_response(resp)
        assert any("blocked" in i.lower() for i in issues)

    def test_validate_json_format_invalid(self):
        ctrl = self._controller(output_format="json")
        resp = _make_response(content="not json at all")
        issues = ctrl.validate_response(resp)
        assert any("JSON" in i for i in issues)

    def test_validate_response_valid(self):
        ctrl = self._controller(
            allowed_tools=["search"],
            output_format="json",
        )
        resp = _make_response(content='{"ok": true}', tool_names=["search"])
        issues = ctrl.validate_response(resp)
        assert issues == []

    # -- hallucination markers ----------------------------------------------

    def test_hallucination_markers_detected(self):
        ctrl = InferenceController()
        text = "As an AI, I cannot browse the web. As of my training cutoff, I don't have access to that."
        markers = ctrl.check_hallucination_markers(text)
        assert len(markers) == 4

    def test_hallucination_markers_clean(self):
        ctrl = InferenceController()
        markers = ctrl.check_hallucination_markers("The weather today is sunny.")
        assert markers == []

    # -- needs_confirmation -------------------------------------------------

    def test_needs_confirmation(self):
        ctrl = self._controller(require_confirmation_for=["deploy", "delete"])
        assert ctrl.needs_confirmation("deploy") is True
        assert ctrl.needs_confirmation("delete") is True
        assert ctrl.needs_confirmation("read") is False

    # -- build_verification_prompt ------------------------------------------

    def test_build_verification_prompt_format(self):
        ctrl = InferenceController()
        prompt = ctrl.build_verification_prompt("What is 2+2?", "4")
        assert "What is 2+2?" in prompt
        assert "4" in prompt
        assert "CORRECT" in prompt

    # -- layer_system_prompts -----------------------------------------------

    def test_layer_system_prompts_skips_empty(self):
        ctrl = InferenceController()
        result = ctrl.layer_system_prompts("Base prompt", "", "  ", "Extra rules")
        assert result == "Base prompt\n\n---\n\nExtra rules"
        # Verify empty/whitespace-only layers are excluded
        assert result.count("---") == 1


# ===========================================================================
# TestTokenBucketLimiter
# ===========================================================================

class TestTokenBucketLimiter:

    @pytest.mark.asyncio
    async def test_acquire_immediate_when_available(self):
        limiter = TokenBucketLimiter(tokens_per_minute=600, burst=10)
        # Bucket starts full at capacity (10), so acquiring 5 should be instant
        await limiter.acquire(5)
        assert limiter._tokens < 10

    @pytest.mark.asyncio
    async def test_acquire_waits_when_empty(self):
        limiter = TokenBucketLimiter(tokens_per_minute=60, burst=1)
        # Drain the bucket
        await limiter.acquire(1)
        assert limiter._tokens == 0.0

        # Patch asyncio.sleep to simulate waiting, and advance _last_refill
        # so _refill() adds tokens on the next loop iteration.
        original_sleep = asyncio.sleep

        async def fake_sleep(duration):
            # Simulate time passing by backdating _last_refill
            limiter._last_refill -= 2.0
            # Don't actually sleep

        with patch("echo_agent.models.rate_limiter.asyncio.sleep", side_effect=fake_sleep):
            await limiter.acquire(1)
        # Should have succeeded after the fake sleep advanced time
        assert limiter._tokens >= 0

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        limiter = TokenBucketLimiter(tokens_per_minute=600, burst=10)
        await limiter.acquire(10)  # drain
        # Simulate time passing
        limiter._last_refill = time.monotonic() - 1.0  # 1 second ago
        limiter._refill()
        # rate = 600/60 = 10/sec, so 1 second -> 10 tokens
        assert limiter._tokens == pytest.approx(10.0, abs=1.0)


# ===========================================================================
# TestRateLimitedProvider
# ===========================================================================

class TestRateLimitedProvider:

    @pytest.mark.asyncio
    async def test_chat_acquires_before_delegating(self):
        inner = _make_provider_mock()
        limiter = TokenBucketLimiter(tokens_per_minute=600, burst=10)
        provider = RateLimitedProvider(inner, limiter)

        msgs = [{"role": "user", "content": "hi"}]
        result = await provider.chat(msgs, tools=None, model="m")

        inner.chat.assert_awaited_once_with(msgs, None, "m", None)
        assert result.content == "ok"
        # Limiter should have consumed 1 token
        assert limiter._tokens < 10

    @pytest.mark.asyncio
    async def test_chat_stream_acquires_before_delegating(self):
        inner = _make_provider_mock()
        limiter = TokenBucketLimiter(tokens_per_minute=600, burst=10)
        provider = RateLimitedProvider(inner, limiter)

        msgs = [{"role": "user", "content": "hi"}]
        result = await provider.chat_stream(msgs, tools=None, model="m")

        inner.chat_stream.assert_awaited_once()
        assert result.content == "streamed"
        assert limiter._tokens < 10

    def test_get_default_model_delegates(self):
        inner = _make_provider_mock()
        limiter = TokenBucketLimiter(tokens_per_minute=60, burst=10)
        provider = RateLimitedProvider(inner, limiter)

        assert provider.get_default_model() == "test-model"
        inner.get_default_model.assert_called_once()
