"""Comprehensive tests for gateway modules: session_context, rate_limiter, hooks, router."""

from __future__ import annotations

import asyncio
import time
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from echo_agent.gateway.session_context import (
    clear_session_vars,
    get_all_session_vars,
    get_session_var,
    set_session_vars,
)
from echo_agent.gateway.rate_limiter import RateLimiter
from echo_agent.gateway.hooks import HookRegistry
from echo_agent.gateway.router import DeliveryRouter
from echo_agent.bus.events import OutboundEvent, ContentBlock, ContentType
from echo_agent.bus.queue import MessageBus


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_outbound(
    channel: str = "test",
    chat_id: str = "c1",
    metadata: dict[str, Any] | None = None,
) -> OutboundEvent:
    return OutboundEvent(
        channel=channel,
        chat_id=chat_id,
        content=[ContentBlock(type=ContentType.TEXT, text="hello")],
        metadata=metadata or {},
    )


# ===================================================================
# TestSessionContext
# ===================================================================

class TestSessionContext:
    """Tests for gateway/session_context.py context-var helpers."""

    def test_set_and_get_vars(self):
        tokens = set_session_vars(platform="qq", chat_id="12345")
        try:
            assert get_session_var("platform") == "qq"
            assert get_session_var("chat_id") == "12345"
        finally:
            clear_session_vars(tokens)

    def test_clear_vars_resets_to_default(self):
        tokens = set_session_vars(user_name="alice", user_id="u1")
        clear_session_vars(tokens)
        assert get_session_var("user_name") == ""
        assert get_session_var("user_id") == ""

    def test_get_all_session_vars(self):
        tokens = set_session_vars(platform="wechat", session_key="wechat:g1")
        try:
            result = get_all_session_vars()
            assert isinstance(result, dict)
            assert result["platform"] == "wechat"
            assert result["session_key"] == "wechat:g1"
            # Unset vars should be empty string
            assert result["thread_id"] == ""
        finally:
            clear_session_vars(tokens)

    def test_unknown_var_returns_default(self):
        assert get_session_var("nonexistent") == ""
        assert get_session_var("nonexistent", "fallback") == "fallback"

    def test_isolation_between_set_clear_cycles(self):
        tokens1 = set_session_vars(platform="discord", chat_id="a")
        clear_session_vars(tokens1)

        tokens2 = set_session_vars(platform="telegram", chat_id="b")
        try:
            assert get_session_var("platform") == "telegram"
            assert get_session_var("chat_id") == "b"
        finally:
            clear_session_vars(tokens2)

    def test_set_ignores_unknown_keys(self):
        tokens = set_session_vars(platform="qq", bogus_key="nope")
        try:
            assert get_session_var("platform") == "qq"
            assert get_session_var("bogus_key") == ""
        finally:
            clear_session_vars(tokens)


# ===================================================================
# TestGatewayRateLimiter
# ===================================================================

class TestGatewayRateLimiter:
    """Tests for gateway/rate_limiter.py token-bucket limiter."""

    def test_acquire_within_limit_returns_true(self):
        rl = RateLimiter(default_rpm=30)
        assert rl.acquire("test") is True

    def test_acquire_exhausted_returns_false(self):
        rl = RateLimiter(default_rpm=2)
        assert rl.acquire("test") is True
        assert rl.acquire("test") is True
        # Bucket capacity is 2, should be exhausted now
        assert rl.acquire("test") is False

    def test_configure_custom_platform_rpm(self):
        rl = RateLimiter(default_rpm=5)
        rl.configure("vip", 100)
        # vip platform should get a bucket with capacity 100
        for _ in range(50):
            assert rl.acquire("vip") is True
        # Default platform still has capacity 5
        for _ in range(5):
            assert rl.acquire("default") is True
        assert rl.acquire("default") is False

    def test_get_stats_shows_tokens_and_capacity(self):
        rl = RateLimiter(default_rpm=10)
        rl.acquire("plat")
        stats = rl.get_stats()
        assert "plat" in stats
        assert "tokens_available" in stats["plat"]
        assert "capacity" in stats["plat"]
        assert stats["plat"]["capacity"] == 10.0
        # After one acquire, tokens should be less than capacity
        assert stats["plat"]["tokens_available"] < 10.0

    def test_separate_buckets_per_platform_chat_id(self):
        rl = RateLimiter(default_rpm=2)
        assert rl.acquire("p", "chat_a") is True
        assert rl.acquire("p", "chat_a") is True
        assert rl.acquire("p", "chat_a") is False
        # Different chat_id should have its own bucket
        assert rl.acquire("p", "chat_b") is True

    @pytest.mark.asyncio
    async def test_wait_eventually_acquires(self):
        rl = RateLimiter(default_rpm=60)
        # Drain the bucket
        while rl.acquire("fast"):
            pass
        # Patch time to simulate refill
        original_time = time.time
        call_count = 0
        def advancing_time():
            nonlocal call_count
            call_count += 1
            return original_time() + call_count * 2
        with patch("echo_agent.gateway.rate_limiter.time.time", side_effect=advancing_time):
            await asyncio.wait_for(rl.wait("fast"), timeout=5.0)


# ===================================================================
# TestHookRegistry
# ===================================================================

class TestHookRegistry:
    """Tests for gateway/hooks.py event hook system."""

    @pytest.mark.asyncio
    async def test_register_and_emit(self):
        registry = HookRegistry()
        handler = AsyncMock()
        registry.register("message_received", handler)
        await registry.emit("message_received", text="hi")
        handler.assert_awaited_once_with(text="hi")

    @pytest.mark.asyncio
    async def test_emit_with_no_handlers_no_error(self):
        registry = HookRegistry()
        # Should not raise
        await registry.emit("message_sent", data="x")

    @pytest.mark.asyncio
    async def test_handler_exception_isolated(self):
        registry = HookRegistry()
        bad_handler = AsyncMock(side_effect=RuntimeError("boom"))
        good_handler = AsyncMock()
        registry.register("session_reset", bad_handler)
        registry.register("session_reset", good_handler)
        # Should not raise despite bad_handler failing
        await registry.emit("session_reset")
        good_handler.assert_awaited_once()

    def test_handler_count_property(self):
        registry = HookRegistry()
        assert registry.handler_count == 0
        registry.register("message_received", AsyncMock())
        registry.register("message_sent", AsyncMock())
        registry.register("message_received", AsyncMock())
        assert registry.handler_count == 3

    def test_load_from_dir_with_tmp_path(self, tmp_path: Path):
        hook_file = tmp_path / "my_hook.py"
        hook_file.write_text(
            "async def on_msg(**kw): pass\n"
            "def register_hooks(registry):\n"
            "    registry.register('message_received', on_msg)\n"
        )
        registry = HookRegistry()
        loaded = registry.load_from_dir(tmp_path)
        assert loaded == 1
        assert registry.handler_count == 1

    def test_load_from_dir_nonexistent(self, tmp_path: Path):
        registry = HookRegistry()
        loaded = registry.load_from_dir(tmp_path / "nope")
        assert loaded == 0

    def test_unknown_event_warning_still_registers(self):
        registry = HookRegistry()
        handler = AsyncMock()
        with patch("echo_agent.gateway.hooks.logger") as mock_logger:
            registry.register("totally_custom", handler)
            mock_logger.warning.assert_called_once()
        assert registry.handler_count == 1


# ===================================================================
# TestDeliveryRouter
# ===================================================================

class TestDeliveryRouter:
    """Tests for gateway/router.py delivery routing."""

    def _make_bus(self) -> MessageBus:
        return MessageBus()

    def test_add_rule_and_match(self):
        bus = self._make_bus()
        router = DeliveryRouter(bus)
        router.add_rule(lambda e: e.channel == "src", "dst", "chat_dst")
        assert router.rule_count == 1

    @pytest.mark.asyncio
    async def test_add_rule_routes_matching_event(self):
        bus = self._make_bus()
        router = DeliveryRouter(bus)
        router.add_rule(lambda e: e.channel == "src", "dst", "chat_dst")

        published: list[OutboundEvent] = []
        original_publish = bus.publish_outbound

        async def capture(event: OutboundEvent):
            published.append(event)
            # Don't recurse into global handlers for routed events
        bus.publish_outbound = capture

        event = _make_outbound(channel="src", chat_id="c1")
        await router._on_outbound(event)

        assert len(published) == 1
        assert published[0].channel == "dst"
        assert published[0].chat_id == "chat_dst"
        assert published[0].metadata.get("_routed") is True

    @pytest.mark.asyncio
    async def test_add_cron_route_exact_match(self):
        bus = self._make_bus()
        router = DeliveryRouter(bus)
        router.add_cron_route("cron:daily_report", "qq", "group1")

        published: list[OutboundEvent] = []
        bus.publish_outbound = AsyncMock(side_effect=lambda e: published.append(e))

        event = _make_outbound(channel="internal", metadata={"source_session_key": "cron:daily_report"})
        await router._on_outbound(event)
        assert len(published) == 1
        assert published[0].channel == "qq"

    @pytest.mark.asyncio
    async def test_add_cron_route_wildcard(self):
        bus = self._make_bus()
        router = DeliveryRouter(bus)
        router.add_cron_route("*", "telegram", "broadcast")

        published: list[OutboundEvent] = []
        bus.publish_outbound = AsyncMock(side_effect=lambda e: published.append(e))

        event = _make_outbound(channel="x", metadata={"source_session_key": "cron:anything"})
        await router._on_outbound(event)
        assert len(published) == 1
        assert published[0].chat_id == "broadcast"

    @pytest.mark.asyncio
    async def test_add_cron_route_wildcard_no_match_non_cron(self):
        bus = self._make_bus()
        router = DeliveryRouter(bus)
        router.add_cron_route("*", "telegram", "broadcast")

        published: list[OutboundEvent] = []
        bus.publish_outbound = AsyncMock(side_effect=lambda e: published.append(e))

        event = _make_outbound(channel="x", metadata={"source_session_key": "user:abc"})
        await router._on_outbound(event)
        assert len(published) == 0

    @pytest.mark.asyncio
    async def test_add_mirror_route(self):
        bus = self._make_bus()
        router = DeliveryRouter(bus)
        router.add_mirror_route("qq", "telegram", "tg_chat")

        published: list[OutboundEvent] = []
        bus.publish_outbound = AsyncMock(side_effect=lambda e: published.append(e))

        event = _make_outbound(channel="qq", chat_id="qq_chat")
        await router._on_outbound(event)
        assert len(published) == 1
        assert published[0].channel == "telegram"
        assert published[0].chat_id == "tg_chat"

    @pytest.mark.asyncio
    async def test_routed_flag_prevents_rerouting(self):
        bus = self._make_bus()
        router = DeliveryRouter(bus)
        router.add_rule(lambda e: True, "dst", "c")

        published: list[OutboundEvent] = []
        bus.publish_outbound = AsyncMock(side_effect=lambda e: published.append(e))

        event = _make_outbound(channel="src", metadata={"_routed": True})
        await router._on_outbound(event)
        assert len(published) == 0

    def test_rule_count_property(self):
        bus = self._make_bus()
        router = DeliveryRouter(bus)
        assert router.rule_count == 0
        router.add_rule(lambda e: True, "a", "b")
        router.add_cron_route("cron:x", "c", "d")
        router.add_mirror_route("e", "f", "g")
        assert router.rule_count == 3

    @pytest.mark.asyncio
    async def test_predicate_exception_skipped(self):
        bus = self._make_bus()
        router = DeliveryRouter(bus)

        def bad_pred(e):
            raise ValueError("oops")

        good_published: list[OutboundEvent] = []
        bus.publish_outbound = AsyncMock(side_effect=lambda e: good_published.append(e))

        router.add_rule(bad_pred, "bad_dst", "c1")
        router.add_rule(lambda e: True, "good_dst", "c2")

        event = _make_outbound(channel="src")
        await router._on_outbound(event)
        # Bad predicate skipped, good rule still fires
        assert len(good_published) == 1
        assert good_published[0].channel == "good_dst"

