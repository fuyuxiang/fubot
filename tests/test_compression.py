"""Comprehensive tests for echo_agent.agent.compression (all 8 modules)."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from echo_agent.agent.compression.types import (
    BoundaryResult,
    CompressionResult,
    CompressionStats,
    PruneResult,
)
from echo_agent.agent.compression.engine import ContextEngine
from echo_agent.agent.compression.boundary import BoundaryResolver
from echo_agent.agent.compression.pruner import ToolOutputPruner
from echo_agent.agent.compression.summarizer import LLMSummarizer
from echo_agent.agent.compression.assembler import MessageAssembler
from echo_agent.agent.compression.validator import MessageValidator
from echo_agent.agent.compression.compressor import ConversationCompressor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tok_estimator(msgs: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content", ""))) // 4 + 4 for m in msgs)


def _msg(role: str, content: str = "", **kw: Any) -> dict[str, Any]:
    m: dict[str, Any] = {"role": role, "content": content}
    m.update(kw)
    return m


def _tool_msg(name: str, call_id: str, content: str = "ok") -> dict[str, Any]:
    return {"role": "tool", "name": name, "tool_call_id": call_id, "content": content}


def _assistant_with_calls(*calls: tuple[str, str, str]) -> dict[str, Any]:
    tcs = [{"id": cid, "function": {"name": n, "arguments": a}} for cid, n, a in calls]
    return {"role": "assistant", "content": "", "tool_calls": tcs}


class _ConcreteEngine(ContextEngine):
    async def compress(self, messages, focus_topic=""):
        return CompressionResult(messages=messages, stats=self._stats)


def _make_config(**overrides: Any) -> MagicMock:
    defaults = dict(
        enabled=True,
        trigger_ratio=0.7,
        tool_pruning_enabled=True,
        tool_pruning_tail_budget_ratio=0.3,
        head_protect_count=2,
        tail_budget_ratio=0.3,
        summary_model="",
        summary_target_ratio=0.3,
        summary_min_tokens=100,
        summary_max_tokens=2000,
        summary_cooldown_seconds=60,
        max_compression_count=10,
    )
    defaults.update(overrides)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg

# ---------------------------------------------------------------------------
# 1. types.py — dataclass defaults
# ---------------------------------------------------------------------------

class TestCompressionStats:
    def test_defaults(self):
        s = CompressionStats()
        assert s.compression_count == 0
        assert s.last_compressed_at is None
        assert s.last_summary_failure_at is None
        assert s.total_tokens_saved == 0
        assert s.warnings == []

    def test_warnings_independent(self):
        a, b = CompressionStats(), CompressionStats()
        a.warnings.append("x")
        assert b.warnings == []


class TestBoundaryResult:
    def test_defaults(self):
        r = BoundaryResult(head_end=0, tail_start=0, head_messages=[], middle_messages=[], tail_messages=[])
        assert r.no_compression_needed is False

    def test_no_compression_flag(self):
        r = BoundaryResult(head_end=0, tail_start=0, head_messages=[], middle_messages=[], tail_messages=[], no_compression_needed=True)
        assert r.no_compression_needed is True


class TestPruneResult:
    def test_fields(self):
        r = PruneResult(messages=[{"role": "user"}], pruned_count=3)
        assert r.pruned_count == 3
        assert len(r.messages) == 1


class TestCompressionResult:
    def test_defaults(self):
        r = CompressionResult(messages=[], stats=CompressionStats())
        assert r.summary_text is None
        assert r.was_compressed is False
        assert r.tokens_before == 0
        assert r.tokens_after == 0

# ---------------------------------------------------------------------------
# 2. engine.py — ContextEngine (via concrete subclass)
# ---------------------------------------------------------------------------

class TestContextEngine:
    def _engine(self, window=1000, ratio=0.7):
        return _ConcreteEngine(context_window_tokens=window, trigger_ratio=ratio)

    def test_should_compress_below_threshold(self):
        e = self._engine(window=10000)
        msgs = [_msg("user", "hi")]
        assert e.should_compress(msgs) is False

    def test_should_compress_above_threshold(self):
        e = self._engine(window=100, ratio=0.1)
        msgs = [_msg("user", "x" * 500)]
        assert e.should_compress(msgs) is True

    def test_estimate_tokens_no_counter(self):
        e = self._engine()
        msgs = [_msg("user", "a" * 40)]  # 40//4 + 4 = 14
        assert e.estimate_tokens(msgs) == 14

    def test_estimate_tokens_with_counter(self):
        e = self._engine()
        counter = MagicMock()
        counter.count = MagicMock(return_value=10)
        e.set_token_counter(counter)
        msgs = [_msg("user", "hello")]
        assert e.estimate_tokens(msgs) == 14  # 10 + 4

    def test_estimate_list_content_no_counter(self):
        e = self._engine()
        msgs = [{"role": "user", "content": [{"text": "a" * 20}]}]
        assert e.estimate_tokens(msgs) == 4 + 20 // 4  # 9

    def test_estimate_list_content_with_counter(self):
        e = self._engine()
        counter = MagicMock()
        counter.count = MagicMock(return_value=7)
        e.set_token_counter(counter)
        msgs = [{"role": "user", "content": [{"text": "hello"}]}]
        assert e.estimate_tokens(msgs) == 4 + 7  # 11

    def test_estimate_tool_calls(self):
        e = self._engine()
        msg = _assistant_with_calls(("c1", "read_file", '{"path":"/a"}'))
        assert e.estimate_tokens([msg]) > 4

    def test_estimate_tool_calls_dict_args(self):
        e = self._engine()
        msg = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "run", "arguments": {"cmd": "ls"}}}
        ]}
        tokens = e.estimate_tokens([msg])
        assert tokens > 4

    def test_get_stats(self):
        e = self._engine()
        assert isinstance(e.get_stats(), CompressionStats)

    def test_session_lifecycle(self):
        e = self._engine()
        e.get_stats().compression_count = 5
        e.on_session_start("s1")
        assert e.get_stats().compression_count == 0
        e.get_stats().compression_count = 3
        e.on_session_reset("s1")
        assert e.get_stats().compression_count == 0
        e.on_session_end("s1")

# ---------------------------------------------------------------------------
# 3. boundary.py — BoundaryResolver
# ---------------------------------------------------------------------------

class TestBoundaryResolver:
    def _resolver(self, head=2, tail_ratio=0.3, window=10000):
        return BoundaryResolver(head, tail_ratio, window, _tok_estimator)

    def test_empty_messages(self):
        r = self._resolver().resolve([])
        assert r.no_compression_needed is True
        assert r.head_messages == []

    def test_short_conversation_no_compression(self):
        msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
        r = self._resolver(head=2, tail_ratio=0.9, window=10000).resolve(msgs)
        assert r.no_compression_needed is True

    def test_head_aligned_past_tool_results(self):
        msgs = [
            _msg("user", "go"),
            _assistant_with_calls(("c1", "run", "{}")),
            _tool_msg("run", "c1", "output"),
            _msg("user", "next"),
            _msg("assistant", "a" * 4000),
            _msg("user", "more"),
            _msg("assistant", "b" * 4000),
        ]
        r = BoundaryResolver(2, 0.05, 500, _tok_estimator).resolve(msgs)
        assert r.head_end >= 3  # pushed past tool result at index 2

    def test_tail_contains_last_user(self):
        msgs = [_msg("user", "a")] + [_msg("assistant", "x" * 200) for _ in range(10)] + [_msg("user", "last")]
        r = BoundaryResolver(1, 0.05, 2000, _tok_estimator).resolve(msgs)
        if not r.no_compression_needed:
            roles = [m["role"] for m in r.tail_messages]
            assert "user" in roles

    def test_normal_split(self):
        msgs = (
            [_msg("user", "start"), _msg("assistant", "ok")]
            + [_msg("user", "x" * 200), _msg("assistant", "y" * 200)] * 5
            + [_msg("user", "end"), _msg("assistant", "done")]
        )
        r = BoundaryResolver(2, 0.3, 800, _tok_estimator).resolve(msgs)
        if not r.no_compression_needed:
            assert len(r.head_messages) >= 2
            assert len(r.tail_messages) >= 1
            assert len(r.middle_messages) > 0

# ---------------------------------------------------------------------------
# 4. pruner.py — ToolOutputPruner
# ---------------------------------------------------------------------------

class TestToolOutputPruner:
    def _pruner(self, tail_ratio=0.1, window=100000):
        return ToolOutputPruner(tail_ratio, window, _tok_estimator)

    def test_empty(self):
        r = self._pruner().prune([])
        assert r.pruned_count == 0

    def test_tool_result_pruned_to_summary(self):
        msgs = [
            _msg("user", "go"),
            _assistant_with_calls(("c1", "read_file", "{}")),
            _tool_msg("read_file", "c1", "file content here\nline2\nline3"),
            _msg("user", "thanks"),
            _msg("assistant", "x" * 50000),  # push tail far
        ]
        r = self._pruner(tail_ratio=0.01, window=100000).prune(msgs)
        tool_msgs = [m for m in r.messages if m.get("role") == "tool"]
        assert any("[pruned]" in m.get("content", "") for m in tool_msgs)

    def test_build_summary_terminal(self):
        p = self._pruner()
        s = p._build_summary("terminal", "line1\nline2\nline3")
        assert "3 lines" in s
        assert "[pruned]" in s

    def test_build_summary_read_file(self):
        p = self._pruner()
        s = p._build_summary("read_file", "abc\ndef")
        assert "chars" in s and "lines" in s

    def test_build_summary_search(self):
        p = self._pruner()
        s = p._build_summary("search_files", "m1\nm2\nm3")
        assert "2 matches" in s  # count("\n") == 2

    def test_build_summary_write_file(self):
        p = self._pruner()
        s = p._build_summary("write_file", "written content")
        assert "file operation" in s

    def test_build_summary_web(self):
        p = self._pruner()
        s = p._build_summary("web_fetch", "html content")
        assert "web result" in s

    def test_build_summary_default(self):
        p = self._pruner()
        s = p._build_summary("custom_tool", "short result")
        assert "short result" in s

    def test_build_summary_default_truncates(self):
        p = self._pruner()
        s = p._build_summary("custom_tool", "a" * 200)
        assert "..." in s

    def test_build_summary_empty(self):
        p = self._pruner()
        s = p._build_summary("tool", "")
        assert "empty result" in s

    def test_truncate_long_tool_call_args(self):
        p = self._pruner()
        long_args = "x" * 2000
        msgs = [
            _msg("user", "go"),
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "run", "arguments": long_args}}
            ]},
            _tool_msg("run", "c1", "ok"),
            _msg("user", "end" * 20000),  # push tail
        ]
        r = p.prune(msgs)
        for m in r.messages:
            for tc in m.get("tool_calls", []):
                args = tc["function"]["arguments"]
                assert len(args) <= 1300  # 1200 + '..."}'

    def test_dedup_placeholder(self):
        p = self._pruner()
        msgs = [
            _msg("user", "go"),
            _assistant_with_calls(("c1", "run", "{}")),
            _tool_msg("run", "c1", "first"),
            _assistant_with_calls(("c1", "run", "{}")),
            _tool_msg("run", "c1", "second"),
            _msg("user", "end" * 20000),
        ]
        r = p.prune(msgs)
        contents = [m.get("content", "") for m in r.messages if m.get("role") == "tool"]
        assert any("duplicate" in c for c in contents) or any("[pruned]" in c for c in contents)

# ---------------------------------------------------------------------------
# 5. summarizer.py — LLMSummarizer
# ---------------------------------------------------------------------------

class TestLLMSummarizer:
    def _summarizer(self, cooldown=60):
        provider = AsyncMock()
        return LLMSummarizer(
            provider=provider,
            summary_model="test-model",
            default_model="default-model",
            summary_target_ratio=0.3,
            summary_min_tokens=100,
            summary_max_tokens=2000,
            cooldown_seconds=cooldown,
        ), provider

    @pytest.mark.asyncio
    async def test_empty_returns_none(self):
        s, _ = self._summarizer()
        result = await s.summarize([], "", CompressionStats(), _tok_estimator)
        assert result is None

    @pytest.mark.asyncio
    async def test_cooldown_returns_none(self):
        s, _ = self._summarizer(cooldown=9999)
        stats = CompressionStats(last_summary_failure_at=time.time())
        result = await s.summarize([_msg("user", "hi")], "", stats, _tok_estimator)
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_summary(self):
        s, provider = self._summarizer()
        resp = MagicMock()
        resp.content = "  summary text  "
        resp.finish_reason = "stop"
        provider.chat_with_retry = AsyncMock(return_value=resp)
        result = await s.summarize([_msg("user", "hello")], "topic", CompressionStats(), _tok_estimator)
        assert result == "summary text"

    @pytest.mark.asyncio
    async def test_error_finish_reason(self):
        s, provider = self._summarizer()
        resp = MagicMock()
        resp.content = "error"
        resp.finish_reason = "error"
        provider.chat_with_retry = AsyncMock(return_value=resp)
        stats = CompressionStats()
        result = await s.summarize([_msg("user", "hi")], "", stats, _tok_estimator)
        assert result is None
        assert stats.last_summary_failure_at is not None

    @pytest.mark.asyncio
    async def test_exception_sets_failure(self):
        s, provider = self._summarizer()
        provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("boom"))
        stats = CompressionStats()
        result = await s.summarize([_msg("user", "hi")], "", stats, _tok_estimator)
        assert result is None
        assert stats.last_summary_failure_at is not None

    @pytest.mark.asyncio
    async def test_empty_content_returns_none(self):
        s, provider = self._summarizer()
        resp = MagicMock()
        resp.content = ""
        resp.finish_reason = "stop"
        provider.chat_with_retry = AsyncMock(return_value=resp)
        stats = CompressionStats()
        result = await s.summarize([_msg("user", "hi")], "", stats, _tok_estimator)
        assert result is None

    def test_compute_budget_clamped(self):
        s, _ = self._summarizer()
        assert s._compute_budget(10) == 100  # min
        assert s._compute_budget(100000) == 2000  # max
        assert s._compute_budget(1000) == 300  # 1000 * 0.3

    def test_serialize_format(self):
        s, _ = self._summarizer()
        msgs = [_msg("user", "hello"), _msg("assistant", "world")]
        text = s._serialize(msgs)
        assert "[T1][USER]" in text
        assert "[T1][ASSISTANT]" in text

    def test_serialize_tool_calls(self):
        s, _ = self._summarizer()
        msgs = [_msg("user", "go"), _assistant_with_calls(("c1", "run", '{"cmd":"ls"}')),
                _tool_msg("run", "c1", "output")]
        text = s._serialize(msgs)
        assert "TOOL_CALLS" in text
        assert "TOOL:run" in text

    def test_truncate_content_short(self):
        s, _ = self._summarizer()
        assert s._truncate_content("short") == "short"

    def test_truncate_content_long(self):
        s, _ = self._summarizer()
        long_text = "x" * 10000
        result = s._truncate_content(long_text)
        assert "omitted" in result
        assert len(result) < len(long_text)

    def test_truncate_content_list(self):
        s, _ = self._summarizer()
        result = s._truncate_content([{"text": "hello"}, {"text": "world"}])
        assert "hello" in result

# ---------------------------------------------------------------------------
# 6. assembler.py — MessageAssembler
# ---------------------------------------------------------------------------

class TestMessageAssembler:
    def test_no_summary(self):
        a = MessageAssembler()
        head = [_msg("user", "hi")]
        tail = [_msg("user", "bye")]
        result = a.assemble(head, tail, None)
        assert len(result) == 2
        assert result[0]["content"] == "hi"
        assert result[1]["content"] == "bye"

    def test_with_summary(self):
        a = MessageAssembler()
        head = [_msg("user", "hi")]
        tail = [_msg("user", "bye")]
        result = a.assemble(head, tail, "the summary")
        assert len(result) == 4
        assert result[1]["role"] == "user"
        assert "Conversation Summary" in result[1]["content"]
        assert "the summary" in result[1]["content"]
        assert result[2]["role"] == "assistant"
        assert "context" in result[2]["content"].lower()

    def test_empty_summary_string(self):
        a = MessageAssembler()
        result = a.assemble([_msg("user", "a")], [_msg("user", "b")], "")
        assert len(result) == 2  # empty string is falsy

# ---------------------------------------------------------------------------
# 7. validator.py — MessageValidator
# ---------------------------------------------------------------------------

class TestMessageValidator:
    def _v(self):
        return MessageValidator()

    def test_remove_leading_tool_results(self):
        msgs = [
            _tool_msg("run", "c1"),
            _tool_msg("run", "c2"),
            _msg("user", "hi"),
        ]
        result = self._v().validate(msgs)
        assert result[0]["role"] == "user"

    def test_all_tool_messages_returns_empty(self):
        msgs = [_tool_msg("a", "c1"), _tool_msg("b", "c2")]
        result = self._v().validate(msgs)
        assert result == []

    def test_remove_orphan_tool_results(self):
        msgs = [
            _msg("user", "hi"),
            _assistant_with_calls(("c1", "run", "{}")),
            _tool_msg("run", "c1", "ok"),
            _tool_msg("run", "orphan_id", "stale"),
        ]
        result = self._v().validate(msgs)
        call_ids = [m.get("tool_call_id") for m in result if m.get("role") == "tool"]
        assert "orphan_id" not in call_ids
        assert "c1" in call_ids

    def test_patch_missing_tool_results(self):
        msgs = [
            _msg("user", "hi"),
            _assistant_with_calls(("c1", "run", "{}"), ("c2", "read", "{}")),
            _tool_msg("run", "c1", "ok"),
            # c2 result missing
            _msg("user", "next"),
        ]
        result = self._v().validate(msgs)
        tool_ids = [m.get("tool_call_id") for m in result if m.get("role") == "tool"]
        assert "c2" in tool_ids
        patched = [m for m in result if m.get("tool_call_id") == "c2"]
        assert "lost during context compression" in patched[0]["content"]

    def test_no_changes_needed(self):
        msgs = [
            _msg("user", "hi"),
            _assistant_with_calls(("c1", "run", "{}")),
            _tool_msg("run", "c1", "ok"),
            _msg("assistant", "done"),
        ]
        result = self._v().validate(msgs)
        assert len(result) == 4

    def test_empty_input(self):
        assert self._v().validate([]) == []

# ---------------------------------------------------------------------------
# 8. compressor.py — ConversationCompressor
# ---------------------------------------------------------------------------

class TestConversationCompressor:
    def _compressor(self, **cfg_overrides):
        cfg = _make_config(**cfg_overrides)
        provider = AsyncMock()
        return ConversationCompressor(
            config=cfg,
            context_window_tokens=10000,
            provider=provider,
            default_model="test-model",
        ), provider

    def test_should_compress_disabled(self):
        c, _ = self._compressor(enabled=False)
        msgs = [_msg("user", "x" * 50000)]
        assert c.should_compress(msgs) is False

    def test_should_compress_enabled_below(self):
        c, _ = self._compressor()
        msgs = [_msg("user", "hi")]
        assert c.should_compress(msgs) is False

    def test_should_compress_enabled_above(self):
        c, _ = self._compressor(trigger_ratio=0.01)
        msgs = [_msg("user", "x" * 500)]
        assert c.should_compress(msgs) is True

    @pytest.mark.asyncio
    async def test_compress_no_compression_needed(self):
        c, _ = self._compressor(head_protect_count=100, tail_budget_ratio=0.9)
        msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
        result = await c.compress(msgs)
        assert result.was_compressed is False

    @pytest.mark.asyncio
    async def test_compress_full_pipeline(self):
        c, provider = self._compressor(head_protect_count=1, tail_budget_ratio=0.05)
        resp = MagicMock()
        resp.content = "summary of conversation"
        resp.finish_reason = "stop"
        provider.chat_with_retry = AsyncMock(return_value=resp)

        msgs = (
            [_msg("user", "start"), _msg("assistant", "ok")]
            + [_msg("user", "q" * 300), _msg("assistant", "a" * 300)] * 5
            + [_msg("user", "final"), _msg("assistant", "end")]
        )
        result = await c.compress(msgs, focus_topic="testing")
        assert result.was_compressed is True
        assert result.tokens_before > 0
        assert result.stats.compression_count == 1
        assert result.stats.last_compressed_at is not None

    @pytest.mark.asyncio
    async def test_compress_without_pruner(self):
        c, provider = self._compressor(
            tool_pruning_enabled=False,
            head_protect_count=1,
            tail_budget_ratio=0.05,
        )
        resp = MagicMock()
        resp.content = "summary"
        resp.finish_reason = "stop"
        provider.chat_with_retry = AsyncMock(return_value=resp)

        msgs = (
            [_msg("user", "start")]
            + [_msg("assistant", "a" * 300), _msg("user", "q" * 300)] * 5
            + [_msg("assistant", "end")]
        )
        result = await c.compress(msgs)
        assert result.was_compressed is True

    @pytest.mark.asyncio
    async def test_compress_summary_failure_still_compresses(self):
        c, provider = self._compressor(head_protect_count=1, tail_budget_ratio=0.05)
        provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("fail"))

        msgs = (
            [_msg("user", "start"), _msg("assistant", "ok")]
            + [_msg("user", "q" * 300), _msg("assistant", "a" * 300)] * 5
            + [_msg("user", "final"), _msg("assistant", "end")]
        )
        result = await c.compress(msgs)
        assert result.was_compressed is True
        assert result.summary_text is None

    @pytest.mark.asyncio
    async def test_max_compression_warning(self):
        c, provider = self._compressor(
            head_protect_count=1,
            tail_budget_ratio=0.05,
            max_compression_count=1,
        )
        resp = MagicMock()
        resp.content = "summary"
        resp.finish_reason = "stop"
        provider.chat_with_retry = AsyncMock(return_value=resp)

        msgs = (
            [_msg("user", "start")]
            + [_msg("assistant", "a" * 300), _msg("user", "q" * 300)] * 5
            + [_msg("assistant", "end")]
        )
        result = await c.compress(msgs)
        assert result.was_compressed is True
        assert len(result.stats.warnings) >= 1
        assert "degrading" in result.stats.warnings[0]

    @pytest.mark.asyncio
    async def test_tokens_saved_tracked(self):
        c, provider = self._compressor(head_protect_count=1, tail_budget_ratio=0.05)
        resp = MagicMock()
        resp.content = "short"
        resp.finish_reason = "stop"
        provider.chat_with_retry = AsyncMock(return_value=resp)

        msgs = (
            [_msg("user", "start")]
            + [_msg("assistant", "a" * 300), _msg("user", "q" * 300)] * 5
            + [_msg("assistant", "end")]
        )
        result = await c.compress(msgs)
        assert result.stats.total_tokens_saved >= 0
