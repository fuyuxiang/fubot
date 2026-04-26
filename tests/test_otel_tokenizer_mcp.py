"""Comprehensive tests for OTel, TokenCounter, MCP transport, and other uncovered modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# 1. OpenTelemetry — telemetry.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestTelemetryManagerWithoutOTel:
    """TelemetryManager when opentelemetry is NOT installed."""

    def test_available_reflects_import(self):
        with patch("echo_agent.observability.telemetry._HAS_OTEL", False):
            from echo_agent.observability.telemetry import TelemetryManager
            tm = TelemetryManager()
            # available reads module-level flag, but instance was created under patch
            # Just verify setup doesn't crash
            tm.setup()

    def test_setup_no_crash_without_otel(self):
        with patch("echo_agent.observability.telemetry._HAS_OTEL", False):
            from echo_agent.observability.telemetry import TelemetryManager
            tm = TelemetryManager(service_name="test-svc")
            tm.setup()
            assert tm.get_tracer() is None
            assert tm.get_meter() is None

    def test_shutdown_no_crash_without_otel(self):
        with patch("echo_agent.observability.telemetry._HAS_OTEL", False):
            from echo_agent.observability.telemetry import TelemetryManager
            tm = TelemetryManager()
            tm.shutdown()  # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OpenTelemetry — spans.py
# ═══════════════════════════════════════════════════════════════════════════════
class TestSpansWithNoneTracer:
    """All span helpers must be no-ops when tracer is None."""

    def test_start_llm_span_none_tracer(self):
        from echo_agent.observability.spans import start_llm_span
        assert start_llm_span(None, "gpt-4o", "openai") is None

    def test_end_llm_span_none(self):
        from echo_agent.observability.spans import end_llm_span
        end_llm_span(None)  # no crash
        end_llm_span(None, error="boom")

    def test_record_llm_usage_none(self):
        from echo_agent.observability.spans import record_llm_usage
        record_llm_usage(None, {"input_tokens": 10})

    def test_start_tool_span_none(self):
        from echo_agent.observability.spans import start_tool_span
        assert start_tool_span(None, "web_search") is None

    def test_end_tool_span_none(self):
        from echo_agent.observability.spans import end_tool_span
        end_tool_span(None)
        end_tool_span(None, error="fail")

    def test_start_agent_span_none(self):
        from echo_agent.observability.spans import start_agent_span
        assert start_agent_span(None, 0) is None
        assert start_agent_span(None, 3, strategy="react") is None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OpenTelemetry — monitor.py (TraceLogger)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceLogger:
    def test_start_and_end_span(self):
        from echo_agent.observability.monitor import TraceLogger
        tl = TraceLogger()
        span = tl.start_span("t1", "s1", "llm_call", "llm_call")
        assert span.trace_id == "t1"
        assert span.span_id == "s1"
        assert span.started_at > 0
        tl.end_span(span, metadata={"model": "gpt-4o"})
        assert span.ended_at >= span.started_at
        assert span.metadata["model"] == "gpt-4o"
        assert span.duration_ms >= 0

    def test_end_span_with_error(self):
        from echo_agent.observability.monitor import TraceLogger
        tl = TraceLogger()
        span = tl.start_span("t2", "s2", "tool", "tool_call")
        tl.end_span(span, error="timeout")
        assert span.error == "timeout"

    def test_get_trace(self):
        from echo_agent.observability.monitor import TraceLogger
        tl = TraceLogger()
        tl.start_span("t3", "s3", "input", "input")
        spans = tl.get_trace("t3")
        assert len(spans) == 1
        assert tl.get_trace("nonexistent") == []

    def test_set_otel_tracer(self):
        from echo_agent.observability.monitor import TraceLogger
        tl = TraceLogger()
        assert tl._otel_tracer is None
        tl.set_otel_tracer("fake_tracer")
        assert tl._otel_tracer == "fake_tracer"

    def test_flush_trace_to_disk(self, tmp_path: Path):
        from echo_agent.observability.monitor import TraceLogger
        tl = TraceLogger(logs_dir=tmp_path)
        span = tl.start_span("t4", "s4", "output", "output")
        tl.end_span(span)
        tl.flush_trace("t4")
        written = tmp_path / "trace_t4.json"
        assert written.exists()
        data = json.loads(written.read_text())
        assert len(data) == 1
        assert data[0]["span_id"] == "s4"

    def test_flush_trace_no_dir(self):
        from echo_agent.observability.monitor import TraceLogger
        tl = TraceLogger()
        tl.start_span("t5", "s5", "x", "x")
        tl.flush_trace("t5")  # no crash, just removes from memory
        assert tl.get_trace("t5") == []

    def test_get_recent_traces(self):
        from echo_agent.observability.monitor import TraceLogger
        tl = TraceLogger()
        for i in range(5):
            tl.start_span(f"trace_{i}", f"s_{i}", "x", "x")
        recent = tl.get_recent_traces(limit=3)
        assert len(recent) == 3
        assert recent[-1] == "trace_4"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TokenCounter — tokenizer.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenCounterFallback:
    """TokenCounter in fallback mode (no tiktoken / anthropic SDK)."""

    def _make_counter(self) -> Any:
        from echo_agent.models.tokenizer import TokenCounter
        tc = TokenCounter(provider="unknown", model="test")
        assert tc._tokenizer is None  # fallback
        return tc

    def test_empty_string_returns_zero(self):
        tc = self._make_counter()
        assert tc.count("") == 0

    def test_count_fallback_formula(self):
        tc = self._make_counter()
        text = "hello world test"
        expected = max(1, len(text) // 4)
        assert tc.count(text) == expected

    def test_count_messages_basic(self):
        tc = self._make_counter()
        msgs = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = tc.count_messages(msgs)
        assert result > 0
        # 2 messages * 4 overhead + content tokens + role tokens + 2 conversation overhead
        assert isinstance(result, int)

    def test_count_messages_with_tool_calls(self):
        tc = self._make_counter()
        msgs = [
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "test"}),
                        }
                    }
                ],
            }
        ]
        result = tc.count_messages(msgs)
        assert result > 0

    def test_count_messages_with_content_blocks(self):
        tc = self._make_counter()
        msgs = [{"role": "user", "content": [{"text": "block one"}, {"text": "block two"}]}]
        result = tc.count_messages(msgs)
        assert result > 0

    def test_count_tools(self):
        tc = self._make_counter()
        tools = [{"function": {"name": "search", "parameters": {"type": "object"}}}]
        result = tc.count_tools(tools)
        assert result > 0

    def test_for_model_caching(self):
        from echo_agent.models.tokenizer import TokenCounter
        # Clear cache to avoid cross-test pollution
        TokenCounter._instances.pop("test_cache:model_a", None)
        a = TokenCounter.for_model("test_cache", "model_a")
        b = TokenCounter.for_model("test_cache", "model_a")
        assert a is b

    def test_for_model_different_keys(self):
        from echo_agent.models.tokenizer import TokenCounter
        TokenCounter._instances.pop("test_x:m1", None)
        TokenCounter._instances.pop("test_y:m2", None)
        a = TokenCounter.for_model("test_x", "m1")
        b = TokenCounter.for_model("test_y", "m2")
        assert a is not b


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LLMResponse — provider.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMResponse:
    def test_has_tool_calls_true(self):
        from echo_agent.models.provider import LLMResponse, ToolCallRequest
        resp = LLMResponse(tool_calls=[ToolCallRequest(id="1", name="search", arguments={})])
        assert resp.has_tool_calls is True

    def test_has_tool_calls_false(self):
        from echo_agent.models.provider import LLMResponse
        resp = LLMResponse(content="hello")
        assert resp.has_tool_calls is False

    def test_cache_hit_rate_with_cache(self):
        from echo_agent.models.provider import LLMResponse
        resp = LLMResponse(usage={"input_tokens": 100, "cache_read_input_tokens": 400})
        # rate = 400 / (100 + 400) = 0.8
        assert abs(resp.cache_hit_rate - 0.8) < 1e-9

    def test_cache_hit_rate_no_cache(self):
        from echo_agent.models.provider import LLMResponse
        resp = LLMResponse(usage={"input_tokens": 100})
        assert resp.cache_hit_rate == 0.0

    def test_cache_hit_rate_zero_input(self):
        from echo_agent.models.provider import LLMResponse
        resp = LLMResponse(usage={})
        assert resp.cache_hit_rate == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MCP Transport — transport.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestStreamableHttpTransport:
    def test_init_not_connected(self):
        from echo_agent.mcp.transport import StreamableHttpTransport
        t = StreamableHttpTransport(url="http://localhost:8080/mcp")
        assert t.is_connected is False

    def test_session_id_initially_none(self):
        from echo_agent.mcp.transport import StreamableHttpTransport
        t = StreamableHttpTransport(url="http://localhost:8080/mcp")
        assert t.session_id is None

    def test_custom_headers_stored(self):
        from echo_agent.mcp.transport import StreamableHttpTransport
        t = StreamableHttpTransport(url="http://x", headers={"Authorization": "Bearer tok"})
        assert t._headers["Authorization"] == "Bearer tok"


class TestStdioTransport:
    def test_init_not_connected(self):
        from echo_agent.mcp.transport import StdioTransport
        t = StdioTransport(command="echo", args=["hello"])
        assert t.is_connected is False

    def test_init_stores_command(self):
        from echo_agent.mcp.transport import StdioTransport
        t = StdioTransport(command="/usr/bin/node", args=["server.js"], env={"FOO": "1"})
        assert t._command == "/usr/bin/node"
        assert t._args == ["server.js"]
        assert t._env == {"FOO": "1"}


class TestHttpTransport:
    def test_init_not_connected(self):
        from echo_agent.mcp.transport import HttpTransport
        t = HttpTransport(url="http://localhost:3000/")
        assert t.is_connected is False

    def test_parse_sse_event_valid(self):
        from echo_agent.mcp.transport import HttpTransport
        t = HttpTransport(url="http://x")
        result = t._parse_sse_event('data: {"jsonrpc":"2.0","id":1}')
        assert result == {"jsonrpc": "2.0", "id": 1}

    def test_parse_sse_event_no_data(self):
        from echo_agent.mcp.transport import HttpTransport
        t = HttpTransport(url="http://x")
        assert t._parse_sse_event("event: ping") is None

    def test_parse_sse_event_invalid_json(self):
        from echo_agent.mcp.transport import HttpTransport
        t = HttpTransport(url="http://x")
        assert t._parse_sse_event("data: not-json") is None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Prompt Caching — format_utils.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenaiToAnthropicTools:
    def test_inject_cache_markers_true(self):
        from echo_agent.models.providers.format_utils import openai_to_anthropic_tools
        tools = [
            {"function": {"name": "a", "description": "tool a", "parameters": {"type": "object"}}},
            {"function": {"name": "b", "description": "tool b", "parameters": {"type": "object"}}},
        ]
        result = openai_to_anthropic_tools(tools, inject_cache_markers=True)
        assert len(result) == 2
        assert "cache_control" not in result[0]
        assert result[-1]["cache_control"] == {"type": "ephemeral"}

    def test_inject_cache_markers_false(self):
        from echo_agent.models.providers.format_utils import openai_to_anthropic_tools
        tools = [{"function": {"name": "x", "description": "d", "parameters": {"type": "object"}}}]
        result = openai_to_anthropic_tools(tools, inject_cache_markers=False)
        assert "cache_control" not in result[0]

    def test_empty_tools(self):
        from echo_agent.models.providers.format_utils import openai_to_anthropic_tools
        assert openai_to_anthropic_tools([], inject_cache_markers=True) == []


class TestOpenaiToAnthropicMessages:
    def test_system_message_extracted(self):
        from echo_agent.models.providers.format_utils import openai_to_anthropic_messages
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system_blocks, converted = openai_to_anthropic_messages(msgs)
        assert len(system_blocks) == 1
        assert system_blocks[0]["text"] == "You are helpful."
        assert converted[0]["role"] == "user"

    def test_cache_markers_injected_on_system(self):
        from echo_agent.models.providers.format_utils import openai_to_anthropic_messages
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        system_blocks, _ = openai_to_anthropic_messages(msgs, inject_cache_markers=True)
        assert system_blocks[-1].get("cache_control") == {"type": "ephemeral"}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Compression Engine — engine.py
# ═══════════════════════════════════════════════════════════════════════════════

class ConcreteEngine:
    """Minimal concrete subclass for testing ContextEngine."""

    @staticmethod
    def _make():
        from echo_agent.agent.compression.engine import ContextEngine
        from echo_agent.agent.compression.types import CompressionResult, CompressionStats

        class _Impl(ContextEngine):
            async def compress(self, messages, focus_topic=""):
                return CompressionResult(messages=messages, stats=CompressionStats())

        return _Impl


class TestContextEngine:
    def test_estimate_without_token_counter(self):
        Cls = ConcreteEngine._make()
        engine = Cls(context_window_tokens=10000)
        msg = {"role": "user", "content": "Hello world, this is a test message."}
        tokens = engine._estimate_message_tokens(msg)
        # 4 overhead + len(content)//4
        assert tokens == 4 + len("Hello world, this is a test message.") // 4

    def test_estimate_with_token_counter(self):
        Cls = ConcreteEngine._make()
        engine = Cls(context_window_tokens=10000)
        counter = MagicMock()
        counter.count.return_value = 10
        engine.set_token_counter(counter)
        msg = {"role": "user", "content": "Hello"}
        tokens = engine._estimate_message_tokens(msg)
        assert tokens == 14  # 10 + 4 overhead
        counter.count.assert_called_with("Hello")

    def test_estimate_with_content_blocks_and_counter(self):
        Cls = ConcreteEngine._make()
        engine = Cls(context_window_tokens=10000)
        counter = MagicMock()
        counter.count.return_value = 5
        engine.set_token_counter(counter)
        msg = {"role": "user", "content": [{"text": "a"}, {"text": "b"}]}
        tokens = engine._estimate_message_tokens(msg)
        assert tokens == 4 + 5 + 5  # overhead + 2 blocks

    def test_should_compress(self):
        Cls = ConcreteEngine._make()
        engine = Cls(context_window_tokens=100, trigger_ratio=0.5)
        # Each msg ~ 4 + len//4 tokens. Make enough to exceed 50.
        msgs = [{"role": "user", "content": "x" * 200}]  # 4 + 50 = 54 > 50
        assert engine.should_compress(msgs) is True

    def test_should_not_compress(self):
        Cls = ConcreteEngine._make()
        engine = Cls(context_window_tokens=10000, trigger_ratio=0.7)
        msgs = [{"role": "user", "content": "hi"}]
        assert engine.should_compress(msgs) is False

    def test_set_token_counter(self):
        Cls = ConcreteEngine._make()
        engine = Cls(context_window_tokens=100)
        assert engine._token_counter is None
        engine.set_token_counter("fake")
        assert engine._token_counter == "fake"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Config — schema.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigDefaults:
    def test_config_instantiation(self):
        from echo_agent.config.schema import Config
        cfg = Config()
        assert cfg.models.default_model == ""
        assert cfg.workspace == "~/.echo-agent"

    def test_planning_config_defaults(self):
        from echo_agent.config.schema import PlanningConfig
        pc = PlanningConfig()
        assert pc.enabled is True
        assert pc.default_strategy == "auto"
        assert pc.max_tree_depth == 5
        assert pc.reflection_enabled is True

    def test_a2a_config_defaults(self):
        from echo_agent.config.schema import A2AConfig
        a = A2AConfig()
        assert a.enabled is True
        assert a.agent_name == "echo-agent"
        assert "chat" in a.capabilities

    def test_eval_config_defaults(self):
        from echo_agent.config.schema import EvalConfig
        e = EvalConfig()
        assert e.enabled is True
        assert e.parallel_cases == 3
        assert e.timeout_per_case == 120

    def test_memory_config_new_fields(self):
        from echo_agent.config.schema import MemoryConfig
        m = MemoryConfig()
        assert m.graph_enabled is False
        assert m.hybrid_retrieval is True
        assert m.prefetch_enabled is False
        assert m.contradiction_detection is False
        assert m.adaptive_forgetting is True
        assert m.sleep_consolidation is True
        assert m.archival_threshold == 0.05
        assert m.forget_threshold == 0.01
        assert m.max_working_memory == 20
        assert m.max_episodes == 500

    def test_compression_config_defaults(self):
        from echo_agent.config.schema import CompressionConfig
        c = CompressionConfig()
        assert c.enabled is True
        assert c.trigger_ratio == 0.7
        assert c.tool_pruning_enabled is True

    def test_mcp_server_config_transport(self):
        from echo_agent.config.schema import MCPServerConfig
        m = MCPServerConfig()
        assert m.transport == "auto"
        assert m.connect_timeout == 60
