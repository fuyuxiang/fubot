"""Tests for echo_agent/utils/text.py."""

from __future__ import annotations

import pytest

from echo_agent.utils.text import (
    estimate_messages_tokens,
    estimate_tokens,
    html_to_text,
    split_message,
    strip_markdown,
    strip_thinking,
)


# ---------------------------------------------------------------------------
# split_message
# ---------------------------------------------------------------------------

class TestSplitMessage:
    def test_short_no_split(self):
        assert split_message("hello", max_len=100) == ["hello"]

    def test_split_at_paragraph_boundary(self):
        text = "A" * 90 + "\n\n" + "B" * 50
        chunks = split_message(text, max_len=100)
        assert len(chunks) >= 2
        assert chunks[0].endswith("A" * 10) or "A" in chunks[0]
        combined = "".join(c.strip() for c in chunks)
        assert "A" * 90 in combined
        assert "B" * 50 in combined

    def test_split_at_sentence_boundary(self):
        text = "Hello world. " * 20
        chunks = split_message(text, max_len=100)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_cjk_punctuation(self):
        text = "你好世界。" * 60
        chunks = split_message(text, max_len=100)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_hard_cut_no_boundary(self):
        text = "A" * 300
        chunks = split_message(text, max_len=100)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_min_chunk_ratio(self):
        # With a very early newline, min_chunk_ratio should prevent a tiny first chunk
        text = "Hi\n" + "X" * 200
        chunks = split_message(text, max_len=100, min_chunk_ratio=0.75)
        assert len(chunks) >= 2
        # First chunk should not be just "Hi" (too small relative to max_len)
        assert len(chunks[0]) >= 50 or len(text) <= 100

    def test_empty_string(self):
        assert split_message("") == [""]

    def test_exact_max_len(self):
        text = "A" * 100
        assert split_message(text, max_len=100) == [text]


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_ascii(self):
        result = estimate_tokens("hello world")  # 11 ascii chars
        assert result == 11 // 4 + 1  # 3

    def test_cjk(self):
        text = "你好世界"  # 4 non-ascii chars
        result = estimate_tokens(text)
        assert result == 4 // 2 + 1  # 3

    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_mixed(self):
        text = "hello你好"  # 5 ascii + 2 non-ascii
        result = estimate_tokens(text)
        assert result == 5 // 4 + 2 // 2 + 1  # 1 + 1 + 1 = 3


# ---------------------------------------------------------------------------
# estimate_messages_tokens
# ---------------------------------------------------------------------------

class TestEstimateMessagesTokens:
    def test_basic_messages(self):
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = estimate_messages_tokens(messages)
        assert result > 0

    def test_empty_messages(self):
        assert estimate_messages_tokens([]) == 0

    def test_message_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": '{"q": "test"}'}}
                ],
            }
        ]
        result = estimate_messages_tokens(messages)
        assert result > 0


# ---------------------------------------------------------------------------
# strip_thinking
# ---------------------------------------------------------------------------

class TestStripThinking:
    def test_removes_think_block(self):
        text = "Before <think>internal reasoning</think> After"
        assert strip_thinking(text) == "Before  After"

    def test_multiline_think(self):
        text = "Start\n<think>\nline1\nline2\n</think>\nEnd"
        result = strip_thinking(text)
        assert "<think>" not in result
        assert "End" in result

    def test_no_think_block(self):
        assert strip_thinking("plain text") == "plain text"


# ---------------------------------------------------------------------------
# strip_markdown
# ---------------------------------------------------------------------------

class TestStripMarkdown:
    def test_links(self):
        assert "click here" in strip_markdown("[click here](https://example.com)")

    def test_bold(self):
        result = strip_markdown("**bold text**")
        assert "bold text" in result
        assert "**" not in result

    def test_headers(self):
        result = strip_markdown("## Section Title\nContent")
        assert result.startswith("Section Title")

    def test_image(self):
        result = strip_markdown("![alt text](image.png)")
        assert "alt text" in result
        assert "![" not in result


# ---------------------------------------------------------------------------
# html_to_text
# ---------------------------------------------------------------------------

class TestHtmlToText:
    def test_br_tag(self):
        result = html_to_text("line1<br>line2")
        assert "line1\nline2" == result

    def test_p_tag(self):
        result = html_to_text("<p>paragraph one</p><p>paragraph two</p>")
        assert "paragraph one" in result
        assert "paragraph two" in result

    def test_entities(self):
        result = html_to_text("&amp; &lt; &gt; &quot; &#39;")
        assert "& < > \"" in result

    def test_nbsp(self):
        result = html_to_text("hello&nbsp;world")
        assert "hello world" == result

    def test_strips_tags(self):
        result = html_to_text("<div><span>text</span></div>")
        assert result == "text"

