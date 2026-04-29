"""Comprehensive tests for echo_agent.memory.reviewer — MemoryReviewer."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from echo_agent.memory.reviewer import MemoryReviewer, _MAX_REVIEW_ITERATIONS
from echo_agent.memory.types import MemoryEntry, MemoryType


# ---------------------------------------------------------------------------
# Factories / helpers
# ---------------------------------------------------------------------------

def _make_entry(**overrides: Any) -> MemoryEntry:
    defaults = dict(
        id=uuid.uuid4().hex[:12],
        type=MemoryType.USER,
        key="test_key",
        content="test content",
        importance=0.5,
    )
    defaults.update(overrides)
    return MemoryEntry(**defaults)


def _make_tool_call(
    arguments: dict[str, Any],
    tc_id: str = "tc_1",
    name: str = "memory_manage",
) -> MagicMock:
    tc = MagicMock()
    tc.id = tc_id
    tc.name = name
    tc.arguments = arguments
    tc.to_openai_format.return_value = {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return tc


def _make_response(
    content: str = "",
    tool_calls: list | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = tool_calls or []
    resp.has_tool_calls = bool(tool_calls)
    resp.finish_reason = "stop"
    return resp


def _build_reviewer(session_key: str = "sess_1") -> tuple[MemoryReviewer, MagicMock, MagicMock]:
    """Return (reviewer, mock_provider, mock_store)."""
    provider = AsyncMock()
    store = MagicMock()
    reviewer = MemoryReviewer(provider=provider, store=store, model="test-model", session_key=session_key)
    return reviewer, provider, store


# ---------------------------------------------------------------------------
# TestResolveEntry
# ---------------------------------------------------------------------------

class TestResolveEntry:
    """Tests for MemoryReviewer._resolve_entry."""

    def test_key_found(self):
        reviewer, _, store = _build_reviewer()
        existing = _make_entry(key="lang")
        store.find_by_key.return_value = existing

        entry, err = reviewer._resolve_entry("lang", "", MemoryType.USER)

        assert entry is existing
        assert err is None
        store.find_by_key.assert_called_once_with("lang", MemoryType.USER, session_key="sess_1")

    def test_key_not_found_old_text_single_match(self):
        reviewer, _, store = _build_reviewer()
        existing = _make_entry(key="pref")
        store.find_by_key.return_value = None
        store.find_by_content_matches.return_value = [existing]

        entry, err = reviewer._resolve_entry("missing_key", "some old text", MemoryType.USER)

        assert entry is existing
        assert err is None
        store.find_by_content_matches.assert_called_once_with(
            "some old text", mem_type=MemoryType.USER, limit=6, session_key="sess_1",
        )

    def test_old_text_multiple_matches_returns_error(self):
        reviewer, _, store = _build_reviewer()
        m1 = _make_entry(key="a")
        m2 = _make_entry(key="b")
        store.find_by_key.return_value = None
        store.find_by_content_matches.return_value = [m1, m2]

        entry, err = reviewer._resolve_entry("", "ambiguous", MemoryType.ENVIRONMENT)

        assert entry is None
        assert err is not None
        assert "multiple matching memories" in err

    def test_neither_key_nor_old_text(self):
        reviewer, _, store = _build_reviewer()

        entry, err = reviewer._resolve_entry("", "", MemoryType.USER)

        assert entry is None
        assert err is None
        store.find_by_key.assert_not_called()
        store.find_by_content_matches.assert_not_called()

    def test_key_not_found_old_text_no_matches(self):
        reviewer, _, store = _build_reviewer()
        store.find_by_key.return_value = None
        store.find_by_content_matches.return_value = []

        entry, err = reviewer._resolve_entry("nope", "also nope", MemoryType.USER)

        assert entry is None
        assert err is None


# ---------------------------------------------------------------------------
# TestExecute
# ---------------------------------------------------------------------------

class TestExecute:
    """Tests for MemoryReviewer._execute."""

    # -- add --

    def test_add_success(self):
        reviewer, _, store = _build_reviewer()
        added = _make_entry(key="color")
        store.add.return_value = added

        result = reviewer._execute({"action": "add", "target": "user", "key": "color", "content": "blue"})

        assert result == "Added [user] color"
        store.add.assert_called_once()
        arg_entry: MemoryEntry = store.add.call_args[0][0]
        assert arg_entry.key == "color"
        assert arg_entry.content == "blue"
        assert arg_entry.type == MemoryType.USER
        assert arg_entry.source_session == "sess_1"

    def test_add_missing_key(self):
        reviewer, _, _ = _build_reviewer()
        result = reviewer._execute({"action": "add", "target": "user", "content": "blue"})
        assert result.startswith("Error")

    def test_add_missing_content(self):
        reviewer, _, _ = _build_reviewer()
        result = reviewer._execute({"action": "add", "target": "user", "key": "color"})
        assert result.startswith("Error")

    def test_add_store_raises_value_error(self):
        reviewer, _, store = _build_reviewer()
        store.add.side_effect = ValueError("duplicate key")

        result = reviewer._execute({"action": "add", "target": "user", "key": "k", "content": "c"})

        assert result == "Error: duplicate key"

    def test_add_environment_no_source_session(self):
        reviewer, _, store = _build_reviewer()
        added = _make_entry(key="proj")
        store.add.return_value = added

        reviewer._execute({"action": "add", "target": "environment", "key": "proj", "content": "python"})

        arg_entry: MemoryEntry = store.add.call_args[0][0]
        assert arg_entry.source_session == ""
        assert arg_entry.type == MemoryType.ENVIRONMENT

    # -- replace --

    def test_replace_existing_entry(self):
        reviewer, _, store = _build_reviewer()
        existing = _make_entry(id="e1", key="lang")
        store.find_by_key.return_value = existing

        result = reviewer._execute({
            "action": "replace", "target": "user", "key": "lang", "content": "rust",
        })

        assert result == "Updated [user] lang"
        store.update.assert_called_once_with("e1", content="rust")

    def test_replace_no_entry_creates_new(self):
        reviewer, _, store = _build_reviewer()
        store.find_by_key.return_value = None
        store.find_by_content_matches.return_value = []

        result = reviewer._execute({
            "action": "replace", "target": "user", "key": "theme", "content": "dark",
        })

        assert "Added (new)" in result
        assert "[user]" in result
        store.add.assert_called_once()

    def test_replace_resolve_error(self):
        reviewer, _, store = _build_reviewer()
        store.find_by_key.return_value = None
        m1 = _make_entry(key="a")
        m2 = _make_entry(key="b")
        store.find_by_content_matches.return_value = [m1, m2]

        result = reviewer._execute({
            "action": "replace", "target": "user", "old_text": "ambig", "content": "new",
        })

        assert result.startswith("Error")
        assert "multiple matching" in result

    def test_replace_missing_content(self):
        reviewer, _, _ = _build_reviewer()
        result = reviewer._execute({"action": "replace", "target": "user", "key": "k"})
        assert result == "Error: content required"

    def test_replace_update_raises_value_error(self):
        reviewer, _, store = _build_reviewer()
        existing = _make_entry(id="e1", key="lang")
        store.find_by_key.return_value = existing
        store.update.side_effect = ValueError("bad update")

        result = reviewer._execute({
            "action": "replace", "target": "user", "key": "lang", "content": "go",
        })

        assert result == "Error: bad update"

    # -- remove --

    def test_remove_success(self):
        reviewer, _, store = _build_reviewer()
        existing = _make_entry(id="e1", key="old_pref")
        store.find_by_key.return_value = existing

        result = reviewer._execute({"action": "remove", "target": "user", "key": "old_pref"})

        assert result == "Removed [user] old_pref"
        store.delete.assert_called_once_with("e1")

    def test_remove_no_matching_entry(self):
        reviewer, _, store = _build_reviewer()
        store.find_by_key.return_value = None
        store.find_by_content_matches.return_value = []

        result = reviewer._execute({"action": "remove", "target": "user", "key": "gone"})

        assert result == "Error: no matching memory found"

    def test_remove_resolve_error(self):
        reviewer, _, store = _build_reviewer()
        store.find_by_key.return_value = None
        m1 = _make_entry(key="x")
        m2 = _make_entry(key="y")
        store.find_by_content_matches.return_value = [m1, m2]

        result = reviewer._execute({
            "action": "remove", "target": "environment", "old_text": "ambig",
        })

        assert "multiple matching" in result

    # -- unknown --

    def test_unknown_action(self):
        reviewer, _, _ = _build_reviewer()
        result = reviewer._execute({"action": "dance", "target": "user"})
        assert result == "Error: unknown action 'dance'"

    def test_importance_clamped(self):
        reviewer, _, store = _build_reviewer()
        added = _make_entry(key="k")
        store.add.return_value = added

        reviewer._execute({"action": "add", "target": "user", "key": "k", "content": "c", "importance": 5.0})

        arg_entry: MemoryEntry = store.add.call_args[0][0]
        assert arg_entry.importance == 1.0


# ---------------------------------------------------------------------------
# TestReview
# ---------------------------------------------------------------------------

class TestReview:
    """Tests for MemoryReviewer.review (async)."""

    @pytest.mark.asyncio
    async def test_no_actions(self):
        reviewer, provider, _ = _build_reviewer()
        provider.chat_with_retry.return_value = _make_response(content="No memory changes needed.")

        actions = await reviewer.review([{"role": "user", "content": "hi"}])

        assert actions == []
        provider.chat_with_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_single_add_action(self):
        reviewer, provider, store = _build_reviewer()
        added = _make_entry(key="color")
        store.add.return_value = added

        tc = _make_tool_call({"action": "add", "target": "user", "key": "color", "content": "blue"})
        first_resp = _make_response(content="I'll save that.", tool_calls=[tc])
        second_resp = _make_response(content="Done.")

        provider.chat_with_retry.side_effect = [first_resp, second_resp]

        actions = await reviewer.review([{"role": "user", "content": "I like blue"}])

        assert len(actions) == 1
        assert "Added [user] color" in actions[0]

    @pytest.mark.asyncio
    async def test_multiple_actions_one_iteration(self):
        reviewer, provider, store = _build_reviewer()
        added1 = _make_entry(key="lang")
        added2 = _make_entry(key="editor")
        store.add.side_effect = [added1, added2]

        tc1 = _make_tool_call(
            {"action": "add", "target": "user", "key": "lang", "content": "python"}, tc_id="tc_1",
        )
        tc2 = _make_tool_call(
            {"action": "add", "target": "environment", "key": "editor", "content": "vim"}, tc_id="tc_2",
        )
        first_resp = _make_response(tool_calls=[tc1, tc2])
        second_resp = _make_response(content="All done.")

        provider.chat_with_retry.side_effect = [first_resp, second_resp]

        actions = await reviewer.review([{"role": "user", "content": "setup"}])

        assert len(actions) == 2

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self):
        reviewer, provider, store = _build_reviewer()
        added = _make_entry(key="k")
        store.add.return_value = added

        tc = _make_tool_call({"action": "add", "target": "user", "key": "k", "content": "v"})
        looping_resp = _make_response(tool_calls=[tc])
        provider.chat_with_retry.return_value = looping_resp

        actions = await reviewer.review([{"role": "user", "content": "loop"}])

        assert provider.chat_with_retry.await_count == _MAX_REVIEW_ITERATIONS
        assert len(actions) == _MAX_REVIEW_ITERATIONS

    @pytest.mark.asyncio
    async def test_llm_exception_breaks_loop(self):
        reviewer, provider, _ = _build_reviewer()
        provider.chat_with_retry.side_effect = RuntimeError("API down")

        actions = await reviewer.review([{"role": "user", "content": "hi"}])

        assert actions == []
        provider.chat_with_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_error_results_not_added_to_actions(self):
        reviewer, provider, store = _build_reviewer()
        # add with missing key -> Error
        tc = _make_tool_call({"action": "add", "target": "user", "content": "no key"})
        first_resp = _make_response(tool_calls=[tc])
        second_resp = _make_response(content="Oops.")

        provider.chat_with_retry.side_effect = [first_resp, second_resp]

        actions = await reviewer.review([{"role": "user", "content": "test"}])

        assert actions == []

    @pytest.mark.asyncio
    async def test_review_appends_review_prompt(self):
        """The review prompt is appended as the last user message."""
        reviewer, provider, _ = _build_reviewer()
        provider.chat_with_retry.return_value = _make_response(content="Nothing to save.")

        convo = [{"role": "user", "content": "hello"}]
        await reviewer.review(convo)

        sent_messages = provider.chat_with_retry.call_args[1]["messages"]
        # The review prompt is the user message right before the assistant reply
        review_msg = next(m for m in sent_messages if m["role"] == "user" and "Review the conversation" in m["content"])
        assert review_msg is not None
        # Original conversation should not be mutated
        assert len(convo) == 1

    @pytest.mark.asyncio
    async def test_tool_results_appended_to_messages(self):
        """Tool results are fed back as tool-role messages."""
        reviewer, provider, store = _build_reviewer()
        added = _make_entry(key="k")
        store.add.return_value = added

        tc = _make_tool_call({"action": "add", "target": "user", "key": "k", "content": "v"})
        first_resp = _make_response(content="Saving.", tool_calls=[tc])
        second_resp = _make_response(content="Done.")
        provider.chat_with_retry.side_effect = [first_resp, second_resp]

        await reviewer.review([{"role": "user", "content": "hi"}])

        second_call_msgs = provider.chat_with_retry.call_args_list[1][1]["messages"]
        tool_msgs = [m for m in second_call_msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc_1"
        assert "Added [user] k" in tool_msgs[0]["content"]
