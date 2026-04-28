"""Tests for the 6 new channel features:
typing indicators, reactions, polls, message deletion, read receipts, voice messages.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from echo_agent.bus.events import (
    ContentType,
    InboundEvent,
    OutboundEvent,
    PollRequest,
    ProcessingOutcome,
)
from echo_agent.bus.queue import MessageBus
from echo_agent.channels.base import BaseChannel, SendResult
from echo_agent.channels.manager import ChannelManager, _emoji
from echo_agent.config.schema import (
    ChannelsConfig,
    TelegramChannelConfig,
    DiscordChannelConfig,
    SlackChannelConfig,
    MatrixChannelConfig,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class _FakeConfig:
    reactions_enabled: bool = True
    allow_from: list[str] = field(default_factory=list)


class _TrackingChannel(BaseChannel):
    """Fake channel that records all feature method calls."""

    def __init__(self, name: str, bus: MessageBus, *, reactions_enabled: bool = True):
        cfg = _FakeConfig(reactions_enabled=reactions_enabled)
        super().__init__(config=cfg, bus=bus)
        self.name = name
        self._running = True
        self.typing_calls: list[tuple[str, dict | None]] = []
        self.stop_typing_calls: list[str] = []
        self.reaction_calls: list[tuple[str, str, str]] = []
        self.remove_reaction_calls: list[tuple[str, str, str]] = []
        self.read_receipt_calls: list[tuple[str, str]] = []
        self.poll_calls: list[tuple[str, PollRequest]] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.voice_calls: list[tuple[str, str]] = []
        self.sent: list[OutboundEvent] = []

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send(self, event: OutboundEvent) -> SendResult:
        self.sent.append(event)
        return SendResult(success=True, message_id="msg-1")

    async def send_typing(self, chat_id: str, metadata: dict[str, Any] | None = None) -> None:
        self.typing_calls.append((chat_id, metadata))

    async def stop_typing(self, chat_id: str) -> None:
        self.stop_typing_calls.append(chat_id)

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        self.reaction_calls.append((chat_id, message_id, emoji))
        return SendResult(success=True)

    async def remove_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        self.remove_reaction_calls.append((chat_id, message_id, emoji))
        return SendResult(success=True)

    async def send_read_receipt(self, chat_id: str, message_id: str, metadata: dict[str, Any] | None = None) -> None:
        self.read_receipt_calls.append((chat_id, message_id))

    async def send_poll(self, chat_id: str, poll: PollRequest, metadata: dict[str, Any] | None = None) -> SendResult:
        self.poll_calls.append((chat_id, poll))
        return SendResult(success=True)

    async def delete_message(self, chat_id: str, message_id: str, metadata: dict[str, Any] | None = None) -> SendResult:
        self.delete_calls.append((chat_id, message_id))
        return SendResult(success=True)

    async def send_voice(self, chat_id: str, audio_source: str, metadata: dict[str, Any] | None = None) -> SendResult:
        self.voice_calls.append((chat_id, audio_source))
        return SendResult(success=True)


def _make_inbound(channel: str, chat_id: str = "chat-1", reply_to_id: str | None = "plat-msg-1") -> InboundEvent:
    return InboundEvent.text_message(
        channel=channel, sender_id="user-1", chat_id=chat_id, text="hello",
        reply_to_id=reply_to_id, metadata={},
    )


def _make_outbound_final(channel: str, chat_id: str = "chat-1", inbound_event_id: str = "", is_error: bool = False) -> OutboundEvent:
    event = OutboundEvent.text_reply(channel=channel, chat_id=chat_id, text="reply")
    event.is_final = True
    event.message_kind = "final"
    event.metadata = {"_inbound_event_id": inbound_event_id}
    if is_error:
        event.metadata["_error"] = True
    return event


# ── 1. Event model tests ────────────────────────────────────────────────────


def test_content_type_voice_exists() -> None:
    assert ContentType.VOICE == "voice"
    assert ContentType.VOICE in ContentType.__members__.values()


def test_processing_outcome_enum() -> None:
    assert ProcessingOutcome.SUCCESS == "success"
    assert ProcessingOutcome.FAILURE == "failure"


def test_poll_request_defaults() -> None:
    poll = PollRequest(question="Favorite?", options=["A", "B"])
    assert poll.question == "Favorite?"
    assert poll.options == ["A", "B"]
    assert poll.allow_multiple is False
    assert poll.duration_seconds is None


def test_poll_request_custom() -> None:
    poll = PollRequest(question="Pick", options=["X", "Y", "Z"], allow_multiple=True, duration_seconds=120)
    assert poll.allow_multiple is True
    assert poll.duration_seconds == 120


# ── 2. BaseChannel default no-op tests ──────────────────────────────────────


class _MinimalChannel(BaseChannel):
    def __init__(self, bus: MessageBus):
        super().__init__(config=object(), bus=bus)
        self.name = "minimal"

    async def start(self) -> None: pass
    async def stop(self) -> None: pass
    async def send(self, event: OutboundEvent) -> SendResult:
        return SendResult(success=True)


@pytest.mark.asyncio
async def test_base_send_typing_is_noop() -> None:
    bus = MessageBus()
    ch = _MinimalChannel(bus)
    await ch.send_typing("chat-1")


@pytest.mark.asyncio
async def test_base_stop_typing_is_noop() -> None:
    bus = MessageBus()
    ch = _MinimalChannel(bus)
    await ch.stop_typing("chat-1")


@pytest.mark.asyncio
async def test_base_send_reaction_returns_not_supported() -> None:
    bus = MessageBus()
    ch = _MinimalChannel(bus)
    result = await ch.send_reaction("chat-1", "msg-1", "👍")
    assert not result.success
    assert "does not support reactions" in result.error


@pytest.mark.asyncio
async def test_base_remove_reaction_returns_not_supported() -> None:
    bus = MessageBus()
    ch = _MinimalChannel(bus)
    result = await ch.remove_reaction("chat-1", "msg-1", "👍")
    assert not result.success


@pytest.mark.asyncio
async def test_base_send_poll_returns_not_supported() -> None:
    bus = MessageBus()
    ch = _MinimalChannel(bus)
    poll = PollRequest(question="Q?", options=["A", "B"])
    result = await ch.send_poll("chat-1", poll)
    assert not result.success
    assert "does not support polls" in result.error


@pytest.mark.asyncio
async def test_base_delete_message_returns_not_supported() -> None:
    bus = MessageBus()
    ch = _MinimalChannel(bus)
    result = await ch.delete_message("chat-1", "msg-1")
    assert not result.success
    assert "does not support message deletion" in result.error


@pytest.mark.asyncio
async def test_base_send_read_receipt_is_noop() -> None:
    bus = MessageBus()
    ch = _MinimalChannel(bus)
    await ch.send_read_receipt("chat-1", "msg-1")


@pytest.mark.asyncio
async def test_base_send_voice_returns_not_supported() -> None:
    bus = MessageBus()
    ch = _MinimalChannel(bus)
    result = await ch.send_voice("chat-1", "/tmp/audio.ogg")
    assert not result.success
    assert "does not support voice messages" in result.error


# ── 3. ChannelManager inbound lifecycle tests ───────────────────────────────


@pytest.mark.asyncio
async def test_inbound_triggers_typing() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram")
    await manager._on_inbound_lifecycle(inbound)

    assert len(ch.typing_calls) == 1
    assert ch.typing_calls[0][0] == "chat-1"


@pytest.mark.asyncio
async def test_inbound_triggers_read_receipt_when_reply_to_id_present() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram", reply_to_id="plat-msg-1")
    await manager._on_inbound_lifecycle(inbound)

    assert len(ch.read_receipt_calls) == 1
    assert ch.read_receipt_calls[0] == ("chat-1", "plat-msg-1")


@pytest.mark.asyncio
async def test_inbound_no_read_receipt_without_reply_to_id() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram", reply_to_id=None)
    await manager._on_inbound_lifecycle(inbound)

    assert ch.read_receipt_calls == []


@pytest.mark.asyncio
async def test_inbound_triggers_processing_reaction() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus, reactions_enabled=True)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram")
    await manager._on_inbound_lifecycle(inbound)

    assert len(ch.reaction_calls) == 1
    assert ch.reaction_calls[0][2] == "\U0001f440"


@pytest.mark.asyncio
async def test_inbound_no_reaction_when_disabled() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus, reactions_enabled=False)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram")
    await manager._on_inbound_lifecycle(inbound)

    assert ch.reaction_calls == []


@pytest.mark.asyncio
async def test_inbound_stores_msg_id_mapping() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram", reply_to_id="plat-42")
    await manager._on_inbound_lifecycle(inbound)

    assert inbound.event_id in manager._inbound_msg_ids
    assert manager._inbound_msg_ids[inbound.event_id] == ("telegram", "plat-42")


# ── 4. ChannelManager outbound lifecycle tests ──────────────────────────────


@pytest.mark.asyncio
async def test_outbound_final_stops_typing() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus)
    manager._channels["telegram"] = ch

    outbound = _make_outbound_final("telegram")
    await manager._on_outbound_final(outbound)

    assert len(ch.stop_typing_calls) == 1
    assert ch.stop_typing_calls[0] == "chat-1"


@pytest.mark.asyncio
async def test_outbound_final_swaps_reaction_to_success() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus, reactions_enabled=True)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram", reply_to_id="plat-msg-1")
    await manager._on_inbound_lifecycle(inbound)
    ch.reaction_calls.clear()

    outbound = _make_outbound_final("telegram", inbound_event_id=inbound.event_id)
    await manager._on_outbound_final(outbound)

    assert len(ch.remove_reaction_calls) == 1
    assert ch.remove_reaction_calls[0][2] == "\U0001f440"
    assert len(ch.reaction_calls) == 1
    assert ch.reaction_calls[0][2] == "\U0001f44d"


@pytest.mark.asyncio
async def test_outbound_final_swaps_reaction_to_failure_on_error() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus, reactions_enabled=True)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram", reply_to_id="plat-msg-1")
    await manager._on_inbound_lifecycle(inbound)
    ch.reaction_calls.clear()

    outbound = _make_outbound_final("telegram", inbound_event_id=inbound.event_id, is_error=True)
    await manager._on_outbound_final(outbound)

    assert len(ch.reaction_calls) == 1
    assert ch.reaction_calls[0][2] == "❌"


@pytest.mark.asyncio
async def test_outbound_final_no_reaction_swap_when_disabled() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus, reactions_enabled=False)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram", reply_to_id="plat-msg-1")
    await manager._on_inbound_lifecycle(inbound)

    outbound = _make_outbound_final("telegram", inbound_event_id=inbound.event_id)
    await manager._on_outbound_final(outbound)

    assert ch.remove_reaction_calls == []


@pytest.mark.asyncio
async def test_outbound_final_cleans_up_mapping() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus, reactions_enabled=True)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram")
    await manager._on_inbound_lifecycle(inbound)
    assert inbound.event_id in manager._inbound_msg_ids

    outbound = _make_outbound_final("telegram", inbound_event_id=inbound.event_id)
    await manager._on_outbound_final(outbound)

    assert inbound.event_id not in manager._inbound_msg_ids


# ── 5. Emoji mapping tests ─────────────────────────────────────────────────


def test_emoji_mapping_telegram() -> None:
    assert _emoji("telegram", "processing") == "\U0001f440"
    assert _emoji("telegram", "success") == "\U0001f44d"
    assert _emoji("telegram", "failure") == "❌"


def test_emoji_mapping_slack() -> None:
    assert _emoji("slack", "processing") == "eyes"
    assert _emoji("slack", "success") == "white_check_mark"
    assert _emoji("slack", "failure") == "x"


def test_emoji_mapping_unknown_channel() -> None:
    assert _emoji("unknown_channel", "processing") == ""


def test_emoji_mapping_unknown_kind() -> None:
    assert _emoji("telegram", "unknown_kind") == ""


# ── 6. Config reactions_enabled tests ───────────────────────────────────────


def test_telegram_config_reactions_enabled_default() -> None:
    cfg = TelegramChannelConfig()
    assert cfg.reactions_enabled is True


def test_discord_config_reactions_enabled_default() -> None:
    cfg = DiscordChannelConfig()
    assert cfg.reactions_enabled is True


def test_slack_config_reactions_enabled_default() -> None:
    cfg = SlackChannelConfig()
    assert cfg.reactions_enabled is True


def test_matrix_config_reactions_enabled_default() -> None:
    cfg = MatrixChannelConfig()
    assert cfg.reactions_enabled is True


def test_config_reactions_can_be_disabled() -> None:
    cfg = TelegramChannelConfig(reactions_enabled=False)
    assert cfg.reactions_enabled is False


# ── 7. Telegram platform-specific tests ─────────────────────────────────────


class _FakeResp:
    def __init__(self, status: int = 200, data: dict | None = None):
        self.status = status
        self._data = data or {"ok": True, "result": True}

    async def json(self):
        return self._data

    async def text(self):
        return str(self._data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_telegram_send_typing() -> None:
    from echo_agent.channels.telegram import TelegramChannel
    cfg = TelegramChannelConfig(enabled=True, token="fake-token")
    bus = MessageBus()
    ch = TelegramChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200, {"ok": True, "result": True}))

    await ch.send_typing("12345")
    ch._session.post.assert_called_once()
    call_args = ch._session.post.call_args
    assert "sendChatAction" in call_args[0][0]


@pytest.mark.asyncio
async def test_telegram_send_reaction() -> None:
    from echo_agent.channels.telegram import TelegramChannel
    cfg = TelegramChannelConfig(enabled=True, token="fake-token", reactions_enabled=True)
    bus = MessageBus()
    ch = TelegramChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200, {"ok": True, "result": True}))

    result = await ch.send_reaction("12345", "100", "\U0001f44d")
    assert result.success
    call_args = ch._session.post.call_args
    assert "setMessageReaction" in call_args[0][0]


@pytest.mark.asyncio
async def test_telegram_remove_reaction() -> None:
    from echo_agent.channels.telegram import TelegramChannel
    cfg = TelegramChannelConfig(enabled=True, token="fake-token", reactions_enabled=True)
    bus = MessageBus()
    ch = TelegramChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200, {"ok": True, "result": True}))

    result = await ch.remove_reaction("12345", "100", "\U0001f44d")
    assert result.success
    call_json = ch._session.post.call_args[1]["json"]
    assert call_json["reaction"] == []


@pytest.mark.asyncio
async def test_telegram_send_reaction_disabled() -> None:
    from echo_agent.channels.telegram import TelegramChannel
    cfg = TelegramChannelConfig(enabled=True, token="fake-token", reactions_enabled=False)
    bus = MessageBus()
    ch = TelegramChannel(cfg, bus)

    result = await ch.send_reaction("12345", "100", "\U0001f44d")
    assert not result.success
    assert "disabled" in result.error


@pytest.mark.asyncio
async def test_telegram_send_poll() -> None:
    from echo_agent.channels.telegram import TelegramChannel
    cfg = TelegramChannelConfig(enabled=True, token="fake-token")
    bus = MessageBus()
    ch = TelegramChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200, {"ok": True, "result": {"message_id": 42}}))

    poll = PollRequest(question="Favorite?", options=["A", "B", "C"], duration_seconds=60)
    result = await ch.send_poll("12345", poll)
    assert result.success
    call_json = ch._session.post.call_args[1]["json"]
    assert call_json["question"] == "Favorite?"
    assert len(call_json["options"]) == 3
    assert call_json["open_period"] == 60


@pytest.mark.asyncio
async def test_telegram_delete_message() -> None:
    from echo_agent.channels.telegram import TelegramChannel
    cfg = TelegramChannelConfig(enabled=True, token="fake-token")
    bus = MessageBus()
    ch = TelegramChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200, {"ok": True, "result": True}))

    result = await ch.delete_message("12345", "100")
    assert result.success
    assert "deleteMessage" in ch._session.post.call_args[0][0]


@pytest.mark.asyncio
async def test_telegram_send_voice() -> None:
    from echo_agent.channels.telegram import TelegramChannel
    cfg = TelegramChannelConfig(enabled=True, token="fake-token")
    bus = MessageBus()
    ch = TelegramChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200, {"ok": True, "result": {"message_id": 99}}))

    result = await ch.send_voice("12345", "file_id_abc")
    assert result.success
    assert "sendVoice" in ch._session.post.call_args[0][0]


# ── 8. Discord platform-specific tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_discord_send_typing_creates_loop_task() -> None:
    from echo_agent.channels.discord import DiscordChannel
    cfg = DiscordChannelConfig(enabled=True, token="fake-token")
    bus = MessageBus()
    ch = DiscordChannel(cfg, bus)
    ch._running = True
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200))

    await ch.send_typing("chan-1")
    assert "chan-1" in ch._typing_tasks
    task = ch._typing_tasks["chan-1"]
    assert not task.done()

    await ch.stop_typing("chan-1")
    assert "chan-1" not in ch._typing_tasks
    await asyncio.sleep(0.05)
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_discord_send_typing_no_duplicate() -> None:
    from echo_agent.channels.discord import DiscordChannel
    cfg = DiscordChannelConfig(enabled=True, token="fake-token")
    bus = MessageBus()
    ch = DiscordChannel(cfg, bus)
    ch._running = True
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200))

    await ch.send_typing("chan-1")
    first_task = ch._typing_tasks["chan-1"]
    await ch.send_typing("chan-1")
    assert ch._typing_tasks["chan-1"] is first_task

    await ch.stop_typing("chan-1")


@pytest.mark.asyncio
async def test_discord_send_reaction() -> None:
    from echo_agent.channels.discord import DiscordChannel
    cfg = DiscordChannelConfig(enabled=True, token="fake-token", reactions_enabled=True)
    bus = MessageBus()
    ch = DiscordChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.put = MagicMock(return_value=_FakeResp(204))

    result = await ch.send_reaction("chan-1", "msg-1", "👍")
    assert result.success
    call_url = ch._session.put.call_args[0][0]
    assert "/reactions/" in call_url
    assert "/@me" in call_url


@pytest.mark.asyncio
async def test_discord_remove_reaction() -> None:
    from echo_agent.channels.discord import DiscordChannel
    cfg = DiscordChannelConfig(enabled=True, token="fake-token", reactions_enabled=True)
    bus = MessageBus()
    ch = DiscordChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.delete = MagicMock(return_value=_FakeResp(204))

    result = await ch.remove_reaction("chan-1", "msg-1", "👍")
    assert result.success
    call_url = ch._session.delete.call_args[0][0]
    assert "/reactions/" in call_url


@pytest.mark.asyncio
async def test_discord_delete_message() -> None:
    from echo_agent.channels.discord import DiscordChannel
    cfg = DiscordChannelConfig(enabled=True, token="fake-token")
    bus = MessageBus()
    ch = DiscordChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.delete = MagicMock(return_value=_FakeResp(204))

    result = await ch.delete_message("chan-1", "msg-1")
    assert result.success
    call_url = ch._session.delete.call_args[0][0]
    assert "/channels/chan-1/messages/msg-1" in call_url


# ── 9. Slack platform-specific tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_send_typing() -> None:
    from echo_agent.channels.slack import SlackChannel
    cfg = SlackChannelConfig(enabled=True, bot_token="xoxb-fake", app_token="xapp-fake")
    bus = MessageBus()
    ch = SlackChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200, {"ok": True}))

    await ch.send_typing("C123", metadata={"thread_ts": "1234.5678"})
    call_url = ch._session.post.call_args[0][0]
    assert "assistant.threads.setStatus" in call_url


@pytest.mark.asyncio
async def test_slack_send_typing_no_thread_ts_is_noop() -> None:
    from echo_agent.channels.slack import SlackChannel
    cfg = SlackChannelConfig(enabled=True, bot_token="xoxb-fake", app_token="xapp-fake")
    bus = MessageBus()
    ch = SlackChannel(cfg, bus)
    ch._session = MagicMock()

    await ch.send_typing("C123", metadata={})
    ch._session.post.assert_not_called()


@pytest.mark.asyncio
async def test_slack_send_reaction() -> None:
    from echo_agent.channels.slack import SlackChannel
    cfg = SlackChannelConfig(enabled=True, bot_token="xoxb-fake", app_token="xapp-fake", reactions_enabled=True)
    bus = MessageBus()
    ch = SlackChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200, {"ok": True}))

    result = await ch.send_reaction("C123", "1234.5678", "eyes")
    assert result.success
    call_url = ch._session.post.call_args[0][0]
    assert "reactions.add" in call_url


@pytest.mark.asyncio
async def test_slack_remove_reaction() -> None:
    from echo_agent.channels.slack import SlackChannel
    cfg = SlackChannelConfig(enabled=True, bot_token="xoxb-fake", app_token="xapp-fake", reactions_enabled=True)
    bus = MessageBus()
    ch = SlackChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200, {"ok": True}))

    result = await ch.remove_reaction("C123", "1234.5678", "eyes")
    assert result.success
    call_url = ch._session.post.call_args[0][0]
    assert "reactions.remove" in call_url


@pytest.mark.asyncio
async def test_slack_delete_message() -> None:
    from echo_agent.channels.slack import SlackChannel
    cfg = SlackChannelConfig(enabled=True, bot_token="xoxb-fake", app_token="xapp-fake")
    bus = MessageBus()
    ch = SlackChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200, {"ok": True}))

    result = await ch.delete_message("C123", "1234.5678")
    assert result.success
    call_url = ch._session.post.call_args[0][0]
    assert "chat.delete" in call_url


# ── 10. Matrix platform-specific tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_matrix_send_typing() -> None:
    from echo_agent.channels.matrix import MatrixChannel
    cfg = MatrixChannelConfig(enabled=True, homeserver="https://matrix.example.com", user_id="@bot:example.com", access_token="fake")
    bus = MessageBus()
    ch = MatrixChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.put = MagicMock(return_value=_FakeResp(200))

    await ch.send_typing("!room:example.com")
    call_url = ch._session.put.call_args[0][0]
    assert "/typing/" in call_url
    call_json = ch._session.put.call_args[1]["json"]
    assert call_json["typing"] is True


@pytest.mark.asyncio
async def test_matrix_stop_typing() -> None:
    from echo_agent.channels.matrix import MatrixChannel
    cfg = MatrixChannelConfig(enabled=True, homeserver="https://matrix.example.com", user_id="@bot:example.com", access_token="fake")
    bus = MessageBus()
    ch = MatrixChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.put = MagicMock(return_value=_FakeResp(200))

    await ch.stop_typing("!room:example.com")
    call_json = ch._session.put.call_args[1]["json"]
    assert call_json["typing"] is False


@pytest.mark.asyncio
async def test_matrix_send_reaction() -> None:
    from echo_agent.channels.matrix import MatrixChannel
    cfg = MatrixChannelConfig(enabled=True, homeserver="https://matrix.example.com", user_id="@bot:example.com", access_token="fake", reactions_enabled=True)
    bus = MessageBus()
    ch = MatrixChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.put = MagicMock(return_value=_FakeResp(200, {"event_id": "$reaction1"}))

    result = await ch.send_reaction("!room:example.com", "$msg1", "👍")
    assert result.success
    call_url = ch._session.put.call_args[0][0]
    assert "/send/m.reaction/" in call_url
    call_json = ch._session.put.call_args[1]["json"]
    assert call_json["m.relates_to"]["rel_type"] == "m.annotation"
    assert call_json["m.relates_to"]["key"] == "👍"


@pytest.mark.asyncio
async def test_matrix_send_read_receipt() -> None:
    from echo_agent.channels.matrix import MatrixChannel
    cfg = MatrixChannelConfig(enabled=True, homeserver="https://matrix.example.com", user_id="@bot:example.com", access_token="fake")
    bus = MessageBus()
    ch = MatrixChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.post = MagicMock(return_value=_FakeResp(200))

    await ch.send_read_receipt("!room:example.com", "$msg1")
    call_url = ch._session.post.call_args[0][0]
    assert "/receipt/m.read/" in call_url


@pytest.mark.asyncio
async def test_matrix_delete_message() -> None:
    from echo_agent.channels.matrix import MatrixChannel
    cfg = MatrixChannelConfig(enabled=True, homeserver="https://matrix.example.com", user_id="@bot:example.com", access_token="fake")
    bus = MessageBus()
    ch = MatrixChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.put = MagicMock(return_value=_FakeResp(200))

    result = await ch.delete_message("!room:example.com", "$msg1")
    assert result.success
    call_url = ch._session.put.call_args[0][0]
    assert "/redact/" in call_url


@pytest.mark.asyncio
async def test_matrix_send_poll() -> None:
    from echo_agent.channels.matrix import MatrixChannel
    cfg = MatrixChannelConfig(enabled=True, homeserver="https://matrix.example.com", user_id="@bot:example.com", access_token="fake")
    bus = MessageBus()
    ch = MatrixChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.put = MagicMock(return_value=_FakeResp(200, {"event_id": "$poll1"}))

    poll = PollRequest(question="Pick one", options=["X", "Y"], allow_multiple=False)
    result = await ch.send_poll("!room:example.com", poll)
    assert result.success
    call_url = ch._session.put.call_args[0][0]
    assert "/send/m.poll.start/" in call_url
    call_json = ch._session.put.call_args[1]["json"]
    assert call_json["org.matrix.msc3381.v2.poll"]["max_selections"] == 1


@pytest.mark.asyncio
async def test_matrix_send_voice() -> None:
    from echo_agent.channels.matrix import MatrixChannel
    cfg = MatrixChannelConfig(enabled=True, homeserver="https://matrix.example.com", user_id="@bot:example.com", access_token="fake")
    bus = MessageBus()
    ch = MatrixChannel(cfg, bus)
    ch._session = MagicMock()
    ch._session.put = MagicMock(return_value=_FakeResp(200, {"event_id": "$voice1"}))

    result = await ch.send_voice("!room:example.com", "mxc://example.com/audio123")
    assert result.success
    call_url = ch._session.put.call_args[0][0]
    assert "/send/m.room.message/" in call_url
    call_json = ch._session.put.call_args[1]["json"]
    assert call_json["msgtype"] == "m.audio"


# ── 11. Full lifecycle integration test ─────────────────────────────────────


@pytest.mark.asyncio
async def test_full_lifecycle_inbound_to_outbound() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus, reactions_enabled=True)
    manager._channels["telegram"] = ch
    bus.subscribe_outbound("telegram", ch.send)

    inbound = _make_inbound("telegram", reply_to_id="plat-99")
    await manager._on_inbound_lifecycle(inbound)

    assert len(ch.typing_calls) == 1
    assert len(ch.read_receipt_calls) == 1
    assert len(ch.reaction_calls) == 1
    assert ch.reaction_calls[0][2] == "\U0001f440"

    outbound = _make_outbound_final("telegram", inbound_event_id=inbound.event_id)
    await manager._on_outbound_final(outbound)

    assert len(ch.stop_typing_calls) == 1
    assert len(ch.remove_reaction_calls) == 1
    assert ch.remove_reaction_calls[0][2] == "\U0001f440"
    assert len(ch.reaction_calls) == 2
    assert ch.reaction_calls[1][2] == "\U0001f44d"


@pytest.mark.asyncio
async def test_full_lifecycle_error_path() -> None:
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus, reactions_enabled=True)
    manager._channels["telegram"] = ch

    inbound = _make_inbound("telegram", reply_to_id="plat-99")
    await manager._on_inbound_lifecycle(inbound)
    ch.reaction_calls.clear()

    outbound = _make_outbound_final("telegram", inbound_event_id=inbound.event_id, is_error=True)
    await manager._on_outbound_final(outbound)

    assert ch.reaction_calls[0][2] == "❌"


@pytest.mark.asyncio
async def test_existing_token_streaming_still_works() -> None:
    """Ensure the new lifecycle hooks don't break existing token streaming."""
    bus = MessageBus()
    manager = ChannelManager(ChannelsConfig(), bus)
    ch = _TrackingChannel("telegram", bus)
    ch.supports_edit = True
    ch._next_id = 0
    manager._channels["telegram"] = ch
    bus.subscribe_outbound("telegram", ch.send)

    stream_event = OutboundEvent.text_reply(channel="telegram", chat_id="chat-1", text="hello")
    stream_event.is_final = True
    stream_event.message_kind = "final"
    stream_event.metadata = {}

    await bus.publish_outbound(stream_event)
    assert len(ch.sent) == 1
