"""Comprehensive tests for echo_agent.bus.events and echo_agent.bus.queue."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from echo_agent.bus.events import (
    ContentBlock,
    ContentType,
    EventType,
    InboundEvent,
    OutboundEvent,
    PollRequest,
    ProcessingOutcome,
)
from echo_agent.bus.queue import MessageBus


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def make_inbound(**kwargs) -> InboundEvent:
    defaults = dict(channel="qq", sender_id="u1", chat_id="c1")
    defaults.update(kwargs)
    return InboundEvent(**defaults)


def make_outbound(**kwargs) -> OutboundEvent:
    defaults = dict(channel="qq", chat_id="c1")
    defaults.update(kwargs)
    return OutboundEvent(**defaults)


# ===================================================================
# events.py tests
# ===================================================================


class TestEnums:
    def test_event_type_values(self):
        assert EventType.MESSAGE == "message"
        assert EventType.WEBHOOK == "webhook"
        assert EventType.CRON == "cron"
        assert EventType.CLI == "cli"
        assert EventType.SYSTEM == "system"

    def test_content_type_values(self):
        assert ContentType.TEXT == "text"
        assert ContentType.IMAGE == "image"
        assert ContentType.FILE == "file"
        assert ContentType.AUDIO == "audio"
        assert ContentType.VIDEO == "video"
        assert ContentType.VOICE == "voice"
        assert ContentType.MIXED == "mixed"

    def test_processing_outcome_values(self):
        assert ProcessingOutcome.SUCCESS == "success"
        assert ProcessingOutcome.FAILURE == "failure"


class TestPollRequest:
    def test_defaults(self):
        pr = PollRequest(question="Pick one", options=["A", "B"])
        assert pr.question == "Pick one"
        assert pr.options == ["A", "B"]
        assert pr.allow_multiple is False
        assert pr.duration_seconds is None

    def test_custom_values(self):
        pr = PollRequest("Q", ["X"], allow_multiple=True, duration_seconds=60)
        assert pr.allow_multiple is True
        assert pr.duration_seconds == 60


class TestContentBlock:
    def test_defaults(self):
        cb = ContentBlock()
        assert cb.type == ContentType.TEXT
        assert cb.text == ""
        assert cb.url == ""
        assert cb.mime_type == ""
        assert cb.metadata == {}

    def test_metadata_isolation(self):
        a = ContentBlock()
        b = ContentBlock()
        a.metadata["k"] = "v"
        assert "k" not in b.metadata


class TestInboundEvent:
    def test_defaults(self):
        ev = InboundEvent()
        assert len(ev.event_id) == 16
        assert ev.event_type == EventType.MESSAGE
        assert ev.channel == ""
        assert ev.content == []
        assert ev.reply_to_id is None
        assert ev.thread_id is None
        assert ev.session_key_override is None

    def test_unique_event_ids(self):
        ids = {InboundEvent().event_id for _ in range(50)}
        assert len(ids) == 50

    def test_session_key_default(self):
        ev = make_inbound(channel="qq", chat_id="room42")
        assert ev.session_key == "qq:room42"

    def test_session_key_override(self):
        ev = make_inbound(session_key_override="custom-key")
        assert ev.session_key == "custom-key"

    def test_text_property_single_block(self):
        ev = make_inbound(content=[ContentBlock(text="hello")])
        assert ev.text == "hello"

    def test_text_property_multiple_blocks(self):
        ev = make_inbound(content=[
            ContentBlock(text="line1"),
            ContentBlock(text=""),
            ContentBlock(text="line2"),
        ])
        assert ev.text == "line1\nline2"

    def test_text_property_empty(self):
        ev = make_inbound(content=[])
        assert ev.text == ""

    def test_media_urls_excludes_text_type(self):
        ev = make_inbound(content=[
            ContentBlock(type=ContentType.TEXT, url="http://a.txt"),
            ContentBlock(type=ContentType.IMAGE, url="http://img.png"),
            ContentBlock(type=ContentType.AUDIO, url="http://audio.mp3"),
        ])
        assert ev.media_urls == ["http://img.png", "http://audio.mp3"]

    def test_media_urls_skips_empty_url(self):
        ev = make_inbound(content=[
            ContentBlock(type=ContentType.IMAGE, url=""),
        ])
        assert ev.media_urls == []

    def test_text_message_factory(self):
        ev = InboundEvent.text_message("qq", "u1", "c1", "hi")
        assert ev.channel == "qq"
        assert ev.sender_id == "u1"
        assert ev.chat_id == "c1"
        assert ev.text == "hi"
        assert len(ev.content) == 1
        assert ev.content[0].type == ContentType.TEXT

    def test_text_message_factory_kwargs(self):
        ev = InboundEvent.text_message(
            "qq", "u1", "c1", "hi",
            event_type=EventType.CLI,
            thread_id="t1",
        )
        assert ev.event_type == EventType.CLI
        assert ev.thread_id == "t1"


class TestOutboundEvent:
    def test_defaults(self):
        ev = OutboundEvent()
        assert len(ev.event_id) == 16
        assert ev.is_final is True
        assert ev.message_kind == "final"
        assert ev.task_id is None
        assert ev.workflow_id is None

    def test_text_property(self):
        ev = make_outbound(content=[
            ContentBlock(text="a"),
            ContentBlock(text="b"),
        ])
        assert ev.text == "a\nb"

    def test_text_reply_factory(self):
        ev = OutboundEvent.text_reply("qq", "c1", "bye", reply_to_id="msg1")
        assert ev.channel == "qq"
        assert ev.chat_id == "c1"
        assert ev.text == "bye"
        assert ev.reply_to_id == "msg1"

    def test_text_reply_factory_no_reply_to(self):
        ev = OutboundEvent.text_reply("qq", "c1", "ok")
        assert ev.reply_to_id is None


class TestFromTextWithMedia:
    def test_no_tags_returns_single_text_block(self):
        ev = OutboundEvent.from_text_with_media("qq", "c1", "plain text")
        assert len(ev.content) == 1
        assert ev.content[0].type == ContentType.TEXT
        assert ev.content[0].text == "plain text"

    def test_qqimg_tag(self):
        raw = "<qqimg>http://example.com/pic.png</qqimg>"
        ev = OutboundEvent.from_text_with_media("qq", "c1", raw)
        assert len(ev.content) == 1
        assert ev.content[0].type == ContentType.IMAGE
        assert ev.content[0].url == "http://example.com/pic.png"

    def test_qqvoice_tag(self):
        raw = "<qqvoice>http://example.com/a.mp3</qqvoice>"
        ev = OutboundEvent.from_text_with_media("qq", "c1", raw)
        assert ev.content[0].type == ContentType.AUDIO
        assert ev.content[0].url == "http://example.com/a.mp3"

    def test_qqvideo_tag(self):
        raw = "<qqvideo>http://example.com/v.mp4</qqvideo>"
        ev = OutboundEvent.from_text_with_media("qq", "c1", raw)
        assert ev.content[0].type == ContentType.VIDEO

    def test_qqfile_tag(self):
        raw = "<qqfile>http://example.com/f.zip</qqfile>"
        ev = OutboundEvent.from_text_with_media("qq", "c1", raw)
        assert ev.content[0].type == ContentType.FILE

    def test_qqmedia_tag(self):
        raw = "<qqmedia>http://example.com/m.bin</qqmedia>"
        ev = OutboundEvent.from_text_with_media("qq", "c1", raw)
        assert ev.content[0].type == ContentType.FILE

    def test_mixed_text_and_media(self):
        raw = "Hello <qqimg>http://img.png</qqimg> world"
        ev = OutboundEvent.from_text_with_media("qq", "c1", raw)
        assert len(ev.content) == 3
        assert ev.content[0].type == ContentType.TEXT
        assert ev.content[0].text == "Hello"
        assert ev.content[1].type == ContentType.IMAGE
        assert ev.content[1].url == "http://img.png"
        assert ev.content[2].type == ContentType.TEXT
        assert ev.content[2].text == "world"

    def test_multiple_media_tags(self):
        raw = "<qqimg>http://a.png</qqimg><qqvoice>http://b.mp3</qqvoice>"
        ev = OutboundEvent.from_text_with_media("qq", "c1", raw)
        assert len(ev.content) == 2
        assert ev.content[0].type == ContentType.IMAGE
        assert ev.content[1].type == ContentType.AUDIO

    def test_reply_to_id_passed_through(self):
        ev = OutboundEvent.from_text_with_media("qq", "c1", "hi", reply_to_id="r1")
        assert ev.reply_to_id == "r1"

    def test_kwargs_forwarded(self):
        ev = OutboundEvent.from_text_with_media(
            "qq", "c1", "hi", is_final=False, task_id="t1"
        )
        assert ev.is_final is False
        assert ev.task_id == "t1"


# ===================================================================
# queue.py tests
# ===================================================================


@pytest.fixture
def bus():
    return MessageBus(max_queue_size=5)


class TestMessageBusPublishInbound:
    @pytest.mark.asyncio
    async def test_publish_inbound_enqueues(self, bus):
        ev = make_inbound()
        await bus.publish_inbound(ev)
        assert bus.pending_inbound == 1

    @pytest.mark.asyncio
    async def test_queue_full_drops_event(self):
        small_bus = MessageBus(max_queue_size=2)
        await small_bus.publish_inbound(make_inbound())
        await small_bus.publish_inbound(make_inbound())
        # Third should be silently dropped
        await small_bus.publish_inbound(make_inbound())
        assert small_bus.pending_inbound == 2


class TestMessageBusSubscribeInbound:
    @pytest.mark.asyncio
    async def test_inbound_handler_called(self, bus):
        handler = AsyncMock()
        bus.subscribe_inbound(handler)
        ev = make_inbound()
        await bus.publish_inbound(ev)

        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        handler.assert_called_once_with(ev)

    @pytest.mark.asyncio
    async def test_inbound_handler_exception_does_not_stop_dispatch(self, bus):
        bad_handler = AsyncMock(side_effect=RuntimeError("boom"))
        good_handler = AsyncMock()
        bus.subscribe_inbound(bad_handler)
        bus.subscribe_inbound(good_handler)

        await bus.publish_inbound(make_inbound())
        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        bad_handler.assert_called_once()
        good_handler.assert_called_once()


class TestMessageBusPublishOutbound:
    @pytest.mark.asyncio
    async def test_channel_handler_called(self, bus):
        handler = AsyncMock()
        bus.subscribe_outbound("qq", handler)
        ev = make_outbound(channel="qq")
        await bus.publish_outbound(ev)
        handler.assert_called_once_with(ev)

    @pytest.mark.asyncio
    async def test_no_handler_logs_warning(self, bus):
        # No handlers registered; should not raise
        ev = make_outbound(channel="unknown")
        await bus.publish_outbound(ev)

    @pytest.mark.asyncio
    async def test_global_handler_called(self, bus):
        global_h = AsyncMock()
        bus.subscribe_outbound_global(global_h)
        bus.subscribe_outbound("qq", AsyncMock())
        ev = make_outbound(channel="qq")
        await bus.publish_outbound(ev)
        global_h.assert_called_once_with(ev)

    @pytest.mark.asyncio
    async def test_drop_metadata_skips_channel_handlers(self, bus):
        global_h = AsyncMock()
        channel_h = AsyncMock()
        bus.subscribe_outbound_global(global_h)
        bus.subscribe_outbound("qq", channel_h)

        ev = make_outbound(channel="qq", metadata={"_drop": True})
        await bus.publish_outbound(ev)

        global_h.assert_called_once_with(ev)
        channel_h.assert_not_called()

    @pytest.mark.asyncio
    async def test_outbound_handler_exception_isolated(self, bus):
        bad = AsyncMock(side_effect=RuntimeError("fail"))
        good = AsyncMock()
        bus.subscribe_outbound("qq", bad)
        bus.subscribe_outbound("qq", good)

        ev = make_outbound(channel="qq")
        await bus.publish_outbound(ev)

        bad.assert_called_once()
        good.assert_called_once()

    @pytest.mark.asyncio
    async def test_global_handler_exception_isolated(self, bus):
        bad_global = AsyncMock(side_effect=ValueError("oops"))
        channel_h = AsyncMock()
        bus.subscribe_outbound_global(bad_global)
        bus.subscribe_outbound("qq", channel_h)

        ev = make_outbound(channel="qq")
        await bus.publish_outbound(ev)

        bad_global.assert_called_once()
        channel_h.assert_called_once()


class TestMessageBusLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_dispatch_task(self, bus):
        await bus.start()
        assert bus._dispatch_task is not None
        assert not bus._dispatch_task.done()
        await bus.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_dispatch_task(self, bus):
        await bus.start()
        task = bus._dispatch_task
        await bus.stop()
        assert task.done()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, bus):
        # Should not raise
        await bus.stop()

    @pytest.mark.asyncio
    async def test_pending_inbound_reflects_queue_size(self, bus):
        assert bus.pending_inbound == 0
        await bus.publish_inbound(make_inbound())
        assert bus.pending_inbound == 1
        await bus.publish_inbound(make_inbound())
        assert bus.pending_inbound == 2
