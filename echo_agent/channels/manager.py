"""Channel manager — lifecycle management for all channel adapters."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from echo_agent.bus.events import InboundEvent, OutboundEvent
from echo_agent.bus.queue import MessageBus
from echo_agent.channels.base import BaseChannel
from echo_agent.channels.cli import CLIChannel
from echo_agent.channels.cron import CronChannel
from echo_agent.channels.dingtalk import DingTalkChannel
from echo_agent.channels.discord import DiscordChannel
from echo_agent.channels.email import EmailChannel
from echo_agent.channels.feishu import FeishuChannel
from echo_agent.channels.matrix import MatrixChannel
from echo_agent.channels.qqbot import QQBotChannel
from echo_agent.channels.slack import SlackChannel
from echo_agent.channels.telegram import TelegramChannel
from echo_agent.channels.wecom import WeComChannel
from echo_agent.channels.webhook import WebhookChannel
from echo_agent.channels.weixin import WeixinChannel
from echo_agent.channels.whatsapp import WhatsAppChannel
from echo_agent.config.schema import ChannelsConfig

_CHANNEL_REGISTRY: dict[str, type[BaseChannel]] = {
    "cli": CLIChannel,
    "webhook": WebhookChannel,
    "cron": CronChannel,
    "telegram": TelegramChannel,
    "discord": DiscordChannel,
    "slack": SlackChannel,
    "whatsapp": WhatsAppChannel,
    "weixin": WeixinChannel,
    "qqbot": QQBotChannel,
    "feishu": FeishuChannel,
    "dingtalk": DingTalkChannel,
    "email": EmailChannel,
    "wecom": WeComChannel,
    "matrix": MatrixChannel,
}


_STREAM_CURSOR = " ..."
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class _StreamState:
    raw_text: str = ""
    message_id: str = ""
    rendered_text: str = ""
    failed: bool = False


def register_channel_type(name: str, cls: type[BaseChannel]) -> None:
    _CHANNEL_REGISTRY[name] = cls


_EMOJI_MAP: dict[str, dict[str, str]] = {
    "processing": {"telegram": "\U0001f440", "discord": "\U0001f440", "slack": "eyes", "matrix": "\U0001f440"},
    "success": {"telegram": "\U0001f44d", "discord": "✅", "slack": "white_check_mark", "matrix": "✅"},
    "failure": {"telegram": "❌", "discord": "❌", "slack": "x", "matrix": "❌"},
}


def _emoji(channel_name: str, kind: str) -> str:
    return _EMOJI_MAP.get(kind, {}).get(channel_name, "")


class ChannelManager:
    """Manages the lifecycle of all enabled channel adapters."""

    def __init__(self, config: ChannelsConfig, bus: MessageBus, on_cli_exit: Callable[[], None] | None = None):
        self.config = config
        self.bus = bus
        self._channels: dict[str, BaseChannel] = {}
        self._send_progress = config.send_progress
        self._send_tool_hints = config.send_tool_hints
        self._on_cli_exit = on_cli_exit
        self._stream_states: dict[str, _StreamState] = {}
        self._inbound_msg_ids: dict[str, tuple[str, str]] = {}
        self.bus.subscribe_outbound_global(self._filter_and_dispatch)
        self.bus.subscribe_inbound(self._on_inbound_lifecycle)

    def get_channel(self, name: str) -> BaseChannel | None:
        return self._channels.get(name)

    @property
    def active_channels(self) -> list[str]:
        return [name for name, ch in self._channels.items() if ch.is_running]

    async def _on_inbound_lifecycle(self, event: InboundEvent) -> None:
        channel = self._channels.get(event.channel)
        if not channel:
            return
        if event.reply_to_id:
            self._inbound_msg_ids[event.event_id] = (event.channel, event.reply_to_id)
        try:
            await channel.send_typing(event.chat_id, metadata=event.metadata)
        except Exception as e:
            logger.debug("send_typing failed on {}: {}", event.channel, e)
        if event.reply_to_id:
            try:
                await channel.send_read_receipt(event.chat_id, event.reply_to_id, metadata=event.metadata)
            except Exception as e:
                logger.debug("send_read_receipt failed on {}: {}", event.channel, e)
            emoji = _emoji(event.channel, "processing")
            if emoji and getattr(getattr(channel, "config", None), "reactions_enabled", False):
                try:
                    await channel.send_reaction(event.chat_id, event.reply_to_id, emoji, metadata=event.metadata)
                except Exception as e:
                    logger.debug("send_reaction failed on {}: {}", event.channel, e)

    async def _filter_and_dispatch(self, event: OutboundEvent) -> None:
        if event.metadata.get("_token_stream"):
            await self._handle_token_stream(event)
            if event.metadata.get("_drop"):
                return

        if event.metadata.get("_progress"):
            is_tool_hint = event.metadata.get("_tool_hint", False)
            if is_tool_hint and not self._send_tool_hints:
                event.metadata["_drop"] = True
                return
            if not is_tool_hint and not self._send_progress:
                event.metadata["_drop"] = True
                return

        if event.is_final and event.message_kind == "final":
            await self._on_outbound_final(event)

    async def _on_outbound_final(self, event: OutboundEvent) -> None:
        channel = self._channels.get(event.channel)
        if not channel:
            return
        try:
            await channel.stop_typing(event.chat_id)
        except Exception as e:
            logger.debug("stop_typing failed on {}: {}", event.channel, e)
        inbound_event_id = str(event.metadata.get("_inbound_event_id", ""))
        mapping = self._inbound_msg_ids.pop(inbound_event_id, None)
        if not mapping:
            return
        _, platform_msg_id = mapping
        if not getattr(getattr(channel, "config", None), "reactions_enabled", False):
            return
        is_error = event.metadata.get("_error", False)
        processing_emoji = _emoji(event.channel, "processing")
        outcome_emoji = _emoji(event.channel, "failure" if is_error else "success")
        if processing_emoji:
            try:
                await channel.remove_reaction(event.chat_id, platform_msg_id, processing_emoji)
            except Exception as e:
                logger.debug("remove_reaction failed on {}: {}", event.channel, e)
        if outcome_emoji:
            try:
                await channel.send_reaction(event.chat_id, platform_msg_id, outcome_emoji)
            except Exception as e:
                logger.debug("send_reaction failed on {}: {}", event.channel, e)

    async def _handle_token_stream(self, event: OutboundEvent) -> None:
        channel = self._channels.get(event.channel)
        if channel is None:
            return

        if not channel.supports_edit:
            if not event.is_final:
                event.metadata["_drop"] = True
            return

        event.metadata["_drop"] = True
        stream_key = self._stream_key(event)
        state = self._stream_states.setdefault(stream_key, _StreamState())

        if event.metadata.get("_stream_full_text"):
            state.raw_text = event.text or ""
        else:
            state.raw_text += event.text or ""

        visible_text = self._visible_stream_text(state.raw_text, final=event.is_final)

        if state.failed:
            if event.is_final:
                await self._send_stream_fallback(channel, event, visible_text)
                self._stream_states.pop(stream_key, None)
            return

        rendered_text = self._render_stream_text(visible_text, final=event.is_final)
        if not rendered_text:
            if event.is_final:
                self._stream_states.pop(stream_key, None)
            return

        if rendered_text == state.rendered_text:
            if event.is_final:
                self._stream_states.pop(stream_key, None)
            return

        if state.message_id:
            result = await channel.edit_message(
                event.chat_id,
                state.message_id,
                rendered_text,
                metadata=self._public_metadata(event.metadata),
                finalize=event.is_final,
            )
        else:
            send_event = OutboundEvent.text_reply(
                channel=event.channel,
                chat_id=event.chat_id,
                text=rendered_text,
                reply_to_id=event.reply_to_id,
            )
            send_event.is_final = event.is_final
            send_event.message_kind = event.message_kind
            send_event.metadata = self._public_metadata(event.metadata)
            result = await channel.send(send_event)

        if not result or not result.success:
            error = result.error if result else "channel returned no send result"
            logger.warning("Token stream delivery failed on {}: {}", event.channel, error)
            state.failed = True
            if event.is_final:
                await self._send_stream_fallback(channel, event, visible_text)
                self._stream_states.pop(stream_key, None)
            return

        if result.message_id:
            state.message_id = result.message_id
        elif not event.is_final:
            logger.warning("Token stream send on {} did not return a message id", event.channel)
            state.failed = True
            return

        state.rendered_text = rendered_text
        if event.is_final:
            self._stream_states.pop(stream_key, None)

    async def _send_stream_fallback(self, channel: BaseChannel, event: OutboundEvent, text: str) -> None:
        final_text = self._render_stream_text(text, final=True)
        if not final_text:
            return
        fallback = OutboundEvent.text_reply(
            channel=event.channel,
            chat_id=event.chat_id,
            text=final_text,
            reply_to_id=event.reply_to_id,
        )
        fallback.is_final = True
        fallback.message_kind = "final"
        fallback.metadata = self._public_metadata(event.metadata)
        result = await channel.send(fallback)
        if result and not result.success:
            logger.warning("Token stream fallback send failed on {}: {}", event.channel, result.error)

    @staticmethod
    def _stream_key(event: OutboundEvent) -> str:
        stream_id = str(event.metadata.get("_inbound_event_id") or event.event_id)
        return f"{event.channel}:{event.chat_id}:{stream_id}"

    @staticmethod
    def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in metadata.items() if not key.startswith("_")}

    @staticmethod
    def _visible_stream_text(text: str, *, final: bool) -> str:
        if not text:
            return ""
        visible = _THINK_BLOCK_RE.sub("", text)
        visible = _THINK_CLOSE_RE.sub("", visible)
        visible = _THINK_OPEN_RE.sub("", visible)
        if not final:
            visible = ChannelManager._trim_partial_think_marker(visible)
            return visible
        return visible.strip()

    @staticmethod
    def _trim_partial_think_marker(text: str) -> str:
        marker = "<think>"
        lower = text.lower()
        for length in range(len(marker) - 1, 0, -1):
            if lower.endswith(marker[:length]):
                return text[:-length]
        return text

    @staticmethod
    def _render_stream_text(text: str, *, final: bool) -> str:
        rendered = text.strip() if final else text.rstrip()
        if final:
            return rendered
        if not rendered:
            return ""
        return f"{rendered}{_STREAM_CURSOR}"

    async def start_all(self) -> None:
        for name, cls in _CHANNEL_REGISTRY.items():
            channel_config = getattr(self.config, name, None)
            if channel_config is None:
                continue
            if not getattr(channel_config, "enabled", False):
                continue
            try:
                if cls is CLIChannel:
                    channel = cls(channel_config, self.bus, on_exit=self._on_cli_exit)
                else:
                    channel = cls(channel_config, self.bus)
                if self.config.transcription_api_key:
                    channel.transcription_api_key = self.config.transcription_api_key
                await channel.start()
                self._channels[name] = channel
                if channel.is_running:
                    logger.info("Channel {} started", name)
                else:
                    logger.info("Channel {} inactive", name)
            except Exception as e:
                logger.error("Failed to start channel {}: {}", name, e)

    async def stop_all(self) -> None:
        for name, channel in self._channels.items():
            try:
                await channel.stop()
                logger.info("Channel {} stopped", name)
            except Exception as e:
                logger.error("Failed to stop channel {}: {}", name, e)
        self._channels.clear()
