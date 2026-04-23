"""Channel manager — lifecycle management for all channel adapters."""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger

from echo_agent.bus.events import OutboundEvent
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
from echo_agent.channels.wechat import WeChatChannel
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
    "wechat": WeChatChannel,
    "weixin": WeixinChannel,
    "qqbot": QQBotChannel,
    "feishu": FeishuChannel,
    "dingtalk": DingTalkChannel,
    "email": EmailChannel,
    "wecom": WeComChannel,
    "matrix": MatrixChannel,
}


def register_channel_type(name: str, cls: type[BaseChannel]) -> None:
    _CHANNEL_REGISTRY[name] = cls


class ChannelManager:
    """Manages the lifecycle of all enabled channel adapters."""

    def __init__(self, config: ChannelsConfig, bus: MessageBus, on_cli_exit: Callable[[], None] | None = None):
        self.config = config
        self.bus = bus
        self._channels: dict[str, BaseChannel] = {}
        self._send_progress = config.send_progress
        self._send_tool_hints = config.send_tool_hints
        self._on_cli_exit = on_cli_exit
        self.bus.subscribe_outbound_global(self._filter_and_dispatch)

    def get_channel(self, name: str) -> BaseChannel | None:
        return self._channels.get(name)

    @property
    def active_channels(self) -> list[str]:
        return [name for name, ch in self._channels.items() if ch.is_running]

    async def _filter_and_dispatch(self, event: OutboundEvent) -> None:
        if event.metadata.get("_progress"):
            is_tool_hint = event.metadata.get("_tool_hint", False)
            if is_tool_hint and not self._send_tool_hints:
                event.metadata["_drop"] = True
                return
            if not is_tool_hint and not self._send_progress:
                event.metadata["_drop"] = True
                return

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
