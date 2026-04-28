"""Base channel interface — all platform adapters implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from echo_agent.bus.events import InboundEvent, OutboundEvent, ContentBlock, ContentType, PollRequest
from echo_agent.bus.queue import MessageBus


@dataclass
class SendResult:
    success: bool
    message_id: str = ""
    error: str = ""


class BaseChannel(ABC):
    """Abstract base for chat channel implementations.

    Each channel (Telegram, Discord, Webhook, CLI, Cron, etc.) implements this
    to integrate with the message bus.
    """

    name: str = "base"
    transcription_api_key: str = ""
    supports_edit: bool = False

    def __init__(self, config: Any, bus: MessageBus):
        self.config = config
        self.bus = bus
        self._running = False
        self.transcription_api_key = getattr(config, "transcription_api_key", "") or ""

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop and clean up resources."""

    @abstractmethod
    async def send(self, event: OutboundEvent) -> SendResult | None:
        """Send a message through this channel."""

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        finalize: bool = False,
    ) -> SendResult:
        """Edit an existing platform message when the channel supports it."""
        return SendResult(success=False, error=f"channel {self.name} does not support message editing")

    async def send_typing(self, chat_id: str, metadata: dict[str, Any] | None = None) -> None:
        pass

    async def stop_typing(self, chat_id: str) -> None:
        pass

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        return SendResult(success=False, error=f"channel {self.name} does not support reactions")

    async def remove_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        return SendResult(success=False, error=f"channel {self.name} does not support reactions")

    async def send_poll(self, chat_id: str, poll: PollRequest, metadata: dict[str, Any] | None = None) -> SendResult:
        return SendResult(success=False, error=f"channel {self.name} does not support polls")

    async def delete_message(self, chat_id: str, message_id: str, metadata: dict[str, Any] | None = None) -> SendResult:
        return SendResult(success=False, error=f"channel {self.name} does not support message deletion")

    async def send_read_receipt(self, chat_id: str, message_id: str, metadata: dict[str, Any] | None = None) -> None:
        pass

    async def send_voice(self, chat_id: str, audio_source: str, metadata: dict[str, Any] | None = None) -> SendResult:
        return SendResult(success=False, error=f"channel {self.name} does not support voice messages")

    def is_allowed(self, sender_id: str) -> bool:
        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list:
            return True
        if "*" in allow_list:
            return True
        return str(sender_id) in allow_list

    def _build_event(
        self,
        sender_id: str,
        chat_id: str,
        text: str,
        media: list[dict[str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        reply_to_id: str | None = None,
        thread_id: str | None = None,
    ) -> InboundEvent:
        if not self.is_allowed(sender_id):
            logger.warning("Access denied for sender {} on channel {}", sender_id, self.name)
            raise PermissionError(f"sender {sender_id} is not allowed on channel {self.name}")

        content_blocks = [ContentBlock(type=ContentType.TEXT, text=text)]
        for item in (media or []):
            content_blocks.append(ContentBlock(
                type=ContentType(item.get("type", "file")),
                url=item.get("url", ""),
                mime_type=item.get("mime_type", ""),
            ))

        return InboundEvent(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content_blocks,
            reply_to_id=reply_to_id,
            thread_id=thread_id,
            session_key_override=session_key,
            metadata=metadata or {},
        )

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        text: str,
        media: list[dict[str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        reply_to_id: str | None = None,
        thread_id: str | None = None,
    ) -> InboundEvent | None:
        try:
            event = self._build_event(
                sender_id=sender_id,
                chat_id=chat_id,
                text=text,
                media=media,
                metadata=metadata,
                session_key=session_key,
                reply_to_id=reply_to_id,
                thread_id=thread_id,
            )
        except PermissionError:
            return None
        await self.bus.publish_inbound(event)
        return event

    @property
    def is_running(self) -> bool:
        return self._running

    async def transcribe_audio(self, file_path: str | Path) -> str:
        """Transcribe audio file via Groq Whisper API."""
        api_key = self.transcription_api_key
        if not api_key:
            return ""
        path = Path(file_path)
        if not path.exists():
            return ""
        try:
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            data = aiohttp.FormData()
            data.add_field("file", path.open("rb"), filename=path.name)
            data.add_field("model", "whisper-large-v3")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers={"Authorization": f"Bearer {api_key}"}) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("text", "")
                    logger.warning("Transcription failed ({}): {}", resp.status, await resp.text())
        except Exception as e:
            logger.error("Audio transcription error: {}", e)
        return ""
