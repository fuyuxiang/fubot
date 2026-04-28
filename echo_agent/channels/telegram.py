"""Telegram channel — Bot API with long polling."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
from loguru import logger

from echo_agent.bus.events import OutboundEvent
from echo_agent.bus.queue import MessageBus
from echo_agent.channels.base import BaseChannel, SendResult
from echo_agent.config.schema import TelegramChannelConfig
from echo_agent.utils.text import split_message
from echo_agent.bus.events import PollRequest

_API = "https://api.telegram.org/bot{token}/{method}"
_MAX_TEXT = 4096


class TelegramChannel(BaseChannel):
    name = "telegram"
    supports_edit = True

    def __init__(self, config: TelegramChannelConfig, bus: MessageBus):
        super().__init__(config, bus)
        self._token = config.token
        self._session: aiohttp.ClientSession | None = None
        self._poll_task: asyncio.Task | None = None
        self._offset = 0
        self._group_policy = config.group_policy
        self._bot_id: str = ""
        self._bot_username: str = ""

    async def start(self) -> None:
        connector = None
        if self.config.proxy:
            from aiohttp_socks import ProxyConnector
            connector = ProxyConnector.from_url(self.config.proxy)
        self._session = aiohttp.ClientSession(connector=connector)
        me = await self._api("getMe")
        if me:
            self._bot_id = str(me.get("id", ""))
            self._bot_username = me.get("username", "")
            logger.info("Telegram bot: @{}", self._bot_username)
        self._running = True
        self.bus.subscribe_outbound(self.name, self.send)
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Telegram channel started")

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()

    async def send(self, event: OutboundEvent) -> SendResult | None:
        text = event.text or ""
        if not text:
            return None
        chat_id = event.chat_id
        reply_to = event.reply_to_id
        first_result: SendResult | None = None
        for chunk in self._chunk_text(text, _MAX_TEXT):
            result = await self._api("sendMessage", json={
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                **({"reply_to_message_id": reply_to} if reply_to else {}),
            })
            send_result = self._send_result(result, "Telegram sendMessage failed")
            if first_result is None:
                first_result = send_result
            reply_to = None
        return first_result

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        finalize: bool = False,
    ) -> SendResult:
        if not text:
            return SendResult(success=False, message_id=message_id, error="empty text")
        result = await self._api("editMessageText", json={
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
        })
        return self._send_result(result, "Telegram editMessageText failed", fallback_message_id=message_id)

    async def send_typing(self, chat_id: str, metadata: dict[str, Any] | None = None) -> None:
        await self._api("sendChatAction", json={"chat_id": chat_id, "action": "typing"})

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not getattr(self.config, "reactions_enabled", True):
            return SendResult(success=False, error="reactions disabled")
        result = await self._api("setMessageReaction", json={
            "chat_id": chat_id,
            "message_id": message_id,
            "reaction": [{"type": "emoji", "emoji": emoji}],
        })
        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error="Telegram setMessageReaction failed")

    async def remove_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not getattr(self.config, "reactions_enabled", True):
            return SendResult(success=False, error="reactions disabled")
        result = await self._api("setMessageReaction", json={
            "chat_id": chat_id,
            "message_id": message_id,
            "reaction": [],
        })
        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error="Telegram remove reaction failed")

    async def send_poll(self, chat_id: str, poll: PollRequest, metadata: dict[str, Any] | None = None) -> SendResult:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "question": poll.question,
            "options": [{"text": o} for o in poll.options],
            "is_anonymous": False,
            "allows_multiple_answers": poll.allow_multiple,
        }
        if poll.duration_seconds and 5 <= poll.duration_seconds <= 600:
            payload["open_period"] = poll.duration_seconds
        result = await self._api("sendPoll", json=payload)
        return self._send_result(result, "Telegram sendPoll failed")

    async def delete_message(self, chat_id: str, message_id: str, metadata: dict[str, Any] | None = None) -> SendResult:
        result = await self._api("deleteMessage", json={
            "chat_id": chat_id,
            "message_id": message_id,
        })
        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error="Telegram deleteMessage failed")

    async def send_voice(self, chat_id: str, audio_source: str, metadata: dict[str, Any] | None = None) -> SendResult:
        result = await self._api("sendVoice", json={
            "chat_id": chat_id,
            "voice": audio_source,
        })
        return self._send_result(result, "Telegram sendVoice failed")

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                updates = await self._api("getUpdates", json={
                    "offset": self._offset,
                    "timeout": 30,
                    "allowed_updates": ["message"],
                })
                if not updates:
                    continue
                for update in updates:
                    self._offset = update["update_id"] + 1
                    await self._process_update(update)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Telegram poll error: {}", e)
                await asyncio.sleep(5)

    async def _process_update(self, update: dict[str, Any]) -> None:
        msg = update.get("message")
        if not msg:
            return
        chat = msg.get("chat", {})
        chat_id = str(chat.get("id", ""))
        sender = msg.get("from", {})
        sender_id = str(sender.get("id", ""))
        text = msg.get("text", "") or msg.get("caption", "") or ""

        if chat.get("type") in ("group", "supergroup") and self._group_policy == "mention":
            if not self._is_mentioned(msg, text):
                return

        media: list[dict[str, str]] = []
        for kind in ("photo", "document", "audio", "video", "voice"):
            if kind in msg:
                file_obj = msg[kind][-1] if kind == "photo" else msg[kind]
                file_id = file_obj.get("file_id", "")
                if file_id:
                    media.append({"type": "image" if kind == "photo" else kind, "url": file_id})

        if not text and not media:
            return

        await self._api("sendChatAction", json={"chat_id": chat_id, "action": "typing"})

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            text=text,
            media=media if media else None,
            reply_to_id=str(msg.get("message_id", "")),
            metadata={"chat_type": chat.get("type", "private")},
        )

    def _is_mentioned(self, msg: dict[str, Any], text: str) -> bool:
        if self._bot_username and f"@{self._bot_username}" in text:
            return True
        entities = msg.get("entities", [])
        for ent in entities:
            if ent.get("type") == "mention":
                mention = text[ent["offset"]:ent["offset"] + ent["length"]]
                if mention.lower() == f"@{self._bot_username.lower()}":
                    return True
        reply = msg.get("reply_to_message", {})
        if reply and str(reply.get("from", {}).get("id", "")) == self._bot_id:
            return True
        return False

    async def _api(self, method: str, **kwargs: Any) -> Any:
        if not self._session:
            return None
        url = _API.format(token=self._token, method=method)
        try:
            async with self._session.post(url, **kwargs) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    logger.warning("Telegram API {}: {}", method, data.get("description", ""))
                    return None
                return data.get("result")
        except Exception as e:
            logger.error("Telegram API {} failed: {}", method, e)
            return None

    @staticmethod
    def _send_result(
        result: Any,
        error: str,
        *,
        fallback_message_id: str = "",
    ) -> SendResult:
        if isinstance(result, dict):
            message_id = str(result.get("message_id") or fallback_message_id)
            return SendResult(success=True, message_id=message_id)
        return SendResult(success=False, message_id=fallback_message_id, error=error)

    @staticmethod
    def _chunk_text(text: str, limit: int) -> list[str]:
        return split_message(text, limit)
