"""Discord channel — WebSocket gateway + REST API."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
from loguru import logger

from echo_agent.bus.events import OutboundEvent
from echo_agent.bus.queue import MessageBus
from echo_agent.channels.base import BaseChannel, SendResult
from echo_agent.config.schema import DiscordChannelConfig
from echo_agent.utils.text import split_message

_GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"
_API_BASE = "https://discord.com/api/v10"
_MAX_TEXT = 2000
_INTENTS = (1 << 0) | (1 << 9) | (1 << 15)  # GUILDS | GUILD_MESSAGES | MESSAGE_CONTENT


class DiscordChannel(BaseChannel):
    name = "discord"
    supports_edit = True

    def __init__(self, config: DiscordChannelConfig, bus: MessageBus):
        super().__init__(config, bus)
        self._token = config.token
        self._group_policy = config.group_policy
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._heartbeat_interval: float = 41.25
        self._seq: int | None = None
        self._session_id: str = ""
        self._bot_id: str = ""
        self._resume_url: str = ""
        self._typing_tasks: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(headers={
            "Authorization": f"Bot {self._token}",
        })
        self._running = True
        self.bus.subscribe_outbound(self.name, self.send)
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("Discord channel started")

    async def stop(self) -> None:
        self._running = False
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()

    async def send(self, event: OutboundEvent) -> SendResult | None:
        text = event.text or ""
        if not text or not self._session:
            return None
        url = f"{_API_BASE}/channels/{event.chat_id}/messages"
        first_result: SendResult | None = None
        for chunk in _chunk_text(text, _MAX_TEXT):
            payload: dict[str, Any] = {"content": chunk}
            if event.reply_to_id:
                payload["message_reference"] = {"message_id": event.reply_to_id}
            try:
                async with self._session.post(url, json=payload) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        logger.warning("Discord send failed ({}): {}", resp.status, body[:200])
                        send_result = SendResult(success=False, error=body[:200])
                    else:
                        data = await resp.json()
                        send_result = SendResult(success=True, message_id=str(data.get("id", "")))
            except Exception as e:
                logger.error("Discord send error: {}", e)
                send_result = SendResult(success=False, error=str(e))
            if first_result is None:
                first_result = send_result
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
        if not text or not self._session:
            return SendResult(success=False, message_id=message_id, error="empty text or missing session")
        if len(text) > _MAX_TEXT:
            return SendResult(success=False, message_id=message_id, error="message exceeds Discord edit limit")
        url = f"{_API_BASE}/channels/{chat_id}/messages/{message_id}"
        try:
            async with self._session.patch(url, json={"content": text}) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning("Discord edit failed ({}): {}", resp.status, body[:200])
                    return SendResult(success=False, message_id=message_id, error=body[:200])
                data = await resp.json()
                return SendResult(success=True, message_id=str(data.get("id") or message_id))
        except Exception as e:
            logger.error("Discord edit error: {}", e)
            return SendResult(success=False, message_id=message_id, error=str(e))

    async def send_typing(self, chat_id: str, metadata: dict[str, Any] | None = None) -> None:
        if not self._session or chat_id in self._typing_tasks:
            return

        async def _typing_loop() -> None:
            try:
                while self._running and self._session:
                    async with self._session.post(f"{_API_BASE}/channels/{chat_id}/typing") as resp:
                        if resp.status >= 400:
                            break
                    await asyncio.sleep(8)
            except asyncio.CancelledError:
                pass

        self._typing_tasks[chat_id] = asyncio.create_task(_typing_loop())

    async def stop_typing(self, chat_id: str) -> None:
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not getattr(self.config, "reactions_enabled", True) or not self._session:
            return SendResult(success=False, error="reactions disabled or no session")
        from urllib.parse import quote
        encoded_emoji = quote(emoji)
        url = f"{_API_BASE}/channels/{chat_id}/messages/{message_id}/reactions/{encoded_emoji}/@me"
        try:
            async with self._session.put(url) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    return SendResult(success=False, error=body[:200])
                return SendResult(success=True)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def remove_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not getattr(self.config, "reactions_enabled", True) or not self._session:
            return SendResult(success=False, error="reactions disabled or no session")
        from urllib.parse import quote
        encoded_emoji = quote(emoji)
        url = f"{_API_BASE}/channels/{chat_id}/messages/{message_id}/reactions/{encoded_emoji}/@me"
        try:
            async with self._session.delete(url) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    return SendResult(success=False, error=body[:200])
                return SendResult(success=True)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def delete_message(self, chat_id: str, message_id: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not self._session:
            return SendResult(success=False, error="no session")
        url = f"{_API_BASE}/channels/{chat_id}/messages/{message_id}"
        try:
            async with self._session.delete(url) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    return SendResult(success=False, error=body[:200])
                return SendResult(success=True)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_voice(self, chat_id: str, audio_source: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not self._session:
            return SendResult(success=False, error="no session")
        url = f"{_API_BASE}/channels/{chat_id}/messages"
        try:
            from pathlib import Path
            path = Path(audio_source)
            if path.exists():
                data = aiohttp.FormData()
                data.add_field("files[0]", path.open("rb"), filename=path.name, content_type="audio/ogg")
                data.add_field("payload_json", json.dumps({"flags": 8192}))
                async with self._session.post(url, data=data) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        return SendResult(success=False, error=body[:200])
                    result = await resp.json()
                    return SendResult(success=True, message_id=str(result.get("id", "")))
            else:
                return SendResult(success=False, error="audio file not found")
        except Exception as e:
            return SendResult(success=False, error=str(e))

    # ── WebSocket lifecycle ──────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Discord WS error: {}", e)
            if self._running:
                await asyncio.sleep(5)

    async def _connect_and_listen(self) -> None:
        if not self._session:
            return
        url = self._resume_url or _GATEWAY_URL
        self._ws = await self._session.ws_connect(url)

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_ws_message(json.loads(msg.data))
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

    async def _handle_ws_message(self, data: dict[str, Any]) -> None:
        op = data.get("op")
        seq = data.get("s")
        if seq is not None:
            self._seq = seq
        t = data.get("t")
        d = data.get("d", {})

        if op == 10:  # HELLO
            self._heartbeat_interval = d.get("heartbeat_interval", 41250) / 1000
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            if self._session_id:
                await self._send_ws({"op": 6, "d": {
                    "token": self._token, "session_id": self._session_id, "seq": self._seq,
                }})
            else:
                await self._send_ws({"op": 2, "d": {
                    "token": self._token, "intents": _INTENTS,
                    "properties": {"os": "linux", "browser": "echo-agent", "device": "echo-agent"},
                }})
        elif op == 0:  # DISPATCH
            if t == "READY":
                self._session_id = d.get("session_id", "")
                self._resume_url = d.get("resume_gateway_url", "")
                user = d.get("user", {})
                self._bot_id = str(user.get("id", ""))
                logger.info("Discord ready as {}", user.get("username", ""))
            elif t == "MESSAGE_CREATE":
                await self._on_message(d)
        elif op == 7:  # RECONNECT
            if self._ws and not self._ws.closed:
                await self._ws.close()
        elif op == 9:  # INVALID SESSION
            self._session_id = ""
            if self._ws and not self._ws.closed:
                await self._ws.close()
        elif op == 11:  # HEARTBEAT ACK
            pass

    async def _on_message(self, d: dict[str, Any]) -> None:
        author = d.get("author", {})
        if author.get("bot"):
            return
        sender_id = str(author.get("id", ""))
        channel_id = str(d.get("channel_id", ""))
        content = d.get("content", "")
        guild_id = d.get("guild_id")

        if guild_id and self._group_policy == "mention":
            if f"<@{self._bot_id}>" not in content and f"<@!{self._bot_id}>" not in content:
                ref = d.get("referenced_message")
                if not (ref and str(ref.get("author", {}).get("id", "")) == self._bot_id):
                    return
            content = content.replace(f"<@{self._bot_id}>", "").replace(f"<@!{self._bot_id}>", "").strip()

        if not content:
            return

        media: list[dict[str, str]] = []
        for att in d.get("attachments", []):
            url = att.get("url", "")
            ct = att.get("content_type", "")
            if ct.startswith("image"):
                media.append({"type": "image", "url": url})
            else:
                media.append({"type": "file", "url": url})

        await self._handle_message(
            sender_id=sender_id,
            chat_id=channel_id,
            text=content,
            media=media if media else None,
            reply_to_id=str(d.get("id", "")),
            thread_id=d.get("thread", {}).get("id") if d.get("thread") else None,
            metadata={"guild_id": guild_id or ""},
        )

    async def _heartbeat_loop(self) -> None:
        try:
            while self._running and self._ws and not self._ws.closed:
                await self._send_ws({"op": 1, "d": self._seq})
                await asyncio.sleep(self._heartbeat_interval)
        except asyncio.CancelledError:
            pass

    async def _send_ws(self, data: dict[str, Any]) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.send_json(data)


def _chunk_text(text: str, limit: int) -> list[str]:
    return split_message(text, limit)
