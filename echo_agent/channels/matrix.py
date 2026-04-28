"""Matrix/Element channel — long-polling sync + REST API."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
from loguru import logger

from echo_agent.bus.events import OutboundEvent
from echo_agent.bus.queue import MessageBus
from echo_agent.channels.base import BaseChannel, SendResult
from echo_agent.config.schema import MatrixChannelConfig
from echo_agent.bus.events import PollRequest


class MatrixChannel(BaseChannel):
    name = "matrix"

    def __init__(self, config: MatrixChannelConfig, bus: MessageBus):
        super().__init__(config, bus)
        self._homeserver = config.homeserver.rstrip("/")
        self._user_id = config.user_id
        self._access_token = config.access_token
        self._allow_rooms = set(config.allow_rooms) if config.allow_rooms else None
        self._session: aiohttp.ClientSession | None = None
        self._sync_task: asyncio.Task | None = None
        self._since: str = ""

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(headers={
            "Authorization": f"Bearer {self._access_token}",
        })
        self._running = True
        self.bus.subscribe_outbound(self.name, self.send)
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Matrix channel started ({})", self._homeserver)

    async def stop(self) -> None:
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()

    async def send(self, event: OutboundEvent) -> None:
        text = event.text or ""
        if not text or not self._session:
            return
        room_id = event.chat_id
        txn_id = f"m{id(text)}{asyncio.get_event_loop().time():.0f}"
        url = f"{self._homeserver}/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn_id}"
        payload = {"msgtype": "m.text", "body": text}
        try:
            async with self._session.put(url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning("Matrix send failed ({}): {}", resp.status, body[:200])
        except Exception as e:
            logger.error("Matrix send error: {}", e)

    async def send_typing(self, chat_id: str, metadata: dict[str, Any] | None = None) -> None:
        if not self._session:
            return
        url = f"{self._homeserver}/_matrix/client/v3/rooms/{chat_id}/typing/{self._user_id}"
        try:
            async with self._session.put(url, json={"typing": True, "timeout": 30000}) as resp:
                if resp.status >= 400:
                    logger.warning("Matrix typing failed ({})", resp.status)
        except Exception as e:
            logger.error("Matrix typing error: {}", e)

    async def stop_typing(self, chat_id: str) -> None:
        if not self._session:
            return
        url = f"{self._homeserver}/_matrix/client/v3/rooms/{chat_id}/typing/{self._user_id}"
        try:
            async with self._session.put(url, json={"typing": False}) as resp:
                if resp.status >= 400:
                    logger.warning("Matrix stop typing failed ({})", resp.status)
        except Exception as e:
            logger.error("Matrix stop typing error: {}", e)

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not getattr(self.config, "reactions_enabled", True) or not self._session:
            return SendResult(success=False, error="reactions disabled or no session")
        txn_id = f"r{id(emoji)}{asyncio.get_event_loop().time():.0f}"
        url = f"{self._homeserver}/_matrix/client/v3/rooms/{chat_id}/send/m.reaction/{txn_id}"
        payload = {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": message_id,
                "key": emoji,
            }
        }
        try:
            async with self._session.put(url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    return SendResult(success=False, error=body[:200])
                data = await resp.json()
                return SendResult(success=True, message_id=data.get("event_id", ""))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_read_receipt(self, chat_id: str, message_id: str, metadata: dict[str, Any] | None = None) -> None:
        if not self._session or not message_id:
            return
        url = f"{self._homeserver}/_matrix/client/v3/rooms/{chat_id}/receipt/m.read/{message_id}"
        try:
            async with self._session.post(url, json={}) as resp:
                if resp.status >= 400:
                    logger.warning("Matrix read receipt failed ({})", resp.status)
        except Exception as e:
            logger.error("Matrix read receipt error: {}", e)

    async def delete_message(self, chat_id: str, message_id: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not self._session:
            return SendResult(success=False, error="no session")
        txn_id = f"d{id(message_id)}{asyncio.get_event_loop().time():.0f}"
        url = f"{self._homeserver}/_matrix/client/v3/rooms/{chat_id}/redact/{message_id}/{txn_id}"
        try:
            async with self._session.put(url, json={}) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    return SendResult(success=False, error=body[:200])
                return SendResult(success=True)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_poll(self, chat_id: str, poll: PollRequest, metadata: dict[str, Any] | None = None) -> SendResult:
        if not self._session:
            return SendResult(success=False, error="no session")
        txn_id = f"p{id(poll)}{asyncio.get_event_loop().time():.0f}"
        url = f"{self._homeserver}/_matrix/client/v3/rooms/{chat_id}/send/m.poll.start/{txn_id}"
        answers = [{"id": str(i), "org.matrix.msc3381.v2.text": o} for i, o in enumerate(poll.options)]
        payload = {
            "org.matrix.msc3381.v2.poll": {
                "kind": "org.matrix.msc3381.v2.disclosed",
                "max_selections": len(poll.options) if poll.allow_multiple else 1,
                "question": {"org.matrix.msc3381.v2.text": poll.question},
                "answers": answers,
            }
        }
        try:
            async with self._session.put(url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    return SendResult(success=False, error=body[:200])
                data = await resp.json()
                return SendResult(success=True, message_id=data.get("event_id", ""))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_voice(self, chat_id: str, audio_source: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not self._session:
            return SendResult(success=False, error="no session")
        txn_id = f"v{id(audio_source)}{asyncio.get_event_loop().time():.0f}"
        url = f"{self._homeserver}/_matrix/client/v3/rooms/{chat_id}/send/m.room.message/{txn_id}"
        payload = {
            "msgtype": "m.audio",
            "body": "voice message",
            "url": audio_source,
        }
        try:
            async with self._session.put(url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    return SendResult(success=False, error=body[:200])
                data = await resp.json()
                return SendResult(success=True, message_id=data.get("event_id", ""))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def _sync_loop(self) -> None:
        initial = await self._do_sync(timeout_ms=0)
        if initial:
            self._since = initial.get("next_batch", "")

        while self._running:
            try:
                data = await self._do_sync(timeout_ms=30000)
                if not data:
                    continue
                self._since = data.get("next_batch", self._since)
                rooms = data.get("rooms", {}).get("join", {})
                for room_id, room_data in rooms.items():
                    if self._allow_rooms and room_id not in self._allow_rooms:
                        continue
                    for evt in room_data.get("timeline", {}).get("events", []):
                        await self._on_event(room_id, evt)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Matrix sync error: {}", e)
                await asyncio.sleep(5)

    async def _do_sync(self, timeout_ms: int = 30000) -> dict[str, Any] | None:
        if not self._session:
            return None
        params: dict[str, Any] = {
            "timeout": timeout_ms,
            "filter": json.dumps({"room": {"timeline": {"limit": 20}}}),
        }
        if self._since:
            params["since"] = self._since
        url = f"{self._homeserver}/_matrix/client/v3/sync"
        try:
            async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=timeout_ms / 1000 + 30)) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.warning("Matrix sync failed ({})", resp.status)
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error("Matrix sync error: {}", e)
        return None

    async def _on_event(self, room_id: str, evt: dict[str, Any]) -> None:
        if evt.get("type") != "m.room.message":
            return
        sender = evt.get("sender", "")
        if sender == self._user_id:
            return
        content = evt.get("content", {})
        msgtype = content.get("msgtype", "")
        text = ""
        media: list[dict[str, str]] = []

        if msgtype == "m.text":
            text = content.get("body", "")
        elif msgtype == "m.image":
            mxc = content.get("url", "")
            if mxc:
                media.append({"type": "image", "url": mxc})
            text = content.get("body", "")
        elif msgtype == "m.file":
            mxc = content.get("url", "")
            if mxc:
                media.append({"type": "file", "url": mxc})
            text = content.get("body", "")

        if not text and not media:
            return

        await self._handle_message(
            sender_id=sender, chat_id=room_id, text=text,
            media=media if media else None,
            reply_to_id=evt.get("event_id"),
            metadata={"msgtype": msgtype},
        )
