"""QQ Bot channel — Official API v2 WebSocket gateway.

Connects to QQ Bot via WebSocket for receiving messages and REST API for sending.
Supports C2C (private), group, and guild message types.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import aiohttp
from loguru import logger

from echo_agent.bus.events import OutboundEvent
from echo_agent.bus.queue import MessageBus
from echo_agent.channels.base import BaseChannel
from echo_agent.config.schema import QQBotChannelConfig

_API_BASE = "https://api.sgroup.qq.com"
_SANDBOX_API = "https://sandbox.api.sgroup.qq.com"
_TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"

_INTENTS = (1 << 25) | (1 << 30) | (1 << 12)
_MAX_MESSAGE_LENGTH = 4000
_DEDUP_TTL = 300
_SEND_RETRIES = 3
_RECONNECT_BACKOFFS = [2, 5, 10, 30, 60]
_AT_MENTION_RE = re.compile(r"<@!?\d+>\s*")


class QQBotChannel(BaseChannel):
    name = "qqbot"

    def __init__(self, config: QQBotChannelConfig, bus: MessageBus):
        super().__init__(config, bus)
        self._app_id = config.app_id
        self._app_secret = config.app_secret
        self._sandbox = config.sandbox
        self._markdown = config.markdown_support
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._access_token: str = ""
        self._token_expires: float = 0
        self._token_lock = asyncio.Lock()
        self._seq: int | None = None
        self._session_id: str = ""
        self._heartbeat_interval: float = 41.25
        self._msg_seq: int = 0
        self._seen_messages: dict[str, float] = {}
        self._chat_type_map: dict[str, str] = {}

    @property
    def _api_base(self) -> str:
        return _SANDBOX_API if self._sandbox else _API_BASE

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        await self._refresh_token()
        self._running = True
        self.bus.subscribe_outbound(self.name, self.send)
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("QQBot channel started")

    async def stop(self) -> None:
        self._running = False
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
        logger.info("QQBot channel stopped")

    # ── Dedup ────────────────────────────────────────────────────────────────

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        self._seen_messages = {k: v for k, v in self._seen_messages.items() if now - v < _DEDUP_TTL}
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    def _next_msg_seq(self) -> int:
        self._msg_seq += 1
        return self._msg_seq

    # ── Send ─────────────────────────────────────────────────────────────────

    async def send(self, event: OutboundEvent) -> None:
        text = event.text or ""
        if not text or not self._session:
            return
        await self._ensure_token()
        chat_id = event.chat_id
        msg_type = event.metadata.get("msg_type") or self._chat_type_map.get(chat_id, "group")
        msg_id = event.reply_to_id or ""
        chunks = self._split_text(text)
        for i, chunk in enumerate(chunks):
            reply = msg_id if i == 0 else ""
            await self._send_chunk(chat_id, chunk, msg_type, reply)
            if len(chunks) > 1 and i < len(chunks) - 1:
                await asyncio.sleep(0.5)

    async def _send_chunk(self, chat_id: str, text: str, msg_type: str, reply_to: str) -> None:
        seq = self._next_msg_seq()
        if msg_type == "channel":
            url = f"{self._api_base}/channels/{chat_id}/messages"
            payload: dict[str, Any] = {"content": text}
            if reply_to:
                payload["msg_id"] = reply_to
        elif msg_type == "c2c":
            url = f"{self._api_base}/v2/users/{chat_id}/messages"
            if self._markdown:
                payload = {"markdown": {"content": text}, "msg_type": 2, "msg_seq": seq}
            else:
                payload = {"content": text, "msg_type": 0, "msg_seq": seq}
            if reply_to:
                payload["msg_id"] = reply_to
        else:
            url = f"{self._api_base}/v2/groups/{chat_id}/messages"
            if self._markdown:
                payload = {"markdown": {"content": text}, "msg_type": 2, "msg_seq": seq}
            else:
                payload = {"content": text, "msg_type": 0, "msg_seq": seq}
            if reply_to:
                payload["msg_id"] = reply_to

        for attempt in range(_SEND_RETRIES):
            try:
                async with self._session.post(url, json=payload, headers=self._auth_headers()) as resp:
                    if resp.status < 400:
                        return
                    body = await resp.text()
                    if resp.status == 400 and "40034024" in body and "msg_id" in payload:
                        logger.info("QQBot msg_id expired, retrying without reply reference")
                        payload.pop("msg_id", None)
                        continue
                    if resp.status in (400, 401, 403, 404):
                        logger.warning("QQBot send permanent error ({}): {}", resp.status, body[:200])
                        return
                    logger.warning("QQBot send error ({}), retry {}/{}: {}", resp.status, attempt + 1, _SEND_RETRIES, body[:200])
            except Exception as e:
                logger.error("QQBot send exception, retry {}/{}: {}", attempt + 1, _SEND_RETRIES, e)
            if attempt < _SEND_RETRIES - 1:
                await asyncio.sleep(1 * (attempt + 1))

    @staticmethod
    def _split_text(text: str) -> list[str]:
        if len(text) <= _MAX_MESSAGE_LENGTH:
            return [text]
        chunks = []
        while text:
            chunks.append(text[:_MAX_MESSAGE_LENGTH])
            text = text[_MAX_MESSAGE_LENGTH:]
        return chunks

    # ── WebSocket ────────────────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        backoff_idx = 0
        while self._running:
            try:
                await self._connect_and_listen()
                backoff_idx = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("QQBot WS error: {}", e)
            if self._running:
                delay = _RECONNECT_BACKOFFS[min(backoff_idx, len(_RECONNECT_BACKOFFS) - 1)]
                logger.info("QQBot reconnecting in {}s", delay)
                await asyncio.sleep(delay)
                backoff_idx += 1

    async def _connect_and_listen(self) -> None:
        if not self._session:
            return
        await self._ensure_token()
        gw_url = await self._get_gateway()
        if not gw_url:
            logger.error("Failed to get QQBot gateway URL")
            return

        timeout = aiohttp.ClientTimeout(total=20)
        self._ws = await self._session.ws_connect(gw_url, timeout=timeout)

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_ws_message(json.loads(msg.data))
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

        close_code = self._ws.close_code if self._ws else None
        if close_code:
            self._handle_close_code(close_code)

    def _handle_close_code(self, code: int | None) -> None:
        if code == 4004:
            logger.warning("QQBot: invalid token (4004), will refresh")
            self._token_expires = 0
        elif code in (4006, 4007, 4009):
            logger.warning("QQBot: session invalid ({}), clearing", code)
            self._session_id = ""
            self._seq = None
        elif code == 4008:
            logger.warning("QQBot: rate limited (4008)")

    async def _handle_ws_message(self, data: dict[str, Any]) -> None:
        op = data.get("op")
        s = data.get("s")
        if isinstance(s, int) and (self._seq is None or s > self._seq):
            self._seq = s
        t = data.get("t")
        d = data.get("d", {})

        if op == 10:  # HELLO
            self._heartbeat_interval = d.get("heartbeat_interval", 41250) / 1000 * 0.8
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            if self._session_id and self._seq is not None:
                await self._send_ws({"op": 6, "d": {
                    "token": f"QQBot {self._access_token}",
                    "session_id": self._session_id,
                    "seq": self._seq,
                }})
            else:
                await self._send_ws({"op": 2, "d": {
                    "token": f"QQBot {self._access_token}",
                    "intents": _INTENTS,
                    "shard": [0, 1],
                    "properties": {"$os": "linux", "$browser": "echo-agent", "$device": "echo-agent"},
                }})
        elif op == 0:  # DISPATCH
            if t == "READY":
                self._session_id = d.get("session_id", "")
                logger.info("QQBot ready, session={}", self._session_id)
            elif t == "RESUMED":
                logger.info("QQBot session resumed")
            elif t in ("AT_MESSAGE_CREATE", "MESSAGE_CREATE", "GUILD_AT_MESSAGE_CREATE"):
                asyncio.create_task(self._on_channel_message(d))
            elif t == "GROUP_AT_MESSAGE_CREATE":
                asyncio.create_task(self._on_group_message(d))
            elif t in ("C2C_MESSAGE_CREATE", "DIRECT_MESSAGE_CREATE"):
                asyncio.create_task(self._on_c2c_message(d))
        elif op == 7:  # RECONNECT
            logger.info("QQBot: server requested reconnect")
            if self._ws and not self._ws.closed:
                await self._ws.close()
        elif op == 9:  # INVALID SESSION
            logger.warning("QQBot: invalid session")
            self._session_id = ""
            self._seq = None
            if self._ws and not self._ws.closed:
                await self._ws.close()
        elif op == 11:  # HEARTBEAT ACK
            pass

    # ── Event handlers ───────────────────────────────────────────────────────

    async def _on_channel_message(self, d: dict[str, Any]) -> None:
        author = d.get("author") or {}
        if author.get("bot"):
            return
        sender_id = str(author.get("id", ""))
        channel_id = str(d.get("channel_id", ""))
        content = str(d.get("content", "")).strip()
        msg_id = str(d.get("id", ""))
        if not msg_id or self._is_duplicate(msg_id):
            return
        content = _AT_MENTION_RE.sub("", content).strip()
        if not content:
            return
        self._chat_type_map[channel_id] = "channel"
        await self._handle_message(
            sender_id=sender_id, chat_id=channel_id, text=content,
            reply_to_id=msg_id, metadata={"msg_type": "channel", "guild_id": d.get("guild_id", "")},
        )

    async def _on_group_message(self, d: dict[str, Any]) -> None:
        author = d.get("author") or {}
        sender_id = str(author.get("member_openid", author.get("id", "")))
        group_id = str(d.get("group_openid", d.get("group_id", "")))
        content = str(d.get("content", "")).strip()
        msg_id = str(d.get("id", ""))
        if not msg_id or self._is_duplicate(msg_id):
            return
        content = _AT_MENTION_RE.sub("", content).strip()
        if not content:
            return
        self._chat_type_map[group_id] = "group"
        await self._handle_message(
            sender_id=sender_id, chat_id=group_id, text=content,
            reply_to_id=msg_id, metadata={"msg_type": "group"},
        )

    async def _on_c2c_message(self, d: dict[str, Any]) -> None:
        author = d.get("author") or {}
        sender_id = str(author.get("user_openid", author.get("id", "")))
        content = str(d.get("content", "")).strip()
        msg_id = str(d.get("id", ""))
        if not msg_id or self._is_duplicate(msg_id):
            return
        if not content:
            return
        self._chat_type_map[sender_id] = "c2c"
        await self._handle_message(
            sender_id=sender_id, chat_id=sender_id, text=content,
            reply_to_id=msg_id, metadata={"msg_type": "c2c"},
        )

    # ── Heartbeat ────────────────────────────────────────────────────────────

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

    # ── Auth ─────────────────────────────────────────────────────────────────

    async def _refresh_token(self) -> None:
        if not self._session:
            return
        async with self._token_lock:
            if time.time() < self._token_expires:
                return
            payload = {"appId": self._app_id, "clientSecret": self._app_secret}
            try:
                async with self._session.post(_TOKEN_URL, json=payload) as resp:
                    data = await resp.json()
                    self._access_token = data.get("access_token", "")
                    expires_in = int(data.get("expires_in", 7200))
                    self._token_expires = time.time() + expires_in - 60
                    logger.info("QQBot access token refreshed")
            except Exception as e:
                logger.error("QQBot token refresh failed: {}", e)

    async def _ensure_token(self) -> None:
        if time.time() >= self._token_expires:
            await self._refresh_token()

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"QQBot {self._access_token}"}

    async def _get_gateway(self) -> str | None:
        if not self._session:
            return None
        url = f"{self._api_base}/gateway"
        try:
            async with self._session.get(url, headers=self._auth_headers()) as resp:
                data = await resp.json()
                return data.get("url")
        except Exception as e:
            logger.error("QQBot gateway fetch failed: {}", e)
            return None
