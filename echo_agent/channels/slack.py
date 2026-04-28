"""Slack channel — Socket Mode WebSocket + Web API."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
from loguru import logger

from echo_agent.bus.events import OutboundEvent
from echo_agent.bus.queue import MessageBus
from echo_agent.channels.base import BaseChannel, SendResult
from echo_agent.config.schema import SlackChannelConfig

_API_BASE = "https://slack.com/api"


class SlackChannel(BaseChannel):
    name = "slack"
    supports_edit = True

    def __init__(self, config: SlackChannelConfig, bus: MessageBus):
        super().__init__(config, bus)
        self._bot_token = config.bot_token
        self._app_token = config.app_token
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_task: asyncio.Task | None = None
        self._bot_id: str = ""

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        auth = await self._api("auth.test", token=self._bot_token)
        if auth and auth.get("ok"):
            self._bot_id = auth.get("user_id", "")
            logger.info("Slack bot: {} ({})", auth.get("user", ""), self._bot_id)
        self._running = True
        self.bus.subscribe_outbound(self.name, self.send)
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("Slack channel started")

    async def stop(self) -> None:
        self._running = False
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
        if not text:
            return None
        payload: dict[str, Any] = {
            "channel": event.chat_id,
            "text": text,
        }
        thread_ts = event.metadata.get("thread_ts")
        if thread_ts:
            payload["thread_ts"] = thread_ts
        result = await self._api("chat.postMessage", token=self._bot_token, json_body=payload)
        if result and result.get("ok"):
            return SendResult(success=True, message_id=str(result.get("ts", "")))
        error = str(result.get("error", "Slack chat.postMessage failed")) if result else "Slack chat.postMessage failed"
        return SendResult(success=False, error=error)

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
        result = await self._api(
            "chat.update",
            token=self._bot_token,
            json_body={
                "channel": chat_id,
                "ts": message_id,
                "text": text,
            },
        )
        if result and result.get("ok"):
            return SendResult(success=True, message_id=str(result.get("ts") or message_id))
        error = str(result.get("error", "Slack chat.update failed")) if result else "Slack chat.update failed"
        return SendResult(success=False, message_id=message_id, error=error)

    async def send_typing(self, chat_id: str, metadata: dict[str, Any] | None = None) -> None:
        thread_ts = (metadata or {}).get("thread_ts")
        if not thread_ts:
            return
        await self._api(
            "assistant.threads.setStatus",
            token=self._bot_token,
            json_body={"channel_id": chat_id, "thread_ts": thread_ts, "status": "is thinking..."},
        )

    async def stop_typing(self, chat_id: str) -> None:
        pass

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not getattr(self.config, "reactions_enabled", True):
            return SendResult(success=False, error="reactions disabled")
        result = await self._api(
            "reactions.add",
            token=self._bot_token,
            json_body={"channel": chat_id, "timestamp": message_id, "name": emoji},
        )
        if result and result.get("ok"):
            return SendResult(success=True)
        error = str(result.get("error", "Slack reactions.add failed")) if result else "Slack reactions.add failed"
        return SendResult(success=False, error=error)

    async def remove_reaction(self, chat_id: str, message_id: str, emoji: str, metadata: dict[str, Any] | None = None) -> SendResult:
        if not getattr(self.config, "reactions_enabled", True):
            return SendResult(success=False, error="reactions disabled")
        result = await self._api(
            "reactions.remove",
            token=self._bot_token,
            json_body={"channel": chat_id, "timestamp": message_id, "name": emoji},
        )
        if result and result.get("ok"):
            return SendResult(success=True)
        error = str(result.get("error", "Slack reactions.remove failed")) if result else "Slack reactions.remove failed"
        return SendResult(success=False, error=error)

    async def delete_message(self, chat_id: str, message_id: str, metadata: dict[str, Any] | None = None) -> SendResult:
        result = await self._api(
            "chat.delete",
            token=self._bot_token,
            json_body={"channel": chat_id, "ts": message_id},
        )
        if result and result.get("ok"):
            return SendResult(success=True)
        error = str(result.get("error", "Slack chat.delete failed")) if result else "Slack chat.delete failed"
        return SendResult(success=False, error=error)

    async def send_voice(self, chat_id: str, audio_source: str, metadata: dict[str, Any] | None = None) -> SendResult:
        from pathlib import Path
        path = Path(audio_source)
        if not path.exists() or not self._session:
            return SendResult(success=False, error="audio file not found or no session")
        try:
            upload_result = await self._api(
                "files.getUploadURLExternal",
                token=self._bot_token,
                json_body={"filename": path.name, "length": path.stat().st_size},
            )
            if not upload_result or not upload_result.get("ok"):
                return SendResult(success=False, error="failed to get upload URL")
            upload_url = upload_result["upload_url"]
            file_id = upload_result["file_id"]
            data = aiohttp.FormData()
            data.add_field("file", path.open("rb"), filename=path.name)
            async with self._session.post(upload_url, data=data) as resp:
                if resp.status >= 400:
                    return SendResult(success=False, error="upload failed")
            complete_result = await self._api(
                "files.completeUploadExternal",
                token=self._bot_token,
                json_body={"files": [{"id": file_id}], "channel_id": chat_id},
            )
            if complete_result and complete_result.get("ok"):
                return SendResult(success=True)
            return SendResult(success=False, error="Slack file upload complete failed")
        except Exception as e:
            return SendResult(success=False, error=str(e))

    # ── Socket Mode ──────────────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Slack WS error: {}", e)
            if self._running:
                await asyncio.sleep(5)

    async def _connect_and_listen(self) -> None:
        if not self._session:
            return
        ws_url = await self._get_ws_url()
        if not ws_url:
            logger.error("Failed to get Slack Socket Mode URL")
            await asyncio.sleep(10)
            return

        self._ws = await self._session.ws_connect(ws_url)
        logger.info("Slack Socket Mode connected")

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                await self._handle_ws_event(data)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

    async def _handle_ws_event(self, data: dict[str, Any]) -> None:
        envelope_id = data.get("envelope_id")
        if envelope_id and self._ws and not self._ws.closed:
            await self._ws.send_json({"envelope_id": envelope_id})

        evt_type = data.get("type")
        if evt_type == "events_api":
            payload = data.get("payload", {})
            event = payload.get("event", {})
            await self._on_event(event)
        elif evt_type == "disconnect":
            logger.info("Slack requested disconnect, reconnecting...")
            if self._ws and not self._ws.closed:
                await self._ws.close()

    async def _on_event(self, event: dict[str, Any]) -> None:
        if event.get("type") != "message":
            return
        if event.get("subtype"):
            return
        if event.get("bot_id"):
            return

        sender_id = event.get("user", "")
        channel_id = event.get("channel", "")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts") or event.get("ts", "")

        if not text:
            return

        media: list[dict[str, str]] = []
        for f in event.get("files", []):
            url = f.get("url_private", "")
            if url:
                mimetype = f.get("mimetype", "")
                kind = "image" if mimetype.startswith("image") else "file"
                media.append({"type": kind, "url": url})

        await self._handle_message(
            sender_id=sender_id,
            chat_id=channel_id,
            text=text,
            media=media if media else None,
            metadata={"thread_ts": thread_ts},
        )

    async def _get_ws_url(self) -> str | None:
        result = await self._api("apps.connections.open", token=self._app_token)
        if result and result.get("ok"):
            return result.get("url")
        return None

    async def _api(self, method: str, token: str = "", json_body: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self._session:
            return None
        url = f"{_API_BASE}/{method}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            if json_body:
                headers["Content-Type"] = "application/json; charset=utf-8"
                async with self._session.post(url, json=json_body, headers=headers) as resp:
                    return await resp.json()
            else:
                async with self._session.post(url, headers=headers) as resp:
                    return await resp.json()
        except Exception as e:
            logger.error("Slack API {} failed: {}", method, e)
            return None
