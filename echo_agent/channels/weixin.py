"""Weixin (personal WeChat) channel — iLink Bot API with long-polling.

Connects to personal WeChat accounts via Tencent's iLink Bot API.
No public endpoint required; uses HTTP long-polling for inbound messages.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import struct
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiohttp
from loguru import logger

from echo_agent.bus.events import ContentBlock, ContentType, OutboundEvent
from echo_agent.bus.queue import MessageBus
from echo_agent.channels.base import BaseChannel
from echo_agent.config.schema import WeixinChannelConfig

try:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False

# ── iLink API constants ─────────────────────────────────────────────────────

_CHANNEL_VERSION = "2.2.0"
_APP_ID = "bot"
_APP_CLIENT_VERSION = (2 << 16) | (2 << 8) | 0

_EP_GET_UPDATES = "ilink/bot/getupdates"
_EP_SEND_MESSAGE = "ilink/bot/sendmessage"
_EP_GET_BOT_QR = "ilink/bot/get_bot_qrcode"
_EP_GET_QR_STATUS = "ilink/bot/get_qrcode_status"

_LONG_POLL_TIMEOUT_MS = 35_000
_API_TIMEOUT_MS = 15_000
_MAX_CONSECUTIVE_FAILURES = 3
_RETRY_DELAY = 2
_BACKOFF_DELAY = 30
_SESSION_EXPIRED_ERRCODE = -14
_DEDUP_TTL = 300
_MAX_MESSAGE_LENGTH = 4000

_ITEM_TEXT = 1
_ITEM_IMAGE = 2
_ITEM_VOICE = 3
_ITEM_FILE = 4
_ITEM_VIDEO = 5
_MSG_TYPE_BOT = 2
_MSG_STATE_FINISH = 2


# ── Helpers ──────────────────────────────────────────────────────────────────

def _random_uin() -> str:
    value = struct.unpack(">I", secrets.token_bytes(4))[0]
    return base64.b64encode(str(value).encode()).decode("ascii")


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _headers(token: str | None, body: str) -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Content-Length": str(len(body.encode())),
        "X-WECHAT-UIN": _random_uin(),
        "iLink-App-Id": _APP_ID,
        "iLink-App-ClientVersion": str(_APP_CLIENT_VERSION),
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _login_headers() -> dict[str, str]:
    return {
        "X-WECHAT-UIN": _random_uin(),
        "iLink-App-Id": _APP_ID,
        "iLink-App-ClientVersion": str(_APP_CLIENT_VERSION),
    }


def _pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)


def _aes128_ecb_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography package required for media decryption")
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    if not padded:
        return padded
    pad_len = padded[-1]
    if 1 <= pad_len <= 16 and padded.endswith(bytes([pad_len]) * pad_len):
        return padded[:-pad_len]
    return padded


def _parse_aes_key(aes_key_b64: str) -> bytes:
    decoded = base64.b64decode(aes_key_b64)
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32:
        text = decoded.decode("ascii", errors="ignore")
        if all(ch in "0123456789abcdefABCDEF" for ch in text):
            return bytes.fromhex(text)
    raise ValueError(f"unexpected aes_key format ({len(decoded)} decoded bytes)")


# ── API helpers ──────────────────────────────────────────────────────────────

async def _api_post(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    token: str,
    timeout_ms: int,
) -> dict[str, Any]:
    body = _json_dumps({"base_info": {"channel_version": _CHANNEL_VERSION}, **payload})
    url = f"{base_url}/{endpoint}"
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000 + 5)
    async with session.post(url, data=body.encode(), headers=_headers(token, body), timeout=timeout) as resp:
        return await resp.json(content_type=None)


async def _get_updates(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    sync_buf: str,
    timeout_ms: int,
) -> dict[str, Any]:
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=_EP_GET_UPDATES,
        payload={"get_updates_buf": sync_buf, "longpolling_timeout_ms": timeout_ms},
        token=token,
        timeout_ms=timeout_ms,
    )


async def _send_message(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    to: str,
    text: str,
    context_token: str | None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "from_user_id": "",
        "to_user_id": to,
        "client_id": uuid.uuid4().hex,
        "message_type": _MSG_TYPE_BOT,
        "message_state": _MSG_STATE_FINISH,
        "item_list": [{"type": _ITEM_TEXT, "text_item": {"text": text}}],
    }
    if context_token:
        msg["context_token"] = context_token
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=_EP_SEND_MESSAGE,
        payload={"msg": msg},
        token=token,
        timeout_ms=_API_TIMEOUT_MS,
    )
# PLACEHOLDER_WEIXIN_DEDUP


# ── Deduplicator & ContextTokenStore ─────────────────────────────────────────

class _MessageDeduplicator:
    def __init__(self, ttl: float = _DEDUP_TTL):
        self._ttl = ttl
        self._seen: dict[str, float] = {}

    def is_duplicate(self, message_id: str) -> bool:
        now = time.time()
        self._seen = {k: v for k, v in self._seen.items() if now - v < self._ttl}
        if message_id in self._seen:
            return True
        self._seen[message_id] = now
        return False


class _ContextTokenStore:
    def __init__(self, data_dir: Path):
        self._dir = data_dir
        self._cache: dict[str, str] = {}

    def _path(self, account_id: str) -> Path:
        return self._dir / f"{account_id}.context-tokens.json"

    def restore(self, account_id: str) -> None:
        path = self._path(account_id)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for uid, tok in data.items():
                if isinstance(tok, str) and tok:
                    self._cache[f"{account_id}:{uid}"] = tok
        except Exception as exc:
            logger.warning("weixin: failed to restore context tokens: {}", exc)

    def get(self, account_id: str, user_id: str) -> str | None:
        return self._cache.get(f"{account_id}:{user_id}")

    def set(self, account_id: str, user_id: str, token: str) -> None:
        self._cache[f"{account_id}:{user_id}"] = token
        self._persist(account_id)

    def _persist(self, account_id: str) -> None:
        prefix = f"{account_id}:"
        payload = {k[len(prefix):]: v for k, v in self._cache.items() if k.startswith(prefix)}
        try:
            path = self._path(account_id)
            path.write_text(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            logger.warning("weixin: failed to persist context tokens: {}", exc)


def _extract_text(item_list: list[dict[str, Any]]) -> str:
    for item in item_list:
        if item.get("type") == _ITEM_TEXT:
            text = str((item.get("text_item") or {}).get("text") or "")
            ref = item.get("ref_msg") or {}
            ref_item = ref.get("message_item") or {}
            if ref_item:
                ref_text_item = ref_item if ref_item.get("type") == _ITEM_TEXT else None
                if ref_text_item:
                    inner = str((ref_text_item.get("text_item") or {}).get("text") or "")
                    title = ref.get("title") or ""
                    parts = [p for p in [title, inner] if p]
                    if parts:
                        return f"[引用: {' | '.join(parts)}]\n{text}".strip()
            return text
    for item in item_list:
        if item.get("type") == _ITEM_VOICE:
            return str((item.get("voice_item") or {}).get("text") or "")
    return ""


def _guess_chat_type(message: dict[str, Any], account_id: str) -> tuple[str, str]:
    room_id = str(message.get("room_id") or message.get("chat_room_id") or "").strip()
    if room_id:
        return "group", room_id
    return "dm", str(message.get("from_user_id") or "")


def _media_reference(item: dict[str, Any], key: str) -> dict[str, Any]:
    return (item.get(key) or {}).get("media") or {}


def _load_sync_buf(data_dir: Path, account_id: str) -> str:
    path = data_dir / f"{account_id}.sync.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
        return str(data.get("sync_buf") or "")
    except Exception as e:
        logger.debug("Failed to load sync buffer for {}: {}", account_id, e)
        return ""


def _save_sync_buf(data_dir: Path, account_id: str, sync_buf: str) -> None:
    path = data_dir / f"{account_id}.sync.json"
    try:
        path.write_text(json.dumps({"sync_buf": sync_buf}))
    except Exception as exc:
        logger.warning("weixin: failed to save sync_buf: {}", exc)


# ── Channel implementation ───────────────────────────────────────────────────

class WeixinChannel(BaseChannel):
    name = "weixin"

    def __init__(self, config: WeixinChannelConfig, bus: MessageBus):
        super().__init__(config, bus)
        self._account_id = config.account_id
        self._token = config.token
        self._base_url = config.base_url.rstrip("/")
        self._cdn_base_url = config.cdn_base_url.rstrip("/")
        self._dm_policy = config.dm_policy
        data_dir = config.data_dir or os.path.expanduser("~/.echo-agent/weixin")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._dedup = _MessageDeduplicator()
        self._token_store = _ContextTokenStore(self._data_dir)
        self._poll_session: aiohttp.ClientSession | None = None
        self._send_session: aiohttp.ClientSession | None = None
        self._poll_task: asyncio.Task | None = None

    async def start(self) -> None:
        if not self._token or not self._account_id:
            logger.error("weixin: account_id and token are required")
            return
        self._poll_session = aiohttp.ClientSession(trust_env=True)
        self._send_session = aiohttp.ClientSession(trust_env=True)
        self._token_store.restore(self._account_id)
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop(), name="weixin-poll")
        self.bus.subscribe_outbound(self.name, self.send)
        logger.info("weixin channel started, account={}", self._account_id[:8] if self._account_id else "?")

    async def stop(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None
        if self._poll_session and not self._poll_session.closed:
            await self._poll_session.close()
        self._poll_session = None
        if self._send_session and not self._send_session.closed:
            await self._send_session.close()
        self._send_session = None
        logger.info("weixin channel stopped")

    # ── Send ─────────────────────────────────────────────────────────────────

    async def send(self, event: OutboundEvent) -> None:
        text = event.text or ""
        if not text or not self._send_session or not self._token:
            return
        chat_id = event.chat_id
        context_token = self._token_store.get(self._account_id, chat_id)
        chunks = self._split_text(text)
        for chunk in chunks:
            try:
                resp = await _send_message(
                    self._send_session,
                    base_url=self._base_url,
                    token=self._token,
                    to=chat_id,
                    text=chunk,
                    context_token=context_token,
                )
                errcode = resp.get("errcode", 0)
                if errcode and errcode != 0:
                    logger.warning("weixin send error: errcode={} errmsg={}", errcode, resp.get("errmsg", ""))
            except Exception as exc:
                logger.error("weixin send failed to {}: {}", chat_id[:8] if chat_id else "?", exc)
            if len(chunks) > 1:
                await asyncio.sleep(0.3)

    @staticmethod
    def _split_text(text: str) -> list[str]:
        if len(text) <= _MAX_MESSAGE_LENGTH:
            return [text]
        chunks = []
        while text:
            chunks.append(text[:_MAX_MESSAGE_LENGTH])
            text = text[_MAX_MESSAGE_LENGTH:]
        return chunks

    # ── Poll loop ────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        assert self._poll_session is not None
        sync_buf = _load_sync_buf(self._data_dir, self._account_id)
        timeout_ms = _LONG_POLL_TIMEOUT_MS
        consecutive_failures = 0

        while self._running:
            try:
                response = await _get_updates(
                    self._poll_session,
                    base_url=self._base_url,
                    token=self._token,
                    sync_buf=sync_buf,
                    timeout_ms=timeout_ms,
                )
                suggested = response.get("longpolling_timeout_ms")
                if isinstance(suggested, int) and suggested > 0:
                    timeout_ms = suggested

                ret = response.get("ret", 0)
                errcode = response.get("errcode", 0)
                if ret not in (0, None) or errcode not in (0, None):
                    if ret == _SESSION_EXPIRED_ERRCODE or errcode == _SESSION_EXPIRED_ERRCODE:
                        logger.error("weixin: session expired, pausing 10 minutes")
                        await asyncio.sleep(600)
                        consecutive_failures = 0
                        continue
                    consecutive_failures += 1
                    delay = _BACKOFF_DELAY if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES else _RETRY_DELAY
                    logger.warning("weixin: poll error ret={} errcode={} ({}/{})", ret, errcode, consecutive_failures, _MAX_CONSECUTIVE_FAILURES)
                    await asyncio.sleep(delay)
                    if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        consecutive_failures = 0
                    continue

                consecutive_failures = 0
                new_sync_buf = str(response.get("get_updates_buf") or "")
                if new_sync_buf:
                    sync_buf = new_sync_buf
                    _save_sync_buf(self._data_dir, self._account_id, sync_buf)

                for message in response.get("msgs") or []:
                    asyncio.create_task(self._process_message_safe(message))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                consecutive_failures += 1
                delay = _BACKOFF_DELAY if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES else _RETRY_DELAY
                logger.error("weixin: poll exception ({}/{}): {}", consecutive_failures, _MAX_CONSECUTIVE_FAILURES, exc)
                await asyncio.sleep(delay)
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0

    # ── Message processing ───────────────────────────────────────────────────

    async def _process_message_safe(self, message: dict[str, Any]) -> None:
        try:
            await self._process_message(message)
        except Exception as exc:
            logger.error("weixin: unhandled inbound error: {}", exc, exc_info=True)

    async def _process_message(self, message: dict[str, Any]) -> None:
        sender_id = str(message.get("from_user_id") or "").strip()
        if not sender_id or sender_id == self._account_id:
            return

        message_id = str(message.get("message_id") or "").strip()
        if message_id and self._dedup.is_duplicate(message_id):
            return

        chat_type, effective_chat_id = _guess_chat_type(message, self._account_id)
        if chat_type == "group":
            return
        if self._dm_policy == "disabled":
            return
        if self._dm_policy == "allowlist" and not self.is_allowed(sender_id):
            return

        context_token = str(message.get("context_token") or "").strip()
        if context_token:
            self._token_store.set(self._account_id, sender_id, context_token)

        item_list = message.get("item_list") or []
        text = _extract_text(item_list)

        media: list[dict[str, str]] = []
        for item in item_list:
            m = self._extract_media_info(item)
            if m:
                media.append(m)

        if not text and not media:
            return

        await self._handle_message(
            sender_id=sender_id,
            chat_id=effective_chat_id,
            text=text,
            media=media if media else None,
            metadata={"message_id": message_id, "chat_type": chat_type},
        )

    def _extract_media_info(self, item: dict[str, Any]) -> dict[str, str] | None:
        item_type = item.get("type")
        if item_type == _ITEM_IMAGE:
            media = _media_reference(item, "image_item")
            url = media.get("full_url") or ""
            if url:
                return {"type": "image", "url": url}
        elif item_type == _ITEM_FILE:
            media = _media_reference(item, "file_item")
            url = media.get("full_url") or ""
            name = (item.get("file_item") or {}).get("file_name") or "file"
            if url:
                return {"type": "file", "url": url, "mime_type": name}
        return None

    # ── QR login (static, for CLI use) ───────────────────────────────────────

    @staticmethod
    async def qr_login(
        base_url: str = "https://ilinkai.weixin.qq.com",
        timeout_seconds: int = 480,
    ) -> dict[str, str] | None:
        base_url = base_url.rstrip("/")
        async with aiohttp.ClientSession(trust_env=True) as session:
            url = f"{base_url}/{_EP_GET_BOT_QR}"
            async with session.get(url, params={"bot_type": "3"}, headers=_login_headers()) as resp:
                if resp.status != 200:
                    logger.error("weixin: get_bot_qrcode HTTP {}: {}", resp.status, await resp.text())
                    return None
                data = await resp.json(content_type=None)

            errcode = data.get("errcode") or data.get("err_code")
            if errcode:
                logger.error("weixin: get_bot_qrcode returned error {}: {}", errcode, data.get("errmsg") or data.get("err_msg") or data)

            qrcode = data.get("qrcode")
            qr_url = data.get("qrcode_img_content")
            if not qrcode:
                logger.error("weixin: failed to get QR code, response: {}", data)
                return None

            print(f"\nScan this QR code with WeChat:\n{qr_url}\n")

            deadline = time.time() + timeout_seconds
            refresh_count = 0
            while time.time() < deadline:
                await asyncio.sleep(2)
                check_url = f"{base_url}/{_EP_GET_QR_STATUS}"
                async with session.get(check_url, params={"qrcode": qrcode}, headers=_login_headers()) as resp:
                    status_data = await resp.json(content_type=None)

                status = status_data.get("status", "")
                if status == "confirmed":
                    return {
                        "account_id": str(status_data.get("account_id") or status_data.get("ilink_bot_id") or ""),
                        "token": str(status_data.get("token") or status_data.get("bot_token") or ""),
                        "base_url": str(status_data.get("base_url") or status_data.get("baseurl") or base_url),
                        "user_id": str(status_data.get("user_id") or status_data.get("ilink_user_id") or ""),
                    }
                elif status == "expired":
                    refresh_count += 1
                    if refresh_count >= 3:
                        logger.error("weixin: QR code expired too many times")
                        return None
                    async with session.get(url, params={"bot_type": "3"}, headers=_login_headers()) as resp:
                        data = await resp.json(content_type=None)
                    qrcode = data.get("qrcode")
                    qr_url = data.get("qrcode_img_content")
                    if qrcode:
                        print(f"\nQR expired, scan new one:\n{qr_url}\n")
                elif status == "scaned":
                    pass

            logger.error("weixin: QR login timed out")
            return None
