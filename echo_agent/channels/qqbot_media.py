"""QQ Bot media helpers — type detection, tag parsing, upload cache, and API calls."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import re
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

# ── Constants ────────────────────────────────────────────────────────────────

_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
_UPLOAD_MAX_RETRIES = 2
_UPLOAD_BASE_DELAY = 1.0
_SEND_RETRIES = 3


class MediaFileType(IntEnum):
    IMAGE = 1
    VIDEO = 2
    VOICE = 3
    FILE = 4


# ── Media type detection ─────────────────────────────────────────────────────

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"})
AUDIO_EXTS = frozenset({
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a",
    ".wma", ".opus", ".amr", ".silk", ".slk", ".pcm",
})

IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}

# ── Media type detection functions ────────────────────────────────────────────


def get_clean_extension(file_path: str) -> str:
    clean = file_path.split("?")[0].split("#")[0]
    dot = clean.rfind(".")
    if dot < 0:
        return ""
    return clean[dot:].lower()


def is_image_file(path: str, mime_type: str = "") -> bool:
    if mime_type.startswith("image/"):
        return True
    return get_clean_extension(path) in IMAGE_EXTS


def is_video_file(path: str, mime_type: str = "") -> bool:
    if mime_type.startswith("video/"):
        return True
    return get_clean_extension(path) in VIDEO_EXTS


def is_audio_file(path: str, mime_type: str = "") -> bool:
    if mime_type and (
        mime_type.startswith("audio/") or mime_type == "voice"
        or "silk" in mime_type or "amr" in mime_type
    ):
        return True
    return get_clean_extension(path) in AUDIO_EXTS


def detect_media_kind(file_path: str, mime_type: str = "") -> str:
    if is_audio_file(file_path, mime_type):
        return "voice"
    if is_video_file(file_path, mime_type):
        return "video"
    if is_image_file(file_path, mime_type):
        return "image"
    return "file"


def media_kind_to_file_type(kind: str) -> MediaFileType:
    return {
        "image": MediaFileType.IMAGE,
        "voice": MediaFileType.VOICE,
        "video": MediaFileType.VIDEO,
        "file": MediaFileType.FILE,
    }.get(kind, MediaFileType.FILE)


def is_http_source(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def is_data_source(source: str) -> bool:
    return source.startswith("data:")


def is_local_path(source: str) -> bool:
    return not is_http_source(source) and not is_data_source(source)


# ── Media tag parsing ─────────────────────────────────────────────────────────

VALID_TAGS = ("qqimg", "qqvoice", "qqvideo", "qqfile", "qqmedia")

TAG_ALIASES: dict[str, str] = {
    "qq_img": "qqimg", "qqimage": "qqimg", "qq_image": "qqimg",
    "qqpic": "qqimg", "qq_pic": "qqimg", "qqpicture": "qqimg",
    "qq_picture": "qqimg", "qqphoto": "qqimg", "qq_photo": "qqimg",
    "img": "qqimg", "image": "qqimg", "pic": "qqimg",
    "picture": "qqimg", "photo": "qqimg",
    "qq_voice": "qqvoice", "qqaudio": "qqvoice", "qq_audio": "qqvoice",
    "voice": "qqvoice", "audio": "qqvoice",
    "qq_video": "qqvideo", "video": "qqvideo",
    "qq_file": "qqfile", "qqdoc": "qqfile", "qq_doc": "qqfile",
    "file": "qqfile", "doc": "qqfile", "document": "qqfile",
    "qq_media": "qqmedia", "media": "qqmedia", "attachment": "qqmedia",
    "attach": "qqmedia", "qqattachment": "qqmedia", "qq_attachment": "qqmedia",
    "qqsend": "qqmedia", "qq_send": "qqmedia", "send": "qqmedia",
}

_ALL_TAG_NAMES = sorted(
    list(VALID_TAGS) + list(TAG_ALIASES.keys()), key=len, reverse=True,
)
_TAG_NAME_PATTERN = "|".join(re.escape(t) for t in _ALL_TAG_NAMES)

_LB = r"(?:[<＜]|&lt;)"
_RB = r"(?:[>＞]|&gt;)"

SELF_CLOSING_TAG_RE = re.compile(
    r"`?" + _LB + r"\s*(" + _TAG_NAME_PATTERN + r")"
    r'(?:\s+(?!file|src|path|url)[a-z_-]+\s*=\s*["\']?[^"\'\s<>＜＞]*?["\']?)*'
    r'\s+(?:file|src|path|url)\s*=\s*["\']?'
    r'([^"\'\s>＞]+?)'
    r'["\']?'
    r'(?:\s+[a-z_-]+\s*=\s*["\']?[^"\'\s<>＜＞]*?["\']?)*'
    r"\s*/?\s*" + _RB + r"`?",
    re.IGNORECASE,
)

FUZZY_MEDIA_TAG_RE = re.compile(
    r"`?" + _LB + r"\s*(" + _TAG_NAME_PATTERN + r")\s*" + _RB
    + r'["\']?\s*'
    + r'([^<＜＞>"\' `]+?)'
    + r'\s*["\']?'
    + _LB + r"\s*/?\s*(?:" + _TAG_NAME_PATTERN + r")\s*" + _RB + r"`?",
    re.IGNORECASE,
)

MULTILINE_TAG_CLEANUP_RE = re.compile(
    r"(" + _LB + r"\s*(?:" + _TAG_NAME_PATTERN + r")\s*" + _RB + r")"
    r"([\s\S]*?)"
    r"(" + _LB + r"\s*/?\s*(?:" + _TAG_NAME_PATTERN + r")\s*" + _RB + r")",
    re.IGNORECASE,
)

CANONICAL_TAG_RE = re.compile(
    r"<(qqimg|qqvoice|qqvideo|qqfile|qqmedia)>"
    r"([^<>]+)"
    r"</(?:qqimg|qqvoice|qqvideo|qqfile|qqmedia|img)>",
    re.IGNORECASE,
)


def _resolve_tag_name(raw: str) -> str:
    lower = raw.lower().strip()
    if lower in VALID_TAGS:
        return lower
    return TAG_ALIASES.get(lower, "qqimg")


def _expand_tilde(p: str) -> str:
    if not p:
        return p
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or ""
    if not home:
        return p
    if p == "~":
        return home
    if p.startswith("~/") or p.startswith("~\\"):
        return home + "/" + p[2:]
    return p


def normalize_media_tags(text: str) -> str:
    def _normalize_wrapped(m: re.Match) -> str:
        tag = _resolve_tag_name(m.group(1))
        content = m.group(2).strip()
        if not content:
            return m.group(0)
        expanded = _expand_tilde(content)
        return f"<{tag}>{expanded}</{tag}>"

    cleaned = SELF_CLOSING_TAG_RE.sub(_normalize_wrapped, text)
    cleaned = MULTILINE_TAG_CLEANUP_RE.sub(
        lambda m: m.group(1) + re.sub(r"[\r\n\t]+", " ", m.group(2)).strip() + m.group(3),
        cleaned,
    )
    return FUZZY_MEDIA_TAG_RE.sub(_normalize_wrapped, cleaned)


@dataclass
class SendQueueItem:
    kind: str  # "text" | "image" | "voice" | "video" | "file"
    content: str


def parse_send_queue(text: str) -> list[SendQueueItem]:
    normalized = normalize_media_tags(text)
    queue: list[SendQueueItem] = []
    last_end = 0

    for m in CANONICAL_TAG_RE.finditer(normalized):
        before = normalized[last_end:m.start()].strip()
        if before:
            queue.append(SendQueueItem(kind="text", content=before))

        tag = m.group(1).lower()
        source = m.group(2).strip()
        if tag == "qqmedia":
            kind = detect_media_kind(source)
        else:
            kind = {"qqimg": "image", "qqvoice": "voice", "qqvideo": "video", "qqfile": "file"}[tag]
        queue.append(SendQueueItem(kind=kind, content=source))
        last_end = m.end()

    after = normalized[last_end:].strip()
    if after:
        queue.append(SendQueueItem(kind="text", content=after))

    if not queue and text.strip():
        queue.append(SendQueueItem(kind="text", content=text.strip()))

    return queue


# ── Upload cache ──────────────────────────────────────────────────────────────


@dataclass
class _CacheEntry:
    file_info: str
    file_uuid: str
    expires_at: float
    last_access: float = 0.0


class UploadCache:
    def __init__(self, max_size: int = 500):
        self._max_size = max_size
        self._store: dict[str, _CacheEntry] = {}

    @staticmethod
    def compute_hash(data: str | bytes) -> str:
        if isinstance(data, str):
            data = data.encode()
        return hashlib.md5(data).hexdigest()

    def _key(self, content_hash: str, scope: str, target_id: str, file_type: int) -> str:
        return f"{content_hash}:{scope}:{target_id}:{file_type}"

    def get(self, content_hash: str, scope: str, target_id: str, file_type: int) -> str | None:
        key = self._key(content_hash, scope, target_id, file_type)
        entry = self._store.get(key)
        if not entry:
            return None
        if time.time() >= entry.expires_at:
            del self._store[key]
            return None
        entry.last_access = time.time()
        return entry.file_info

    def set(
        self, content_hash: str, scope: str, target_id: str,
        file_type: int, file_info: str, file_uuid: str, ttl: int,
    ) -> None:
        self._evict_expired()
        if len(self._store) >= self._max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k].last_access)
            del self._store[oldest_key]
        key = self._key(content_hash, scope, target_id, file_type)
        self._store[key] = _CacheEntry(
            file_info=file_info, file_uuid=file_uuid,
            expires_at=time.time() + max(ttl - 60, 0),
            last_access=time.time(),
        )

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._store.items() if now >= v.expires_at]
        for k in expired:
            del self._store[k]


# ── File I/O helpers ──────────────────────────────────────────────────────────


def read_local_file_as_base64(
    file_path: str, max_size: int = _MAX_FILE_SIZE,
) -> tuple[str, str]:
    """Read a local file and return (base64_data, file_name).

    Raises FileNotFoundError or ValueError on problems.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    size = p.stat().st_size
    if size == 0:
        raise ValueError(f"File is empty: {file_path}")
    if size > max_size:
        raise ValueError(
            f"File too large: {size / 1024 / 1024:.1f}MB "
            f"(max {max_size / 1024 / 1024:.0f}MB)"
        )
    data = p.read_bytes()
    return base64.b64encode(data).decode("ascii"), p.name


def _media_upload_path(scope: str, target_id: str) -> str:
    if scope == "c2c":
        return f"/v2/users/{target_id}/files"
    return f"/v2/groups/{target_id}/files"


def _message_path(scope: str, target_id: str) -> str:
    if scope == "c2c":
        return f"/v2/users/{target_id}/messages"
    return f"/v2/groups/{target_id}/messages"


def _next_msg_seq() -> int:
    import random
    time_part = int(time.time() * 1000) % 100_000_000
    random_part = random.randint(0, 65535)
    return (time_part ^ random_part) % 65536


# ── QQ API interaction ────────────────────────────────────────────────────────


async def upload_media(
    session: aiohttp.ClientSession,
    api_base: str,
    headers: dict[str, str],
    scope: str,
    target_id: str,
    file_type: MediaFileType,
    *,
    url: str = "",
    file_data: str = "",
    file_name: str = "",
    cache: UploadCache | None = None,
) -> dict[str, Any]:
    """Upload media to QQ, return {"file_uuid", "file_info", "ttl"}."""
    if not url and not file_data:
        raise ValueError("upload_media: url or file_data is required")

    if file_data and cache:
        content_hash = cache.compute_hash(file_data)
        cached = cache.get(content_hash, scope, target_id, file_type)
        if cached:
            logger.debug("Upload cache hit for {}/{}", scope, target_id)
            return {"file_uuid": "", "file_info": cached, "ttl": 0}

    body: dict[str, Any] = {"file_type": int(file_type), "srv_send_msg": False}
    if url:
        body["url"] = url
    elif file_data:
        body["file_data"] = file_data
    if file_type == MediaFileType.FILE and file_name:
        body["file_name"] = file_name

    path = _media_upload_path(scope, target_id)
    last_error: Exception | None = None

    for attempt in range(_UPLOAD_MAX_RETRIES + 1):
        try:
            async with session.post(
                f"{api_base}{path}", json=body, headers=headers,
            ) as resp:
                if resp.status < 400:
                    result = await resp.json()
                    if file_data and cache and result.get("file_info") and result.get("ttl", 0) > 0:
                        cache.set(
                            content_hash, scope, target_id, file_type,
                            result["file_info"], result.get("file_uuid", ""),
                            result["ttl"],
                        )
                    return result
                resp_text = await resp.text()
                if resp.status in (400, 401):
                    raise RuntimeError(
                        f"Upload failed ({resp.status}): {resp_text[:200]}"
                    )
                last_error = RuntimeError(
                    f"Upload error ({resp.status}): {resp_text[:200]}"
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = e

        if attempt < _UPLOAD_MAX_RETRIES:
            delay = _UPLOAD_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "QQBot upload retry {}/{} in {:.1f}s: {}",
                attempt + 1, _UPLOAD_MAX_RETRIES, delay, last_error,
            )
            await asyncio.sleep(delay)

    raise last_error or RuntimeError("Upload failed after retries")


async def send_media_message(
    session: aiohttp.ClientSession,
    api_base: str,
    headers: dict[str, str],
    scope: str,
    target_id: str,
    file_info: str,
    msg_seq: int,
    *,
    msg_id: str = "",
    content: str = "",
) -> dict[str, Any]:
    """Send a msg_type=7 media message."""
    path = _message_path(scope, target_id)
    payload: dict[str, Any] = {
        "msg_type": 7,
        "media": {"file_info": file_info},
        "msg_seq": msg_seq,
    }
    if msg_id:
        payload["msg_id"] = msg_id
    if content:
        payload["content"] = content

    last_error: Exception | None = None
    for attempt in range(_SEND_RETRIES):
        try:
            async with session.post(
                f"{api_base}{path}", json=payload, headers=headers,
            ) as resp:
                if resp.status < 400:
                    return await resp.json()
                resp_text = await resp.text()
                if resp.status == 400 and "40034024" in resp_text and "msg_id" in payload:
                    logger.info("QQBot msg_id expired, retrying without reply reference")
                    payload.pop("msg_id", None)
                    continue
                if resp.status in (400, 401, 403, 404):
                    raise RuntimeError(
                        f"Send media failed ({resp.status}): {resp_text[:200]}"
                    )
                last_error = RuntimeError(
                    f"Send media error ({resp.status}): {resp_text[:200]}"
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = e

        if attempt < _SEND_RETRIES - 1:
            await asyncio.sleep(1 * (attempt + 1))

    raise last_error or RuntimeError("Send media message failed after retries")
