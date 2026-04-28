"""Unified event types for the message bus.

All channel sources normalize into these types before entering the agent loop.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar


class EventType(str, Enum):
    MESSAGE = "message"
    WEBHOOK = "webhook"
    CRON = "cron"
    CLI = "cli"
    SYSTEM = "system"


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    VOICE = "voice"
    MIXED = "mixed"


class ProcessingOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class PollRequest:
    question: str
    options: list[str]
    allow_multiple: bool = False
    duration_seconds: int | None = None


@dataclass
class ContentBlock:
    """A single piece of content within a message."""
    type: ContentType = ContentType.TEXT
    text: str = ""
    url: str = ""
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InboundEvent:
    """Unified inbound event from any channel source."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    event_type: EventType = EventType.MESSAGE
    channel: str = ""
    sender_id: str = ""
    chat_id: str = ""
    content: list[ContentBlock] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to_id: str | None = None
    thread_id: str | None = None
    session_key_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    gateway_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> str:
        if self.session_key_override:
            return self.session_key_override
        return f"{self.channel}:{self.chat_id}"

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if b.text)

    @property
    def media_urls(self) -> list[str]:
        return [b.url for b in self.content if b.url and b.type != ContentType.TEXT]

    @classmethod
    def text_message(
        cls,
        channel: str,
        sender_id: str,
        chat_id: str,
        text: str,
        **kwargs: Any,
    ) -> InboundEvent:
        return cls(
            channel=channel,
            sender_id=sender_id,
            chat_id=chat_id,
            content=[ContentBlock(type=ContentType.TEXT, text=text)],
            **kwargs,
        )


@dataclass
class OutboundEvent:
    """Unified outbound event to any channel."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    channel: str = ""
    chat_id: str = ""
    content: list[ContentBlock] = field(default_factory=list)
    reply_to_id: str | None = None
    edit_message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_final: bool = True
    task_id: str | None = None
    workflow_id: str | None = None
    message_kind: str = "final"

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if b.text)

    @classmethod
    def text_reply(
        cls,
        channel: str,
        chat_id: str,
        text: str,
        reply_to_id: str | None = None,
        **kwargs: Any,
    ) -> OutboundEvent:
        return cls(
            channel=channel,
            chat_id=chat_id,
            content=[ContentBlock(type=ContentType.TEXT, text=text)],
            reply_to_id=reply_to_id,
            **kwargs,
        )

    _MEDIA_TAG_RE: ClassVar[re.Pattern] = re.compile(
        r"<(qqimg|qqvoice|qqvideo|qqfile|qqmedia)>"
        r"([^<>]+)"
        r"</(?:qqimg|qqvoice|qqvideo|qqfile|qqmedia|img)>",
        re.IGNORECASE,
    )
    _TAG_TO_CONTENT_TYPE: ClassVar[dict[str, ContentType]] = {
        "qqimg": ContentType.IMAGE,
        "qqvoice": ContentType.AUDIO,
        "qqvideo": ContentType.VIDEO,
        "qqfile": ContentType.FILE,
        "qqmedia": ContentType.FILE,
    }

    @classmethod
    def from_text_with_media(
        cls,
        channel: str,
        chat_id: str,
        text: str,
        reply_to_id: str | None = None,
        **kwargs: Any,
    ) -> OutboundEvent:
        """Parse media tags in text and create structured content blocks."""
        blocks: list[ContentBlock] = []
        last_end = 0
        for m in cls._MEDIA_TAG_RE.finditer(text):
            before = text[last_end:m.start()].strip()
            if before:
                blocks.append(ContentBlock(type=ContentType.TEXT, text=before))
            tag = m.group(1).lower()
            source = m.group(2).strip()
            ct = cls._TAG_TO_CONTENT_TYPE.get(tag, ContentType.FILE)
            blocks.append(ContentBlock(type=ct, url=source))
            last_end = m.end()
        after = text[last_end:].strip()
        if after:
            blocks.append(ContentBlock(type=ContentType.TEXT, text=after))
        if not blocks:
            blocks.append(ContentBlock(type=ContentType.TEXT, text=text))
        return cls(
            channel=channel,
            chat_id=chat_id,
            content=blocks,
            reply_to_id=reply_to_id,
            **kwargs,
        )
