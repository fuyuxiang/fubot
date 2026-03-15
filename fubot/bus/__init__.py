"""Message bus module for decoupled channel-agent communication."""

from fubot.bus.events import InboundMessage, OutboundMessage
from fubot.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
