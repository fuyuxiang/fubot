"""CLI channel — interactive terminal input/output."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable

from loguru import logger

from echo_agent.bus.events import OutboundEvent
from echo_agent.bus.queue import MessageBus
from echo_agent.channels.base import BaseChannel
from echo_agent.config.schema import CLIChannelConfig


class CLIChannel(BaseChannel):
    name = "cli"

    def __init__(self, config: CLIChannelConfig, bus: MessageBus, on_exit: Callable[[], None] | None = None):
        super().__init__(config, bus)
        self._task: asyncio.Task | None = None
        self._on_exit = on_exit

    async def start(self) -> None:
        if not sys.stdin.isatty():
            self._running = False
            logger.warning("CLI channel disabled because stdin is not interactive")
            return
        self._running = True
        self.bus.subscribe_outbound(self.name, self.send)
        self._task = asyncio.create_task(self._read_loop())
        logger.info("CLI channel started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def send(self, event: OutboundEvent) -> None:
        text = event.text
        if text:
            print(f"\n{text}\n")

    async def _read_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                line = await loop.run_in_executor(None, self._read_line)
                if line is None:
                    self._running = False
                    self._request_exit()
                    break
                line = line.strip()
                if not line:
                    continue
                if line.lower() in ("exit", "quit", "/quit"):
                    self._running = False
                    self._request_exit()
                    break
                await self._handle_message(
                    sender_id="cli_user",
                    chat_id="cli",
                    text=line,
                    session_key="cli:cli",
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("CLI read error: {}", e)

    def _request_exit(self) -> None:
        if self._on_exit:
            self._on_exit()

    @staticmethod
    def _read_line() -> str | None:
        try:
            return input("You> ")
        except (EOFError, KeyboardInterrupt):
            return None
