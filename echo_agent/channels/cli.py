"""CLI channel — interactive terminal input/output."""

from __future__ import annotations

import asyncio
import sys
import threading
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
        self._reader_stop: threading.Event | None = None
        self._reader_thread: threading.Thread | None = None

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
        if self._reader_stop:
            self._reader_stop.set()
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
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        if self._try_add_stdin_reader(loop, queue):
            self._print_prompt()
            try:
                await self._consume_lines(queue, prompt_after_read=True)
            finally:
                try:
                    loop.remove_reader(sys.stdin.fileno())
                except (AttributeError, NotImplementedError, RuntimeError, ValueError):
                    pass
            return

        await self._read_loop_threaded(loop)

    def _try_add_stdin_reader(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str | None]) -> bool:
        try:
            fd = sys.stdin.fileno()

            def on_stdin_ready() -> None:
                try:
                    line = sys.stdin.readline()
                except (EOFError, KeyboardInterrupt):
                    line = ""
                queue.put_nowait(line if line != "" else None)

            loop.add_reader(fd, on_stdin_ready)
            return True
        except (AttributeError, NotImplementedError, RuntimeError, ValueError):
            return False

    async def _read_loop_threaded(self, loop: asyncio.AbstractEventLoop) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stop_event = threading.Event()
        self._reader_stop = stop_event

        def read_stdin() -> None:
            while not stop_event.is_set():
                try:
                    line = input("You> ")
                except (EOFError, KeyboardInterrupt):
                    line = None
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, line)
                except RuntimeError:
                    break
                if line is None:
                    break

        self._reader_thread = threading.Thread(target=read_stdin, name="echo-agent-cli-input", daemon=True)
        self._reader_thread.start()
        try:
            await self._consume_lines(queue, prompt_after_read=False)
        finally:
            stop_event.set()

    async def _consume_lines(self, queue: asyncio.Queue[str | None], *, prompt_after_read: bool) -> None:
        while self._running:
            try:
                line = await queue.get()
                if line is None:
                    self._running = False
                    self._request_exit()
                    break
                line = line.strip()
                if not line:
                    if prompt_after_read and self._running:
                        self._print_prompt()
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
                if prompt_after_read and self._running:
                    self._print_prompt()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("CLI read error: {}", e)

    def _request_exit(self) -> None:
        if self._on_exit:
            self._on_exit()

    @staticmethod
    def _print_prompt() -> None:
        print("You> ", end="", flush=True)
