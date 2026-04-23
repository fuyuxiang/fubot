"""GatewayServer — HTTP/WebSocket server orchestrating all gateway subsystems.

Provides a unified API layer above the channel system for:
- External message ingestion (HTTP POST, WebSocket)
- Session lifecycle management with reset policies
- Authentication and rate limiting
- Cross-platform delivery routing
- Progressive message editing
- Health monitoring
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
from loguru import logger

from echo_agent.bus.events import InboundEvent, OutboundEvent, ContentBlock, ContentType
from echo_agent.bus.queue import MessageBus
from echo_agent.channels.manager import ChannelManager
from echo_agent.config.schema import GatewayConfig
from echo_agent.gateway.auth import GatewayAuth
from echo_agent.gateway.editor import ProgressiveEditor
from echo_agent.gateway.health import GatewayHealthProvider
from echo_agent.gateway.hooks import HookRegistry
from echo_agent.gateway.media import MediaCache
from echo_agent.gateway.rate_limiter import RateLimiter
from echo_agent.gateway.router import DeliveryRouter
from echo_agent.gateway.session_context import set_session_vars, clear_session_vars
from echo_agent.gateway.session_policy import SessionResetPolicy
from echo_agent.session.manager import SessionManager


class GatewayServer:

    def __init__(
        self,
        config: GatewayConfig,
        bus: MessageBus,
        channel_manager: ChannelManager,
        session_manager: SessionManager,
        workspace: Path,
    ):
        self._config = config
        self._bus = bus
        self.channel_manager = channel_manager
        self.session_manager = session_manager
        self._workspace = workspace

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._ws_clients: dict[str, web.WebSocketResponse] = {}
        self._pending_http: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._running = False

        data_dir = workspace / "data"
        self.auth = GatewayAuth(config.auth, data_dir)
        self.media_cache = MediaCache(
            cache_dir=workspace / config.media_cache_dir,
            max_size_mb=config.media_cache_max_mb,
        )
        self.rate_limiter = RateLimiter()
        self.delivery_router = DeliveryRouter(bus)
        self.hooks = HookRegistry()
        self.editor = ProgressiveEditor(bus)
        self.session_policy = SessionResetPolicy(config.session_policy)
        self.health = GatewayHealthProvider(self)
        self._bus.subscribe_outbound_global(self._handle_outbound)

        for name, plat_cfg in config.platforms.items():
            if plat_cfg.rate_limit_rpm:
                self.rate_limiter.configure(name, plat_cfg.rate_limit_rpm)

        if config.hooks_dir:
            hooks_path = workspace / config.hooks_dir
            if hooks_path.is_dir():
                self.hooks.load_from_dir(hooks_path)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._app = web.Application()
        self._setup_routes()

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner,
            self._config.host,
            self._config.port,
        )
        await self._site.start()
        self._running = True

        await self.hooks.emit("gateway_start")
        logger.info(
            "Gateway listening on {}:{}",
            self._config.host, self._config.port,
        )

    async def stop(self) -> None:
        self._running = False
        await self.hooks.emit("gateway_stop")

        for future in self._pending_http.values():
            if not future.done():
                future.cancel()
        self._pending_http.clear()

        for ws_id, ws in list(self._ws_clients.items()):
            await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=b"shutdown")
        self._ws_clients.clear()

        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

        await self.media_cache.cleanup()
        logger.info("Gateway stopped")

    # ── Route setup ──────────────────────────────────────────────────────────

    def _setup_routes(self) -> None:
        prefix = self._config.api_prefix
        app = self._app
        assert app is not None

        app.router.add_get("/", self._handle_playground)
        app.router.add_post(f"{prefix}/message", self._handle_message)
        app.router.add_get(f"{prefix}/health", self._handle_health)
        app.router.add_get(f"{prefix}/sessions", self._handle_list_sessions)
        app.router.add_delete(f"{prefix}/sessions/{{key}}", self._handle_reset_session)
        app.router.add_post(f"{prefix}/pair", self._handle_pair_generate)
        app.router.add_post(f"{prefix}/pair/verify", self._handle_pair_verify)
        app.router.add_get(f"{prefix}/stats", self._handle_stats)
        app.router.add_get(self._config.ws_path, self._handle_websocket)

    # ── HTTP handlers ────────────────────────────────────────────────────────

    PLACEHOLDER_CONTINUE = "<!-- more -->"

    def _playground_path(self) -> Path:
        return Path(__file__).resolve().parent / "static" / "index.html"

    def _build_outbound_payload(self, event: OutboundEvent) -> dict[str, Any]:
        return {
            "type": "message",
            "event_id": event.event_id,
            "reply_to_id": event.reply_to_id,
            "channel": event.channel,
            "chat_id": event.chat_id,
            "text": event.text,
            "is_final": event.is_final,
            "message_kind": event.message_kind,
            "edit_message_id": event.edit_message_id,
            "metadata": event.metadata,
        }

    async def _handle_playground(self, request: web.Request) -> web.Response:
        path = self._playground_path()
        if path.exists():
            return web.FileResponse(path)
        return web.Response(text="Gateway playground not found.", status=404)

    async def _handle_message(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        platform = body.get("platform", "api")
        user_id = body.get("user_id", "")
        chat_id = body.get("chat_id", user_id)
        text = body.get("text", "")
        media_urls = body.get("media_urls", [])
        wait = bool(body.get("wait", False))
        timeout_seconds = max(1, min(int(body.get("timeout_seconds", 180)), 600))

        if not text and not media_urls:
            return web.json_response({"error": "text or media_urls required"}, status=400)

        if not self.auth.is_authorized(platform, user_id):
            await self.hooks.emit("auth_failed", platform=platform, user_id=user_id)
            return web.json_response({"error": "unauthorized"}, status=403)

        if not self.rate_limiter.acquire(platform, chat_id):
            return web.json_response({"error": "rate limited"}, status=429)

        session_key = f"gateway:{platform}:{chat_id}"
        session = await self.session_manager.get_or_create(session_key)

        if self.session_policy.should_reset(session):
            await self.session_policy.reset(session, self.session_manager)
            await self.hooks.emit("session_reset", session_key=session_key)

        tokens = set_session_vars(
            platform=platform,
            chat_id=chat_id,
            user_id=user_id,
            session_key=session_key,
        )

        try:
            cached_media = []
            for url in media_urls:
                path = await self.media_cache.download(url, platform)
                if path:
                    cached_media.append({"type": "file", "url": str(path)})

            content_blocks = [ContentBlock(type=ContentType.TEXT, text=text)]
            for m in cached_media:
                content_blocks.append(ContentBlock(
                    type=ContentType.FILE,
                    url=m["url"],
                ))

            event = InboundEvent(
                channel=f"gateway:{platform}",
                sender_id=user_id,
                chat_id=chat_id,
                content=content_blocks,
                session_key_override=session_key,
                metadata={
                    "gateway": True,
                    "platform": platform,
                    "user_id": user_id,
                },
            )
            future: asyncio.Future[dict[str, Any]] | None = None
            if wait:
                future = asyncio.get_event_loop().create_future()
                self._pending_http[event.event_id] = future

            await self._bus.publish_inbound(event)
            await self.hooks.emit(
                "message_received",
                platform=platform, user_id=user_id, chat_id=chat_id,
            )

            if future:
                try:
                    payload = await asyncio.wait_for(future, timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    self._pending_http.pop(event.event_id, None)
                    return web.json_response(
                        {
                            "error": "timeout",
                            "event_id": event.event_id,
                            "session_key": session_key,
                        },
                        status=504,
                    )
                return web.json_response(
                    {
                        "status": "completed",
                        "event_id": event.event_id,
                        "session_key": session_key,
                        "reply": payload,
                    }
                )

            return web.json_response({
                "status": "accepted",
                "event_id": event.event_id,
                "session_key": session_key,
            })
        finally:
            clear_session_vars(tokens)

    async def _handle_health(self, request: web.Request) -> web.Response:
        status = await self.health.check()
        code = 200 if status["status"] != "unhealthy" else 503
        return web.json_response(status, status=code)

    async def _handle_list_sessions(self, request: web.Request) -> web.Response:
        sessions = self.session_manager.list_sessions()
        gateway_sessions = [s for s in sessions if s.get("key", "").startswith("gateway:")]
        return web.json_response({"sessions": gateway_sessions})

    async def _handle_reset_session(self, request: web.Request) -> web.Response:
        key = request.match_info["key"]
        session = await self.session_manager.get_or_create(key)
        await self.session_policy.reset(session, self.session_manager)
        await self.hooks.emit("session_reset", session_key=key)
        return web.json_response({"status": "reset", "session_key": key})

    async def _handle_pair_generate(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        platform = body.get("platform", "")
        if not platform:
            return web.json_response({"error": "platform required"}, status=400)

        code = self.auth.generate_pairing_code(platform)
        return web.json_response({"code": code, "ttl_seconds": self._config.auth.pairing_ttl_seconds})

    async def _handle_pair_verify(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        platform = body.get("platform", "")
        user_id = body.get("user_id", "")
        code = body.get("code", "")

        if not all([platform, user_id, code]):
            return web.json_response({"error": "platform, user_id, code required"}, status=400)

        if self.auth.verify_pairing(platform, user_id, code):
            await self.hooks.emit("auth_success", platform=platform, user_id=user_id)
            return web.json_response({"status": "paired"})
        else:
            await self.hooks.emit("auth_failed", platform=platform, user_id=user_id)
            return web.json_response({"error": "invalid or expired code"}, status=403)

    async def _handle_stats(self, request: web.Request) -> web.Response:
        health_data = await self.health.check()
        health_data["ws_clients"] = len(self._ws_clients)
        return web.json_response(health_data)

    # ── WebSocket handler ─────────────────────────────────────────────────

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        ws_id = None
        platform = "ws"
        user_id = ""
        chat_id = ""
        session_key = ""

        try:
            async for raw_msg in ws:
                if raw_msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(raw_msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"error": "invalid JSON"})
                        continue

                    msg_type = data.get("type", "message")

                    if msg_type == "auth":
                        platform = data.get("platform", "ws")
                        user_id = data.get("user_id", "")
                        chat_id = data.get("chat_id", user_id)

                        if not self.auth.is_authorized(platform, user_id):
                            await ws.send_json({"type": "error", "error": "unauthorized"})
                            await ws.close()
                            return ws

                        session_key = f"gateway:{platform}:{chat_id}"
                        ws_id = session_key
                        self._ws_clients[ws_id] = ws

                        session = await self.session_manager.get_or_create(session_key)
                        if self.session_policy.should_reset(session):
                            await self.session_policy.reset(session, self.session_manager)

                        await ws.send_json({"type": "auth_ok", "session_key": session_key})
                        await self.hooks.emit(
                            "auth_success", platform=platform, user_id=user_id,
                        )
                        continue

                    if msg_type == "message":
                        if not session_key:
                            await ws.send_json({"type": "error", "error": "authenticate first"})
                            continue

                        text = data.get("text", "")
                        if not text:
                            continue

                        if not self.rate_limiter.acquire(platform, chat_id):
                            await ws.send_json({"type": "error", "error": "rate limited"})
                            continue

                        tokens = set_session_vars(
                            platform=platform,
                            chat_id=chat_id,
                            user_id=user_id,
                            session_key=session_key,
                        )
                        try:
                            event = InboundEvent.text_message(
                                channel=f"gateway:{platform}",
                                sender_id=user_id,
                                chat_id=chat_id,
                                text=text,
                                session_key_override=session_key,
                            )
                            event.metadata["gateway"] = True
                            event.metadata["platform"] = platform
                            await self._bus.publish_inbound(event)
                            await ws.send_json({
                                "type": "accepted",
                                "event_id": event.event_id,
                            })
                        finally:
                            clear_session_vars(tokens)

                    if msg_type == "ping":
                        await ws.send_json({"type": "pong"})

                elif raw_msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                    break

        except Exception as e:
            logger.error("WebSocket error: {}", e)
        finally:
            if ws_id and ws_id in self._ws_clients:
                del self._ws_clients[ws_id]

        return ws

    async def _handle_outbound(self, event: OutboundEvent) -> None:
        if event.metadata.get("_drop"):
            return
        if not event.channel.startswith("gateway:"):
            return

        _, platform = event.channel.split(":", 1)
        session_key = f"gateway:{platform}:{event.chat_id}"
        payload = self._build_outbound_payload(event)

        if event.reply_to_id:
            future = self._pending_http.get(event.reply_to_id)
            if future is not None and not future.done() and event.is_final:
                future.set_result(payload)
                self._pending_http.pop(event.reply_to_id, None)

        await self.broadcast_to_ws(session_key, payload)

    async def broadcast_to_ws(self, session_key: str, data: dict[str, Any]) -> bool:
        ws = self._ws_clients.get(session_key)
        if ws is None or ws.closed:
            return False
        try:
            await ws.send_json(data)
            return True
        except Exception as e:
            logger.warning("Failed to send WebSocket message: {}", e)
            return False
