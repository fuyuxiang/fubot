"""Echo Agent entry point — bootstraps all subsystems and runs the agent."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")


@dataclass
class _BootstrapResult:
    config: Any = None
    workspace: Path = field(default_factory=lambda: Path("."))
    storage: Any = None
    bus: Any = None
    router: Any = None
    provider: Any = None
    agent: Any = None
    channels: Any = None
    scheduler: Any = None
    health: Any = None


async def _bootstrap(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    on_cli_exit: Callable[[], None] | None = None,
) -> _BootstrapResult:
    """Shared bootstrap: config → storage → providers → bus → agent → channels."""
    from echo_agent.agent.loop import AgentLoop
    from echo_agent.bus.queue import MessageBus
    from echo_agent.channels.manager import ChannelManager
    from echo_agent.config.loader import load_config, resolve_config_file
    from echo_agent.models.provider import LLMProvider, LLMResponse
    from echo_agent.models.providers import create_provider
    from echo_agent.models.router import ModelRouter
    from echo_agent.observability.monitor import HealthChecker
    from echo_agent.storage.sqlite import SQLiteBackend

    config_file = resolve_config_file(config_path)
    config = load_config(config_path=config_file, overrides=overrides)
    _configure_logging(config.observability.log_level)

    workspace_value = Path(config.workspace).expanduser()
    if not workspace_value.is_absolute():
        workspace_base = Path.cwd() if overrides and "workspace" in overrides else (config_file.parent if config_file else Path.cwd())
        workspace_value = workspace_base / workspace_value
    ws = workspace_value.resolve()
    ws.mkdir(parents=True, exist_ok=True)

    storage = SQLiteBackend(ws / config.storage.database_path)
    await storage.initialize()

    bus = MessageBus()
    router = ModelRouter(config.models)
    provider: LLMProvider | None = None
    provider_errors: list[str] = []

    for pc in config.models.providers:
        try:
            p = create_provider(pc, default_model=config.models.default_model)
            router.register_provider(pc.name, p)
            if provider is None:
                provider = p
            logger.info("Registered provider: {}", pc.name)
        except Exception as e:
            provider_errors.append(f"{pc.name or '<unnamed>'}: {e}")
            logger.warning("Failed to create provider '{}': {}", pc.name, e)

    if provider is None:
        if config.models.providers:
            details = "; ".join(provider_errors) or "all configured providers were skipped"
            stub_message = (
                "[No LLM provider could be initialized. Check provider SDK/API key. "
                f"Details: {details}]"
            )
            logger.warning("No providers initialized — using stub: {}", details)
        else:
            stub_message = "[No LLM provider configured. Set up a provider in echo-agent.yaml]"
            logger.warning("No providers configured — using stub")

        class _StubProvider(LLMProvider):
            async def chat(self, messages, tools=None, model=None, tool_choice=None, **kw):
                return LLMResponse(content=stub_message)
            def get_default_model(self):
                return "stub"
        provider = _StubProvider()
        router.register_provider("stub", provider)

    from echo_agent.scheduler.service import Scheduler
    scheduler: Scheduler | None = None
    if config.scheduler.enabled:
        scheduler = Scheduler(
            store_path=ws / "data" / "scheduler.json",
            max_concurrent=config.scheduler.max_concurrent_jobs,
        )

    from echo_agent.tasks.manager import TaskManager
    from echo_agent.tasks.workflow import WorkflowEngine
    task_manager = TaskManager(storage)
    workflow_engine = WorkflowEngine(storage, task_manager)

    agent = AgentLoop(
        bus=bus, config=config, provider=provider, workspace=ws,
        router=router,
        scheduler=scheduler, storage=storage,
        task_manager=task_manager, workflow_engine=workflow_engine,
    )
    channels = ChannelManager(config.channels, bus, on_cli_exit=on_cli_exit)
    health = HealthChecker(check_interval=config.observability.health_check_interval_seconds)

    from echo_agent.observability.monitor import ComponentHealth as CH

    async def _check_bus() -> CH:
        return CH.HEALTHY if bus.pending_inbound < 900 else CH.DEGRADED

    async def _check_agent() -> CH:
        return CH.HEALTHY if agent._running else CH.UNHEALTHY

    async def _check_storage() -> CH:
        return CH.HEALTHY if storage._db else CH.UNHEALTHY

    health.register_check("bus", _check_bus)
    health.register_check("agent", _check_agent)
    health.register_check("storage", _check_storage)

    async def _session_cleanup() -> CH:
        count = await agent.sessions.cleanup_expired()
        if count:
            logger.info("Cleaned up {} expired sessions", count)
        return CH.HEALTHY

    health.register_check("session_cleanup", _session_cleanup)

    return _BootstrapResult(
        config=config, workspace=ws, storage=storage, bus=bus,
        router=router, provider=provider, agent=agent,
        channels=channels, scheduler=scheduler, health=health,
    )


def _install_signal_handler(shutdown: asyncio.Event) -> None:
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:
            pass


async def _run(config_path: str | None = None, workspace: str | None = None) -> None:
    if config_path is None and workspace:
        from echo_agent.config.loader import resolve_config_file
        config_path = str(resolve_config_file(search_dir=workspace) or "")
    overrides = {"workspace": workspace} if workspace else None
    shutdown = asyncio.Event()
    ctx = await _bootstrap(config_path=config_path, overrides=overrides, on_cli_exit=shutdown.set)

    logger.info("Echo Agent starting — workspace: {}", ctx.workspace)

    _install_signal_handler(shutdown)

    await ctx.bus.start()
    await ctx.agent.start()
    await ctx.channels.start_all()
    if not ctx.channels.active_channels and not ctx.config.gateway.enabled:
        logger.error(
            "No active input channels. Run in an interactive terminal, enable gateway, "
            "or configure another channel."
        )
        await ctx.channels.stop_all()
        await ctx.agent.stop()
        await ctx.bus.stop()
        await ctx.storage.close()
        return
    if ctx.scheduler:
        await ctx.scheduler.start()
    await ctx.health.start()

    gateway = None
    if ctx.config.gateway.enabled:
        from echo_agent.gateway.server import GatewayServer
        gateway = GatewayServer(
            config=ctx.config.gateway,
            bus=ctx.bus,
            channel_manager=ctx.channels,
            session_manager=ctx.agent.sessions,
            workspace=ctx.workspace,
            agent_loop=ctx.agent,
            a2a_config=ctx.config.a2a,
        )
        await gateway.start()
        logger.info("Gateway started on {}:{}", ctx.config.gateway.host, ctx.config.gateway.port)

    logger.info("Echo Agent ready — channels: {}", ctx.channels.active_channels)
    await shutdown.wait()

    logger.info("Shutting down...")
    if gateway:
        await gateway.stop()
    await ctx.health.stop()
    if ctx.scheduler:
        await ctx.scheduler.stop()
    await ctx.channels.stop_all()
    await ctx.agent.stop()
    await ctx.bus.stop()
    await ctx.storage.close()
    logger.info("Echo Agent stopped")


async def _run_gateway(config_path: str | None = None, host: str | None = None, port: int | None = None, workspace: str | None = None) -> None:
    if config_path is None and workspace:
        from echo_agent.config.loader import resolve_config_file
        config_path = str(resolve_config_file(search_dir=workspace) or "")
    overrides: dict[str, Any] = {"workspace": workspace} if workspace else {}
    shutdown = asyncio.Event()
    ctx = await _bootstrap(config_path=config_path, overrides=overrides or None, on_cli_exit=shutdown.set)
    ctx.config.gateway.enabled = True
    if host:
        ctx.config.gateway.host = host
    if port:
        ctx.config.gateway.port = port

    _install_signal_handler(shutdown)

    await ctx.bus.start()
    await ctx.agent.start()
    await ctx.channels.start_all()

    from echo_agent.gateway.server import GatewayServer
    gateway = GatewayServer(
        config=ctx.config.gateway,
        bus=ctx.bus,
        channel_manager=ctx.channels,
        session_manager=ctx.agent.sessions,
        workspace=ctx.workspace,
        agent_loop=ctx.agent,
        a2a_config=ctx.config.a2a,
    )
    await gateway.start()
    logger.info("Gateway listening on {}:{}", ctx.config.gateway.host, ctx.config.gateway.port)

    await shutdown.wait()

    await gateway.stop()
    await ctx.channels.stop_all()
    await ctx.agent.stop()
    await ctx.bus.stop()
    await ctx.storage.close()


def _run_eval(args) -> None:
    from pathlib import Path as _Path

    config_path = args.config or getattr(args, "top_config", None)
    workspace = args.workspace or getattr(args, "top_workspace", None)

    from echo_agent.config.loader import load_config, resolve_config_file
    config_file = resolve_config_file(config_path)
    config = load_config(config_path=config_file)

    dataset_path = args.dataset or config.evaluation.dataset_path
    path = _Path(dataset_path)
    if not path.exists():
        print(f"Dataset not found: {path}")
        print("Create a YAML file with test cases. Example:")
        print("  - id: test_001")
        print("    input: 'Hello'")
        print("    expected_contains: ['hello', 'hi']")
        return

    from echo_agent.evaluation import EvalRunner, EvalDataset

    dataset = EvalDataset.from_path(path)
    if args.tag:
        cases = dataset.filter_by_tag(args.tag)
        dataset = EvalDataset(cases)

    if not dataset.cases:
        print("No test cases found.")
        return

    async def run():
        overrides = {"workspace": workspace} if workspace else None
        ctx = await _bootstrap(config_path=config_path, overrides=overrides)
        await ctx.bus.start()
        await ctx.agent.start()

        try:
            runner = EvalRunner(ctx.agent, parallel=args.parallel, timeout=config.evaluation.timeout_per_case)
            report = await runner.run_dataset(dataset)

            from echo_agent.evaluation.reporter import EvalReporter
            reporter = EvalReporter()
            print(reporter.to_table(report))

            if args.output:
                _Path(args.output).write_text(reporter.to_json(report), encoding="utf-8")
                print(f"\nResults saved to {args.output}")
        finally:
            await ctx.agent.stop()
            await ctx.bus.stop()
            await ctx.storage.close()

    asyncio.run(run())


def main() -> None:
    parser = argparse.ArgumentParser(prog="echo-agent", description="Echo Agent — modular AI agent framework")
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_parser = subparsers.add_parser("run", help="Start the agent")
    run_parser.add_argument("-c", "--config", help="Path to config file")
    run_parser.add_argument("-w", "--workspace", help="Workspace directory")

    # setup
    setup_parser = subparsers.add_parser("setup", help="Run the setup wizard")
    setup_parser.add_argument("section", nargs="?", default=None, help="Setup section: model, channel, advanced")
    setup_parser.add_argument("-c", "--config", help="Path to config file")
    setup_parser.add_argument("-w", "--workspace", help="Workspace directory")

    # status
    status_parser = subparsers.add_parser("status", help="Show current configuration status")
    status_parser.add_argument("-c", "--config", help="Path to config file")
    status_parser.add_argument("-w", "--workspace", help="Workspace directory")

    # gateway
    gw_parser = subparsers.add_parser("gateway", help="Start the gateway server")
    gw_parser.add_argument("-c", "--config", help="Path to config file")
    gw_parser.add_argument("-w", "--workspace", help="Workspace directory")
    gw_parser.add_argument("--host", help="Gateway host")
    gw_parser.add_argument("--port", type=int, help="Gateway port")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Run evaluation test suite")
    eval_parser.add_argument("--dataset", "-d", default="", help="Path to eval dataset (YAML/JSON)")
    eval_parser.add_argument("--tag", "-t", default="", help="Filter cases by tag")
    eval_parser.add_argument("--parallel", "-p", type=int, default=3, help="Parallel cases")
    eval_parser.add_argument("--output", "-o", default="", help="Output file for results")
    eval_parser.add_argument("-c", "--config", help="Path to config file")
    eval_parser.add_argument("-w", "--workspace", help="Workspace directory")

    # service
    svc_parser = subparsers.add_parser("service", help="Manage systemd service (Linux)")
    svc_parser.add_argument("action", choices=["install", "uninstall", "start", "stop", "restart", "status", "logs"], help="Service action")
    svc_parser.add_argument("-w", "--workspace", help="Workspace directory (used by install)")

    # top-level flags for backward compat
    parser.add_argument("-c", "--config", help="Path to config file", dest="top_config")
    parser.add_argument("-w", "--workspace", help="Workspace directory", dest="top_workspace")

    args = parser.parse_args()

    if args.command == "setup":
        from echo_agent.cli.setup import run_setup_wizard
        run_setup_wizard(section=args.section, config_path=args.config or args.top_config, workspace=args.workspace or args.top_workspace)
        return

    if args.command == "status":
        from echo_agent.cli.status import show_status
        show_status(config_path=args.config or args.top_config, workspace=args.workspace or args.top_workspace)
        return

    if args.command == "gateway":
        try:
            asyncio.run(_run_gateway(config_path=args.config or args.top_config, host=args.host, port=args.port, workspace=args.workspace or args.top_workspace))
        except KeyboardInterrupt:
            pass
        return

    if args.command == "eval":
        _run_eval(args)
        return

    if args.command == "service":
        from echo_agent.cli.service import run_action
        run_action(args.action, workspace=args.workspace or args.top_workspace)
        return

    # "run" command or no command (backward compat)
    config_path = getattr(args, "config", None) or args.top_config
    workspace = getattr(args, "workspace", None) or args.top_workspace

    from echo_agent.cli.setup import prompt_first_run_setup
    if prompt_first_run_setup(config_path=config_path, workspace=workspace):
        return

    try:
        asyncio.run(_run(config_path=config_path, workspace=workspace))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
