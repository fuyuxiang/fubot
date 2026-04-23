"""Status command — display current configuration summary."""

from __future__ import annotations

from pathlib import Path

from echo_agent.cli.colors import Colors, color, print_header, print_info
from echo_agent.config.loader import load_config, resolve_config_file
from echo_agent.config.schema import ProviderConfig


def _provider_credential_status(provider: ProviderConfig) -> tuple[str, str]:
    name = provider.name.lower()
    if provider.credential_pool:
        return "credential pool configured", Colors.GREEN
    if provider.api_key:
        return "API key configured", Colors.GREEN
    if name in ("bedrock", "aws"):
        return "uses AWS environment/profile", Colors.CYAN
    return "API key missing", Colors.YELLOW


def _effective_workspace(raw_workspace: str, config_file: Path | None, workspace_override: str | None = None) -> Path:
    workspace = Path(workspace_override or raw_workspace).expanduser()
    if workspace.is_absolute():
        return workspace.resolve()
    base = Path.cwd() if workspace_override or not config_file else config_file.parent
    return (base / workspace).resolve()


def show_status(config_path: str | Path | None = None, workspace: str | Path | None = None) -> None:
    config_file = resolve_config_file(config_path=config_path, search_dir=workspace)
    print_header("Echo Agent Status")

    if config_file and config_file.exists():
        print(f"  Config file:  {color(str(config_file), Colors.CYAN)}")
    elif config_file:
        print(f"  Config file:  {color(str(config_file) + ' (not found)', Colors.YELLOW)}")
    else:
        print(f"  Config file:  {color('not found', Colors.YELLOW)}")

    overrides = {"workspace": str(workspace)} if workspace else None
    config = load_config(config_path=config_file, overrides=overrides)
    effective_workspace = _effective_workspace(config.workspace, config_file if config_file and config_file.exists() else None, str(workspace) if workspace else None)

    print(f"  Workspace:    {color(str(effective_workspace), Colors.CYAN)}")
    print()

    # Providers
    print_header("LLM Providers")
    if config.models.providers:
        for p in config.models.providers:
            models = ", ".join(p.models[:3]) if p.models else "—"
            credential_text, credential_color = _provider_credential_status(p)
            print(
                f"  {color(p.name or '<unnamed>', Colors.GREEN)}"
                f"  models: {models}"
                f"  {color(credential_text, credential_color)}"
            )
    else:
        print_info("No providers configured")
    print(f"  Default model: {color(config.models.default_model, Colors.CYAN)}")
    print()

    # Channels
    print_header("Channels")
    channel_names = [
        "cli", "webhook", "cron", "telegram", "discord", "slack",
        "whatsapp", "wechat", "weixin", "qqbot", "feishu", "dingtalk",
        "email", "wecom", "matrix",
    ]
    any_enabled = False
    for name in channel_names:
        ch_cfg = getattr(config.channels, name, None)
        if ch_cfg is None:
            continue
        enabled = getattr(ch_cfg, "enabled", False)
        if enabled:
            print(f"  {color('●', Colors.GREEN)} {name}")
            any_enabled = True
        else:
            print(f"  {color('○', Colors.DIM)} {name}")
    if not any_enabled:
        print_info("Only CLI channel is active by default")
    print()

    # Gateway
    print_header("Gateway")
    if config.gateway.enabled:
        print(f"  {color('●', Colors.GREEN)} Enabled on {config.gateway.host}:{config.gateway.port}")
    else:
        print(f"  {color('○', Colors.DIM)} Disabled")
    print()
