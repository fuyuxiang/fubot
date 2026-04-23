"""Interactive setup wizard for Echo Agent."""

from __future__ import annotations

import sys
from pathlib import Path

from echo_agent.cli.colors import (
    Colors,
    color,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from echo_agent.cli.prompt import (
    is_interactive,
    prompt,
    prompt_checklist,
    prompt_choice,
    prompt_yes_no,
)
from echo_agent.config.loader import save_config, _find_config_file


# ── Provider presets ────────────────────────────────────────────────────────

PROVIDERS = [
    ("openai", "OpenAI", [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o3-mini",
    ]),
    ("anthropic", "Anthropic", [
        "claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-5-20251001",
    ]),
    ("gemini", "Google Gemini", [
        "gemini-2.5-pro", "gemini-2.5-flash",
    ]),
    ("openrouter", "OpenRouter", [
        "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514", "google/gemini-2.5-pro",
    ]),
    ("bedrock", "AWS Bedrock", [
        "anthropic.claude-sonnet-4-20250514-v1:0", "anthropic.claude-haiku-4-5-20251001-v1:0",
    ]),
    ("custom", "Custom (OpenAI-compatible)", []),
]

CHANNEL_DEFS: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("telegram", "Telegram", [("token", "Bot token")]),
    ("discord", "Discord", [("token", "Bot token")]),
    ("slack", "Slack", [("bot_token", "Bot token"), ("app_token", "App token")]),
    ("dingtalk", "DingTalk", [("app_key", "App key"), ("app_secret", "App secret"), ("robot_code", "Robot code")]),
    ("feishu", "Feishu / Lark", [("app_id", "App ID"), ("app_secret", "App secret")]),
    ("wecom", "WeCom", [("corp_id", "Corp ID"), ("agent_id", "Agent ID"), ("secret", "Secret")]),
    ("wechat", "WeChat Official Account", [("app_id", "App ID"), ("app_secret", "App secret")]),
    ("weixin", "WeChat Personal (iLink)", []),
    ("qqbot", "QQ Bot", [("app_id", "App ID"), ("app_secret", "App secret")]),
    ("email", "Email", [("imap_host", "IMAP host"), ("smtp_host", "SMTP host"), ("username", "Username"), ("password", "Password")]),
    ("matrix", "Matrix", [("homeserver", "Homeserver URL"), ("user_id", "User ID"), ("access_token", "Access token")]),
]


# ── Setup sections ──────────────────────────────────────────────────────────

def setup_model(config: dict) -> None:
    print_header("LLM Provider")
    print_info("Choose your inference provider and model.")
    print()

    existing_providers = config.get("models", {}).get("providers", [])
    existing_provider = existing_providers[0] if existing_providers else {}
    existing_name = existing_provider.get("name", "")
    existing_key = existing_provider.get("apiKey", "")
    existing_base = existing_provider.get("apiBase", "")
    existing_model = config.get("models", {}).get("defaultModel", "")

    provider_default = 0
    for i, (key, _, _) in enumerate(PROVIDERS):
        if key == existing_name or (existing_name == "openai" and key == "custom" and existing_base):
            provider_default = i
            break

    provider_names = [p[1] for p in PROVIDERS]
    idx = prompt_choice("Select provider:", provider_names, default=provider_default)
    provider_key, provider_label, preset_models = PROVIDERS[idx]

    api_key = ""
    api_base = ""

    if provider_key == "bedrock":
        print_info("Bedrock uses AWS credentials from environment (AWS_ACCESS_KEY_ID, etc.)")
    elif provider_key == "custom":
        api_base = prompt("  API base URL", default=existing_base)
        if existing_key:
            api_key = prompt("  API key [****saved, Enter to keep]", password=True)
            if not api_key:
                api_key = existing_key
        else:
            api_key = prompt("  API key", password=True)
    else:
        if existing_key and existing_name == provider_key:
            api_key = prompt(f"  {provider_label} API key [****saved, Enter to keep]", password=True)
            if not api_key:
                api_key = existing_key
        else:
            api_key = prompt(f"  {provider_label} API key", password=True)
            if not api_key:
                print_warning("No API key provided. You can set it later in echo-agent.yaml")

    # Model selection
    if preset_models:
        model_default = 0
        if existing_model in preset_models:
            model_default = preset_models.index(existing_model)
        model_choices = preset_models + ["Enter custom model name"]
        model_idx = prompt_choice("Select default model:", model_choices, default=model_default)
        if model_idx == len(preset_models):
            default_model = prompt("  Model name", default=existing_model)
        else:
            default_model = preset_models[model_idx]
    else:
        default_model = prompt("  Default model name", default=existing_model or "gpt-4o")

    actual_name = provider_key if provider_key != "custom" else "openai"

    provider_entry: dict = {"name": actual_name}
    if api_key:
        provider_entry["apiKey"] = api_key
    if api_base:
        provider_entry["apiBase"] = api_base
    if preset_models:
        provider_entry["models"] = preset_models

    config["models"] = {
        "defaultModel": default_model,
        "providers": [provider_entry],
    }

    print_success(f"Provider: {provider_label}, Model: {default_model}")


def _setup_weixin_qr(ch: dict) -> None:
    """Run iLink QR code login flow for WeChat Personal."""
    import asyncio

    print_info("Starting WeChat QR code login...")
    print_info("A QR code URL will appear below. Open it in a browser and scan with WeChat.")
    print()

    from echo_agent.channels.weixin import WeixinChannel

    result = asyncio.run(WeixinChannel.qr_login())
    if result:
        ch["account_id"] = result["account_id"]
        ch["token"] = result["token"]
        if result.get("base_url"):
            ch["base_url"] = result["base_url"]
        print_success("WeChat login successful!")
    else:
        print_error("WeChat QR login failed or timed out.")
        print_info("You can retry later with 'echo-agent setup channel'.")


def setup_channels(config: dict) -> None:
    print_header("Messaging Channels")
    print_info("Select channels to configure. CLI is always enabled.")
    print()

    existing_channels = config.get("channels", {})
    pre_selected = []
    for i, (ch_key, _, _) in enumerate(CHANNEL_DEFS):
        ch_cfg = existing_channels.get(ch_key, {})
        if isinstance(ch_cfg, dict) and ch_cfg.get("enabled"):
            pre_selected.append(i)

    channel_names = [c[1] for c in CHANNEL_DEFS]
    selected = prompt_checklist("Enable channels:", channel_names, pre_selected=pre_selected or None)

    if not selected:
        print_info("No extra channels selected. CLI will be used.")
        return

    channels_cfg = config.setdefault("channels", {})

    for idx in selected:
        ch_key, ch_label, fields = CHANNEL_DEFS[idx]
        print_header(f"{ch_label} Configuration")
        ch = channels_cfg.setdefault(ch_key, {})
        ch["enabled"] = True

        if ch_key == "weixin":
            _setup_weixin_qr(ch)
            continue

        for field_key, field_label in fields:
            is_secret = "key" in field_key.lower() or "secret" in field_key.lower() or "token" in field_key.lower() or "password" in field_key.lower()
            value = prompt(f"  {field_label}", default=ch.get(field_key, ""), password=is_secret)
            if value:
                ch[field_key] = value

    print_success(f"Configured {len(selected)} channel(s)")


def setup_advanced(config: dict) -> None:
    print_header("Advanced Settings")

    ws = prompt("  Workspace directory", default=config.get("workspace", "."))
    if ws and ws != ".":
        config["workspace"] = ws

    log_choices = ["INFO", "DEBUG", "WARNING", "ERROR"]
    existing_log = config.get("observability", {}).get("logLevel", "INFO")
    log_default = log_choices.index(existing_log) if existing_log in log_choices else 0
    log_idx = prompt_choice("Log level:", log_choices, default=log_default)
    config.setdefault("observability", {})["logLevel"] = log_choices[log_idx]

    existing_gw = config.get("gateway", {})
    if prompt_yes_no("Enable gateway (Web API)?", default=existing_gw.get("enabled", False)):
        gw = config.setdefault("gateway", {})
        gw["enabled"] = True
        gw["port"] = int(prompt("  Gateway port", default=str(existing_gw.get("port", 9000))) or "9000")

    print_success("Advanced settings configured")


# ── Section registry ────────────────────────────────────────────────────────

SETUP_SECTIONS: list[tuple[str, str, callable]] = [
    ("model", "Model & Provider", setup_model),
    ("channel", "Messaging Channels", setup_channels),
    ("advanced", "Advanced Settings", setup_advanced),
]


# ── Banner ──────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    print()
    print(color("  ┌─────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("  │          Echo Agent Setup Wizard                │", Colors.CYAN))
    print(color("  ├─────────────────────────────────────────────────┤", Colors.CYAN))
    print(color("  │  Let's configure your Echo Agent installation.  │", Colors.CYAN))
    print(color("  │  Press Ctrl+C at any time to exit.              │", Colors.CYAN))
    print(color("  └─────────────────────────────────────────────────┘", Colors.CYAN))


def _print_summary(config: dict, config_path: Path) -> None:
    print_header("Setup Summary")
    models = config.get("models", {})
    providers = models.get("providers", [])
    if providers:
        p = providers[0]
        print_info(f"Provider:      {p.get('name', 'N/A')}")
    print_info(f"Default model: {models.get('defaultModel', 'N/A')}")

    channels = config.get("channels", {})
    enabled = [k for k, v in channels.items() if isinstance(v, dict) and v.get("enabled")]
    if enabled:
        print_info(f"Channels:      CLI + {', '.join(enabled)}")
    else:
        print_info("Channels:      CLI")

    print_info(f"Config file:   {config_path}")
    print()


# ── Main entry points ───────────────────────────────────────────────────────

def run_setup_wizard(section: str | None = None) -> None:
    if not is_interactive():
        print_error("Setup wizard requires an interactive terminal.")
        print_info("Configure manually by creating echo-agent.yaml")
        return

    # Load existing config if present
    existing_file = _find_config_file()
    config: dict = {}
    if existing_file:
        import yaml
        with open(existing_file, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Section-specific setup
    if section:
        for key, label, func in SETUP_SECTIONS:
            if key == section:
                _print_banner()
                func(config)
                path = save_config(config, existing_file)
                print_success(f"{label} configuration saved to {path}")
                return
        print_error(f"Unknown section: {section}")
        print_info(f"Available: {', '.join(k for k, _, _ in SETUP_SECTIONS)}")
        return

    _print_banner()

    is_existing = bool(existing_file and config.get("models", {}).get("providers"))

    if is_existing:
        print()
        print_success("Existing configuration detected.")
        menu = [
            "Quick Setup — reconfigure provider & model only",
            "Full Setup — reconfigure everything",
            "Model & Provider",
            "Messaging Channels",
            "Advanced Settings",
            "Exit",
        ]
        choice = prompt_choice("What would you like to do?", menu)
        if choice == 5:
            print_info("Exiting. Run 'echo-agent setup' again when ready.")
            return
        elif choice == 0:
            setup_model(config)
            path = save_config(config, existing_file)
            _print_summary(config, path)
            return
        elif choice == 1:
            pass  # fall through to full setup
        elif 2 <= choice <= 4:
            _, label, func = SETUP_SECTIONS[choice - 2]
            func(config)
            path = save_config(config, existing_file)
            print_success(f"{label} configuration saved to {path}")
            return
    else:
        mode = prompt_choice("How would you like to set up Echo Agent?", [
            "Quick Setup — provider, model & channels (recommended)",
            "Full Setup — configure everything",
        ])
        if mode == 0:
            setup_model(config)
            print()
            if prompt_yes_no("Configure messaging channels?", default=False):
                setup_channels(config)
            path = save_config(config)
            print()
            _print_summary(config, path)
            print_success("Setup complete! Run 'echo-agent' to start.")
            return

    # Full setup
    for _key, _label, func in SETUP_SECTIONS:
        func(config)

    path = save_config(config, existing_file)
    print()
    _print_summary(config, path)
    print_success("Setup complete! Run 'echo-agent' to start.")


def has_any_provider_configured() -> bool:
    config_file = _find_config_file()
    if not config_file:
        return False
    import yaml
    with open(config_file, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    providers = data.get("models", {}).get("providers", [])
    return bool(providers)


def prompt_first_run_setup() -> bool:
    if not is_interactive():
        return False
    if has_any_provider_configured():
        return False
    if _find_config_file():
        return False
    print()
    print_warning("No configuration found — Echo Agent is not set up yet.")
    if prompt_yes_no("Run setup wizard now?", default=True):
        run_setup_wizard()
        return True
    print_info("Skipping setup. Using default configuration (stub provider).")
    print_info("Run 'echo-agent setup' later to configure.")
    return False
