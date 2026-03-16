"""CLI commands for fubot."""

import asyncio
import os
import select
import signal
import sys
from pathlib import Path

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        os.environ["PYTHONIOENCODING"] = "utf-8"
        # Re-open stdout/stderr with UTF-8 encoding
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            os.environ.setdefault("PYTHONUTF8", "1")

import typer
from prompt_toolkit import print_formatted_text
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application import run_in_terminal
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from fubot import __logo__, __version__
from fubot.config.paths import get_workspace_path
from fubot.config.schema import Config
from fubot.utils.helpers import sync_workspace_templates

app = typer.Typer(
    name="fubot",
    help=f"{__logo__} fubot - Personal AI Assistant",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input: prompt_toolkit for editing, paste, history, and display
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        termios = None

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        return


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        _SAVED_TERM_ATTRS = None

    from fubot.config.paths import get_cli_history_path

    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,   # Enter submits (single line mode)
    )


def _make_console() -> Console:
    return Console(file=sys.stdout)


def _is_interactive_onboard() -> bool:
    """Return True when onboard is running in a real interactive terminal."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def _onboard_model_ready(config: Config) -> bool:
    """Return True when config looks usable for a first chat."""
    from fubot.providers.registry import find_by_name

    model = config.llm.model_id or config.agents.defaults.model
    provider_name = config.get_provider_name(model) or ""
    api_base = config.get_api_base(model)
    api_key = config.get_api_key(model)
    spec = find_by_name(provider_name) if provider_name else None

    if provider_name == "custom":
        return bool(config.llm.base_url and config.llm.model_id)
    if spec and spec.is_local:
        return bool(model and api_base)
    if spec and spec.is_oauth:
        return True
    return bool(model and (api_key or api_base))


def _prompt_required(prompt_text: str, default: str = "") -> str:
    """Prompt until a non-empty value is provided."""
    while True:
        value = typer.prompt(prompt_text, default=default).strip()
        if value:
            return value
        console.print("[red]This value is required.[/red]")


def _prompt_onboard_setup_mode(current: str | None = None) -> str:
    """Prompt for onboarding setup mode with a small validated choice set."""
    default = current or "custom"
    aliases = {
        "1": "custom",
        "2": "ollama",
        "3": "skip",
        "custom": "custom",
        "compatible": "custom",
        "openai": "custom",
        "ollama": "ollama",
        "local": "ollama",
        "skip": "skip",
    }
    while True:
        raw = typer.prompt(
            "LLM setup mode [custom/ollama/skip]",
            default=default,
            show_default=True,
        ).strip().lower()
        choice = aliases.get(raw)
        if choice:
            return choice
        console.print("[red]Choose one of: custom, ollama, skip.[/red]")


def _collect_onboard_llm_config(config: Config, mode: str) -> None:
    """Update config with interactive quickstart LLM settings."""
    if mode == "ollama":
        base_default = config.llm.base_url if "11434" in (config.llm.base_url or "") else "http://localhost:11434/v1"
        model_default = config.llm.model_id or "llama3.2"
        config.llm.provider = "custom"
        config.llm.base_url = _prompt_required("Ollama OpenAI-compatible base URL", default=base_default)
        config.llm.api_key = ""
        config.llm.model_id = _prompt_required("Ollama model name", default=model_default)
        config.agents.defaults.model = config.llm.model_id
        return

    config.llm.provider = "custom"
    base_default = config.llm.base_url or "https://api.openai.com/v1"
    model_default = config.llm.model_id
    config.llm.base_url = _prompt_required("OpenAI-compatible base URL", default=base_default)

    current_key = config.llm.api_key or ""
    key_prompt = "API key (leave blank if not required)"
    if current_key:
        entered = typer.prompt(
            "API key (leave blank to keep current)",
            default="",
            hide_input=True,
            show_default=False,
        ).strip()
        config.llm.api_key = entered or current_key
    else:
        config.llm.api_key = typer.prompt(
            key_prompt,
            default="",
            hide_input=True,
            show_default=False,
        ).strip()
    config.llm.model_id = _prompt_required("Model ID", default=model_default)
    config.agents.defaults.model = config.llm.model_id


def _verify_onboard_model_connection(config: Config) -> tuple[bool, str]:
    """Run a tiny model probe to verify quickstart settings."""
    model = config.llm.model_id or config.agents.defaults.model
    try:
        provider = _make_provider(config)
    except typer.Exit:
        return False, "provider configuration is incomplete"
    except Exception as exc:
        return False, str(exc)

    async def _probe() -> tuple[bool, str]:
        response = await provider.chat_with_retry(
            messages=[{"role": "user", "content": "Reply with exactly OK."}],
            model=model,
            max_tokens=16,
            temperature=0,
        )
        if response.finish_reason == "error":
            return False, (response.content or "unknown provider error").strip()
        preview = (response.content or "").strip().replace("\n", " ")
        return True, preview or "connected"

    try:
        return asyncio.run(_probe())
    except Exception as exc:
        return False, str(exc)


def _maybe_run_onboard_quickstart(config: Config, verify: bool) -> tuple[bool, bool]:
    """Offer an interactive quickstart so onboard can leave a usable config behind."""
    if not _is_interactive_onboard():
        return _onboard_model_ready(config), False

    already_ready = _onboard_model_ready(config)
    prompt = "Configure LLM now so fubot can be used immediately?"
    if not typer.confirm(prompt, default=not already_ready):
        return already_ready, False

    console.print("\n[bold]Quickstart[/bold]")
    console.print("Configure one model endpoint now so `fubot agent` works without manual config edits.")

    verified = False
    while True:
        mode = _prompt_onboard_setup_mode("custom")
        if mode == "skip":
            return _onboard_model_ready(config), verified

        _collect_onboard_llm_config(config, mode)

        from fubot.config.loader import save_config

        save_config(config)
        console.print(f"[green]✓[/green] Saved LLM settings for model [cyan]{config.llm.model_id}[/cyan]")

        if not verify or not typer.confirm("Test model connection now?", default=True):
            return _onboard_model_ready(config), verified

        console.print("[dim]Testing model connection...[/dim]")
        ok, detail = _verify_onboard_model_connection(config)
        if ok:
            console.print(f"[green]✓[/green] Connection OK: {detail}")
            return True, True

        console.print(f"[yellow]Connection test failed:[/yellow] {detail}")
        verified = False
        if not typer.confirm("Edit LLM settings again?", default=True):
            return _onboard_model_ready(config), verified


_ONBOARD_CHANNEL_ORDER = (
    "telegram",
    "discord",
    "slack",
    "whatsapp",
    "feishu",
    "dingtalk",
    "qq",
    "matrix",
    "wecom",
    "email",
    "mochat",
)

_ONBOARD_CHANNEL_LABELS = {
    "telegram": "Telegram",
    "discord": "Discord",
    "slack": "Slack",
    "whatsapp": "WhatsApp",
    "feishu": "Feishu",
    "dingtalk": "DingTalk",
    "qq": "QQ",
    "matrix": "Matrix",
    "wecom": "WeCom",
    "email": "Email",
    "mochat": "Mochat",
}

_ONBOARD_CHANNEL_DESCRIPTIONS = {
    "telegram": "bot token + allow_from",
    "discord": "bot token + allow_from",
    "slack": "bot token + app token + allow_from",
    "whatsapp": "bridge URL + allow_from, then QR login",
    "feishu": "app id + app secret + allow_from",
    "dingtalk": "client id + client secret + allow_from",
    "qq": "app id + secret + allow_from",
    "matrix": "homeserver + access token + user id + allow_from",
    "wecom": "bot id + secret + allow_from",
    "email": "IMAP/SMTP mailbox + allow_from + explicit consent",
    "mochat": "claw token + target auto-discovery + allow_from",
}

_ONBOARD_CHANNEL_ALIASES = {
    "telegram": "telegram",
    "tg": "telegram",
    "discord": "discord",
    "slack": "slack",
    "whatsapp": "whatsapp",
    "wa": "whatsapp",
    "feishu": "feishu",
    "lark": "feishu",
    "dingtalk": "dingtalk",
    "dingtalk": "dingtalk",
    "qq": "qq",
    "matrix": "matrix",
    "element": "matrix",
    "wecom": "wecom",
    "wechatwork": "wecom",
    "email": "email",
    "mail": "email",
    "mochat": "mochat",
}


def _default_csv(values: list[str] | tuple[str, ...] | None, fallback: str = "") -> str:
    """Format list values as prompt defaults."""
    items = [str(v).strip() for v in (values or []) if str(v).strip()]
    return ",".join(items) if items else fallback


def _prompt_optional(prompt_text: str, default: str = "") -> str:
    """Prompt for an optional free-text value."""
    return typer.prompt(
        prompt_text,
        default=default,
        show_default=bool(default),
    ).strip()


def _prompt_secret(
    prompt_text: str,
    current: str = "",
    *,
    required: bool = True,
) -> str:
    """Prompt for a secret with optional keep-current behavior."""
    while True:
        if current:
            value = typer.prompt(
                f"{prompt_text} (leave blank to keep current)",
                default="",
                hide_input=True,
                show_default=False,
            ).strip()
            if value:
                return value
            if current:
                return current
        else:
            value = typer.prompt(
                prompt_text,
                default="",
                hide_input=True,
                show_default=False,
            ).strip()
            if value or not required:
                return value
        console.print("[red]This value is required.[/red]")


def _prompt_int(prompt_text: str, default: int) -> int:
    """Prompt for an integer value."""
    while True:
        raw = typer.prompt(prompt_text, default=str(default)).strip()
        try:
            return int(raw)
        except ValueError:
            console.print("[red]Enter a valid integer.[/red]")


def _prompt_csv_values(
    prompt_text: str,
    current: list[str] | None = None,
    *,
    fallback: str = "",
    allow_empty: bool = False,
) -> list[str]:
    """Prompt for a comma-separated list."""
    while True:
        raw = typer.prompt(
            prompt_text,
            default=_default_csv(current, fallback),
            show_default=bool(_default_csv(current, fallback)),
        ).strip()
        if not raw:
            if allow_empty:
                return []
            console.print("[red]Enter at least one value.[/red]")
            continue
        values = [part.strip() for part in raw.split(",") if part.strip()]
        if values or allow_empty:
            return values
        console.print("[red]Enter at least one value.[/red]")


def _prompt_allow_from(channel_name: str, current: list[str] | None = None) -> list[str]:
    """Prompt for allow_from with a safe default."""
    return _prompt_csv_values(
        f"{channel_name} allow_from (comma-separated IDs, usernames, emails, or *)",
        current=current,
        fallback="*",
    )


def _prompt_channel_selection() -> list[str]:
    """Prompt for one or more channel names to configure."""
    console.print("\n[bold]Available channels[/bold]")
    for name in _ONBOARD_CHANNEL_ORDER:
        console.print(f"  - [cyan]{name}[/cyan]: {_ONBOARD_CHANNEL_DESCRIPTIONS[name]}")

    while True:
        raw = typer.prompt(
            "Channels to configure (comma-separated names, or none)",
            default="none",
            show_default=True,
        ).strip().lower()
        if raw in {"", "none", "skip"}:
            return []
        if raw == "all":
            return list(_ONBOARD_CHANNEL_ORDER)

        resolved: list[str] = []
        unknown: list[str] = []
        for part in raw.split(","):
            key = part.strip().lower().replace("-", "").replace("_", "")
            if not key:
                continue
            name = _ONBOARD_CHANNEL_ALIASES.get(key)
            if not name:
                unknown.append(part.strip())
                continue
            if name not in resolved:
                resolved.append(name)

        if resolved and not unknown:
            return resolved

        if unknown:
            console.print(f"[red]Unknown channels:[/red] {', '.join(unknown)}")
        else:
            console.print("[red]Choose at least one valid channel or 'none'.[/red]")


def _configure_telegram_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.telegram
    section.token = _prompt_secret("Telegram bot token", section.token)
    section.allow_from = _prompt_allow_from("Telegram", section.allow_from)
    section.enabled = True
    return True, []


def _configure_discord_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.discord
    section.token = _prompt_secret("Discord bot token", section.token)
    section.allow_from = _prompt_allow_from("Discord", section.allow_from)
    section.enabled = True
    return True, []


def _configure_slack_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.slack
    section.bot_token = _prompt_secret("Slack bot token (xoxb-...)", section.bot_token)
    section.app_token = _prompt_secret("Slack app token (xapp-...)", section.app_token)
    section.allow_from = _prompt_allow_from("Slack", section.allow_from)
    section.enabled = True
    return True, []


def _configure_whatsapp_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.whatsapp
    section.bridge_url = _prompt_required("WhatsApp bridge URL", section.bridge_url or "ws://localhost:3001")
    section.bridge_token = _prompt_secret(
        "WhatsApp bridge token (optional)",
        section.bridge_token,
        required=False,
    )
    section.allow_from = _prompt_allow_from("WhatsApp", section.allow_from)
    section.enabled = True
    return True, ["Run `fubot channels login` once to scan the WhatsApp QR code."]


def _configure_feishu_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.feishu
    section.app_id = _prompt_required("Feishu app id", section.app_id)
    section.app_secret = _prompt_secret("Feishu app secret", section.app_secret)
    section.allow_from = _prompt_allow_from("Feishu", section.allow_from)
    section.enabled = True
    return True, []


def _configure_dingtalk_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.dingtalk
    section.client_id = _prompt_required("DingTalk client id", section.client_id)
    section.client_secret = _prompt_secret("DingTalk client secret", section.client_secret)
    section.allow_from = _prompt_allow_from("DingTalk", section.allow_from)
    section.enabled = True
    return True, []


def _configure_qq_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.qq
    section.app_id = _prompt_required("QQ app id", section.app_id)
    section.secret = _prompt_secret("QQ app secret", section.secret)
    section.allow_from = _prompt_allow_from("QQ", section.allow_from)
    section.enabled = True
    return True, []


def _configure_matrix_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.matrix
    section.homeserver = _prompt_required("Matrix homeserver", section.homeserver or "https://matrix.org")
    section.access_token = _prompt_secret("Matrix access token", section.access_token)
    section.user_id = _prompt_required("Matrix user id", section.user_id)
    section.device_id = _prompt_optional("Matrix device id (optional)", section.device_id)
    section.allow_from = _prompt_allow_from("Matrix", section.allow_from)
    section.enabled = True
    return True, []


def _configure_wecom_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.wecom
    section.bot_id = _prompt_required("WeCom bot id", section.bot_id)
    section.secret = _prompt_secret("WeCom bot secret", section.secret)
    section.allow_from = _prompt_allow_from("WeCom", section.allow_from)
    section.enabled = True
    return True, []


def _configure_email_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.email
    consent_default = bool(section.consent_granted)
    if not typer.confirm(
        "You have explicit permission to let fubot read and reply from this mailbox?",
        default=consent_default,
    ):
        console.print("[yellow]Skipping Email setup without explicit consent.[/yellow]")
        return False, []

    section.consent_granted = True
    section.imap_host = _prompt_required("Email IMAP host", section.imap_host)
    section.imap_port = _prompt_int("Email IMAP port", section.imap_port)
    section.imap_username = _prompt_required("Email IMAP username", section.imap_username)
    section.imap_password = _prompt_secret("Email IMAP password", section.imap_password)
    section.smtp_host = _prompt_required("Email SMTP host", section.smtp_host)
    section.smtp_port = _prompt_int("Email SMTP port", section.smtp_port)
    section.smtp_username = _prompt_required("Email SMTP username", section.smtp_username)
    section.smtp_password = _prompt_secret("Email SMTP password", section.smtp_password)
    section.from_address = _prompt_optional(
        "Email from address (optional, defaults to SMTP username)",
        section.from_address or section.smtp_username,
    )
    section.allow_from = _prompt_allow_from("Email", section.allow_from)
    section.enabled = True
    return True, []


def _configure_mochat_channel(config: Config) -> tuple[bool, list[str]]:
    section = config.channels.mochat
    section.base_url = _prompt_required("Mochat base URL", section.base_url or "https://mochat.io")
    section.socket_url = _prompt_optional("Mochat socket URL (optional)", section.socket_url)
    section.claw_token = _prompt_secret("Mochat claw token", section.claw_token)
    section.agent_user_id = _prompt_optional(
        "Mochat agent user id (optional, improves mention detection)",
        section.agent_user_id,
    )
    section.sessions = _prompt_csv_values(
        "Mochat sessions to watch (comma-separated IDs, or * for auto-discover)",
        current=section.sessions,
        fallback="*",
    )
    section.panels = _prompt_csv_values(
        "Mochat panels to watch (comma-separated IDs, or * for auto-discover)",
        current=section.panels,
        fallback="*",
    )
    section.allow_from = _prompt_allow_from("Mochat", section.allow_from)
    section.enabled = True
    return True, []


_ONBOARD_CHANNEL_HANDLERS = {
    "telegram": _configure_telegram_channel,
    "discord": _configure_discord_channel,
    "slack": _configure_slack_channel,
    "whatsapp": _configure_whatsapp_channel,
    "feishu": _configure_feishu_channel,
    "dingtalk": _configure_dingtalk_channel,
    "qq": _configure_qq_channel,
    "matrix": _configure_matrix_channel,
    "wecom": _configure_wecom_channel,
    "email": _configure_email_channel,
    "mochat": _configure_mochat_channel,
}


def _maybe_run_onboard_channel_quickstart(config: Config) -> tuple[list[str], list[str]]:
    """Offer an interactive channel setup flow during onboarding."""
    if not _is_interactive_onboard():
        return [], []
    if not typer.confirm("Configure chat channels now?", default=False):
        return [], []

    selected = _prompt_channel_selection()
    if not selected:
        return [], []

    from fubot.config.loader import save_config

    configured: list[str] = []
    notes: list[str] = []
    console.print("\n[bold]Channel Quickstart[/bold]")
    for name in selected:
        label = _ONBOARD_CHANNEL_LABELS[name]
        console.print(f"[dim]Configuring {label}...[/dim]")
        applied, extra_notes = _ONBOARD_CHANNEL_HANDLERS[name](config)
        save_config(config)
        if applied:
            configured.append(label)
            console.print(f"[green]✓[/green] Configured {label}")
        for note in extra_notes:
            if note not in notes:
                notes.append(note)
    return configured, notes


def _render_interactive_ansi(render_fn) -> str:
    """Render Rich output to ANSI so prompt_toolkit can print it safely."""
    ansi_console = Console(
        force_terminal=True,
        color_system=console.color_system or "standard",
        width=console.width,
    )
    with ansi_console.capture() as capture:
        render_fn(ansi_console)
    return capture.get()


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """Render assistant response with consistent terminal styling."""
    console = _make_console()
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(f"[cyan]{__logo__} fubot[/cyan]")
    console.print(body)
    console.print()


async def _print_interactive_line(text: str) -> None:
    """Print async interactive updates with prompt_toolkit-safe Rich styling."""
    def _write() -> None:
        ansi = _render_interactive_ansi(
            lambda c: c.print(f"  [dim]↳ {text}[/dim]")
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


async def _print_interactive_response(response: str, render_markdown: bool) -> None:
    """Print async interactive replies with prompt_toolkit-safe Rich styling."""
    def _write() -> None:
        content = response or ""
        ansi = _render_interactive_ansi(
            lambda c: (
                c.print(),
                c.print(f"[cyan]{__logo__} fubot[/cyan]"),
                c.print(Markdown(content) if render_markdown else Text(content)),
                c.print(),
            )
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit (handles paste, history, display).

    prompt_toolkit natively handles:
    - Multiline paste (bracketed paste mode)
    - History navigation (up/down arrows)
    - Clean display (no ghost characters or artifacts)
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc



def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} fubot v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """fubot - Personal AI Assistant."""
    return None


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard(
    quickstart: bool = typer.Option(
        True,
        "--quickstart/--no-quickstart",
        help="Prompt for a first-run LLM setup when running interactively.",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Test the configured model connection during quickstart setup.",
    ),
):
    """Initialize fubot configuration and workspace."""
    from fubot.config.loader import get_config_path, load_config, save_config
    from fubot.config.schema import Config

    config_path = get_config_path()

    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
        console.print("  [bold]N[/bold] = refresh config, keeping existing values and adding new fields")
        if typer.confirm("Overwrite?"):
            config = Config()
            save_config(config)
            console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
        else:
            config = load_config()
            save_config(config)
            console.print(f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)")
    else:
        config = Config()
        save_config(config)
        console.print(f"[green]✓[/green] Created config at {config_path}")

    console.print("[dim]Config template now uses `maxTokens` + `contextWindowTokens`; `memoryWindow` is no longer a runtime setting.[/dim]")

    # Create workspace
    workspace = get_workspace_path()

    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace}")

    sync_workspace_templates(workspace)

    ready = _onboard_model_ready(config)
    verified = False
    configured_channels: list[str] = []
    channel_notes: list[str] = []
    if quickstart:
        ready, verified = _maybe_run_onboard_quickstart(config, verify=verify)
        configured_channels, channel_notes = _maybe_run_onboard_channel_quickstart(config)
    else:
        ready = _onboard_model_ready(config)

    console.print(f"\n{__logo__} fubot is ready!")
    console.print("\nNext steps:")
    if ready:
        if verified:
            console.print("  1. Config verified. Start chatting: [cyan]fubot agent -m \"Hello!\"[/cyan]")
        else:
            console.print("  1. Config saved. Start chatting: [cyan]fubot agent -m \"Hello!\"[/cyan]")
        console.print("  2. Check current setup anytime with: [cyan]fubot status[/cyan]")
    else:
        console.print("  1. Add your API key to [cyan]~/.fubot/config.json[/cyan]")
        console.print("     Get one at: https://openrouter.ai/keys")
        console.print("  2. Chat: [cyan]fubot agent -m \"Hello!\"[/cyan]")

    if configured_channels:
        console.print(f"\nConfigured channels: [cyan]{', '.join(configured_channels)}[/cyan]")
        if ready:
            console.print("Start the gateway with: [cyan]fubot gateway[/cyan]")
        else:
            console.print("After finishing LLM setup, start channels with: [cyan]fubot gateway[/cyan]")
        for note in channel_notes:
            console.print(f"  - {note}")
    console.print("\n[dim]Want Telegram/WhatsApp? See README.md for channel setup.[/dim]")





def _make_provider(config: Config):
    """Create the appropriate LLM provider from config."""
    from fubot.providers.factory import ProviderConfigurationError, build_provider

    try:
        return build_provider(config)
    except ProviderConfigurationError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


def _load_runtime_config(config: str | None = None, workspace: str | None = None) -> Config:
    """Load config and optionally override the active workspace."""
    from fubot.config.loader import load_config, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")

    loaded = load_config(config_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return loaded


def _print_deprecated_memory_window_notice(config: Config) -> None:
    """Warn when running with old memoryWindow-only config."""
    if config.agents.defaults.should_warn_deprecated_memory_window:
        console.print(
            "[yellow]Hint:[/yellow] Detected deprecated `memoryWindow` without "
            "`contextWindowTokens`. `memoryWindow` is ignored; run "
            "[cyan]fubot onboard[/cyan] to refresh your config template."
        )


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command()
def gateway(
    port: int | None = typer.Option(None, "--port", "-p", help="Gateway port"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Start the fubot gateway."""
    from fubot.agent.loop import AgentLoop
    from fubot.bus.queue import MessageBus
    from fubot.channels.manager import ChannelManager
    from fubot.config.paths import get_cron_dir
    from fubot.cron.service import CronService
    from fubot.cron.types import CronJob
    from fubot.heartbeat.service import HeartbeatService
    from fubot.session.manager import SessionManager

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    config = _load_runtime_config(config, workspace)
    _print_deprecated_memory_window_notice(config)
    port = port if port is not None else config.gateway.port

    console.print(f"{__logo__} Starting fubot gateway on port {port}...")
    sync_workspace_templates(config.workspace_path)
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)

    # Create cron service first (callback set after agent creation)
    cron_store_path = get_cron_dir() / "jobs.json"
    cron = CronService(cron_store_path)

    # Create agent with cron service
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.llm.model_id or config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        runtime_config=config,
    )

    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent."""
        from fubot.agent.tools.cron import CronTool
        from fubot.agent.tools.message import MessageTool
        reminder_note = (
            "[Scheduled Task] Timer finished.\n\n"
            f"Task '{job.name}' has been triggered.\n"
            f"Scheduled instruction: {job.payload.message}"
        )

        # Prevent the agent from scheduling new cron jobs during execution
        cron_tool = agent.tools.get("cron")
        cron_token = None
        if isinstance(cron_tool, CronTool):
            cron_token = cron_tool.set_cron_context(True)
        try:
            response = await agent.process_direct(
                reminder_note,
                session_key=f"cron:{job.id}",
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to or "direct",
            )
        finally:
            if isinstance(cron_tool, CronTool) and cron_token is not None:
                cron_tool.reset_cron_context(cron_token)

        message_tool = agent.tools.get("message")
        if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
            return response

        if job.payload.deliver and job.payload.to and response:
            from fubot.bus.events import OutboundMessage
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to,
                content=response
            ))
        return response
    cron.on_job = on_cron_job

    # Create channel manager
    channels = ChannelManager(config, bus)

    def _pick_heartbeat_target() -> tuple[str, str]:
        """Pick a routable channel/chat target for heartbeat-triggered messages."""
        enabled = set(channels.enabled_channels)
        # Prefer the most recently updated non-internal session on an enabled channel.
        for item in session_manager.list_sessions():
            key = item.get("key") or ""
            if ":" not in key:
                continue
            channel, chat_id = key.split(":", 1)
            if channel in {"cli", "system"}:
                continue
            if channel in enabled and chat_id:
                return channel, chat_id
        # Fallback keeps prior behavior but remains explicit.
        return "cli", "direct"

    # Create heartbeat service
    async def on_heartbeat_execute(tasks: str) -> str:
        """Phase 2: execute heartbeat tasks through the full agent loop."""
        channel, chat_id = _pick_heartbeat_target()

        async def _silent(*_args, **_kwargs):
            return None

        return await agent.process_direct(
            tasks,
            session_key="heartbeat",
            channel=channel,
            chat_id=chat_id,
            on_progress=_silent,
        )

    async def on_heartbeat_notify(response: str) -> None:
        """Deliver a heartbeat response to the user's channel."""
        from fubot.bus.events import OutboundMessage
        channel, chat_id = _pick_heartbeat_target()
        if channel == "cli":
            return  # No external channel available to deliver to
        await bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, content=response))

    hb_cfg = config.gateway.heartbeat
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        provider=provider,
        model=agent.model,
        on_execute=on_heartbeat_execute,
        on_notify=on_heartbeat_notify,
        interval_s=hb_cfg.interval_s,
        enabled=hb_cfg.enabled,
    )

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    console.print(f"[green]✓[/green] Heartbeat: every {hb_cfg.interval_s}s")

    async def run():
        try:
            await cron.start()
            await heartbeat.start()
            await asyncio.gather(
                agent.run(),
                channels.start_all(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            await agent.close_mcp()
            heartbeat.stop()
            cron.stop()
            agent.stop()
            await channels.stop_all()

    asyncio.run(run())




# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show fubot runtime logs during chat"),
):
    """Interact with the agent directly."""
    from loguru import logger

    from fubot.agent.loop import AgentLoop
    from fubot.bus.queue import MessageBus
    from fubot.config.paths import get_cron_dir
    from fubot.cron.service import CronService
    from fubot.orchestrator.router import ProviderExecutionError

    config = _load_runtime_config(config, workspace)
    _print_deprecated_memory_window_notice(config)
    sync_workspace_templates(config.workspace_path)

    bus = MessageBus()
    provider = _make_provider(config)

    # Create cron service for tool usage (no callback needed for CLI unless running)
    cron_store_path = get_cron_dir() / "jobs.json"
    cron = CronService(cron_store_path)

    if logs:
        logger.enable("fubot")
    else:
        logger.disable("fubot")

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.llm.model_id or config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        runtime_config=config,
    )

    # Show spinner when logs are off (no output to miss); skip when logs are on
    def _thinking_ctx():
        if logs:
            from contextlib import nullcontext
            return nullcontext()
        # Animated spinner is safe to use with prompt_toolkit input handling
        return console.status("[dim]fubot is thinking...[/dim]", spinner="dots")

    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        ch = agent_loop.channels_config
        if ch and tool_hint and not ch.send_tool_hints:
            return
        if ch and not tool_hint and not ch.send_progress:
            return
        console.print(f"  [dim]↳ {content}[/dim]")

    if message:
        # Single message mode — direct call, no bus needed
        async def run_once():
            try:
                with _thinking_ctx():
                    return await agent_loop.process_direct(
                        message,
                        session_id,
                        on_progress=_cli_progress,
                    )
            finally:
                await agent_loop.close_mcp()

        try:
            response = asyncio.run(run_once())
        except ProviderExecutionError as exc:
            console.print(f"[red]Agent failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        except Exception as exc:
            console.print(f"[red]Agent failed:[/red] {exc}")
            raise typer.Exit(1) from exc

        _print_agent_response(response, render_markdown=markdown)
    else:
        # Interactive mode — route through bus like other channels
        from fubot.bus.events import InboundMessage
        _init_prompt_session()
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            _restore_terminal()
            console.print(f"\nReceived {sig_name}, goodbye!")
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        # SIGHUP is not available on Windows
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, _handle_signal)
        # Ignore SIGPIPE to prevent silent process termination when writing to closed pipes
        # SIGPIPE is not available on Windows
        if hasattr(signal, 'SIGPIPE'):
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        async def run_interactive():
            bus_task = asyncio.create_task(agent_loop.run())
            turn_done = asyncio.Event()
            turn_done.set()
            turn_response: list[str] = []

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        if msg.metadata.get("_progress"):
                            is_tool_hint = msg.metadata.get("_tool_hint", False)
                            ch = agent_loop.channels_config
                            if ch and is_tool_hint and not ch.send_tool_hints:
                                continue
                            elif ch and not is_tool_hint and not ch.send_progress:
                                continue
                            else:
                                await _print_interactive_line(msg.content)

                        elif not turn_done.is_set():
                            if msg.content:
                                turn_response.append(msg.content)
                            turn_done.set()
                        elif msg.content:
                            await _print_interactive_response(msg.content, render_markdown=markdown)

                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            outbound_task = asyncio.create_task(_consume_outbound())

            try:
                while True:
                    try:
                        _flush_pending_tty_input()
                        user_input = await _read_interactive_input_async()
                        command = user_input.strip()
                        if not command:
                            continue

                        if _is_exit_command(command):
                            _restore_terminal()
                            console.print("\nGoodbye!")
                            break

                        turn_done.clear()
                        turn_response.clear()

                        await bus.publish_inbound(InboundMessage(
                            channel=cli_channel,
                            sender_id="user",
                            chat_id=cli_chat_id,
                            content=user_input,
                        ))

                        with _thinking_ctx():
                            await turn_done.wait()

                        if turn_response:
                            _print_agent_response(turn_response[0], render_markdown=markdown)
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await agent_loop.close_mcp()

        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from fubot.channels.registry import discover_channel_names, load_channel_class
    from fubot.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")

    for modname in sorted(discover_channel_names()):
        section = getattr(config.channels, modname, None)
        enabled = section and getattr(section, "enabled", False)
        try:
            cls = load_channel_class(modname)
            display = cls.display_name
        except ImportError:
            display = modname.title()
        table.add_row(
            display,
            "[green]\u2713[/green]" if enabled else "[dim]\u2717[/dim]",
        )

    console.print(table)


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess

    # User's bridge location
    from fubot.config.paths import get_bridge_install_dir

    user_bridge = get_bridge_install_dir()

    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge

    # Check for npm
    if not shutil.which("npm"):
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)

    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # fubot/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)

    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge

    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall fubot")
        raise typer.Exit(1)

    console.print(f"{__logo__} Setting up bridge...")

    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))

    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=user_bridge, check=True, capture_output=True)

        console.print("  Building...")
        subprocess.run(["npm", "run", "build"], cwd=user_bridge, check=True, capture_output=True)

        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)

    return user_bridge


@channels_app.command("login")
def channels_login():
    """Link device via QR code."""
    import subprocess

    from fubot.config.loader import load_config
    from fubot.config.paths import get_runtime_subdir

    config = load_config()
    bridge_dir = _get_bridge_dir()

    console.print(f"{__logo__} Starting bridge...")
    console.print("Scan the QR code to connect.\n")

    env = {**os.environ}
    if config.channels.whatsapp.bridge_token:
        env["BRIDGE_TOKEN"] = config.channels.whatsapp.bridge_token
    env["AUTH_DIR"] = str(get_runtime_subdir("whatsapp-auth"))

    try:
        subprocess.run(["npm", "start"], cwd=bridge_dir, check=True, env=env)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Bridge failed: {e}[/red]")
    except FileNotFoundError:
        console.print("[red]npm not found. Please install Node.js.[/red]")


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Show fubot status."""
    from fubot.config.loader import get_config_path, load_config, set_config_path

    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        set_config_path(config_path)
        runtime_config = load_config(config_path)
    else:
        config_path = get_config_path()
        runtime_config = load_config()
    workspace = runtime_config.workspace_path

    console.print(f"{__logo__} fubot Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from fubot.providers.registry import PROVIDERS

        console.print(f"Model: {runtime_config.llm.model_id or runtime_config.agents.defaults.model}")

        # Check API keys from registry
        for spec in PROVIDERS:
            p = getattr(runtime_config.providers, spec.name, None)
            llm_override_active = runtime_config.llm.provider == spec.name and bool(
                runtime_config.llm.model_id or runtime_config.agents.defaults.model
            )
            effective_api_key = runtime_config.llm.api_key if llm_override_active else (
                p.api_key if p else ""
            )
            effective_api_base = runtime_config.llm.base_url if llm_override_active else (
                p.api_base if p else None
            )
            if p is None:
                continue
            if spec.is_oauth:
                console.print(f"{spec.label}: [green]✓ (OAuth)[/green]")
            elif spec.is_local or (llm_override_active and effective_api_base):
                # Local deployments show api_base instead of api_key
                if effective_api_base:
                    console.print(f"{spec.label}: [green]✓ {effective_api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(effective_api_key)
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


# ============================================================================
# OAuth Login
# ============================================================================

provider_app = typer.Typer(help="Manage providers")
app.add_typer(provider_app, name="provider")


_LOGIN_HANDLERS: dict[str, callable] = {}


def _register_login(name: str):
    def decorator(fn):
        _LOGIN_HANDLERS[name] = fn
        return fn
    return decorator


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="OAuth provider (e.g. 'openai-codex', 'github-copilot')"),
):
    """Authenticate with an OAuth provider."""
    from fubot.providers.registry import PROVIDERS

    key = provider.replace("-", "_")
    spec = next((s for s in PROVIDERS if s.name == key and s.is_oauth), None)
    if not spec:
        names = ", ".join(s.name.replace("_", "-") for s in PROVIDERS if s.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)

    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Login - {spec.label}\n")
    handler()


@_register_login("openai_codex")
def _login_openai_codex() -> None:
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive
        token = None
        try:
            token = get_token()
        except Exception:
            token = None
        if not (token and token.access):
            console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
            token = login_oauth_interactive(
                print_fn=lambda s: console.print(s),
                prompt_fn=lambda s: typer.prompt(s),
            )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]")
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)


@_register_login("github_copilot")
def _login_github_copilot() -> None:
    import asyncio

    console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")

    async def _trigger():
        from litellm import acompletion
        await acompletion(model="github_copilot/gpt-4o", messages=[{"role": "user", "content": "hi"}], max_tokens=1)

    try:
        asyncio.run(_trigger())
        console.print("[green]✓ Authenticated with GitHub Copilot[/green]")
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
