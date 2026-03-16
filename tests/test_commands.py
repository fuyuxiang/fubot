import json
import re
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from fubot.cli.commands import app
from fubot.config.schema import Config
from fubot.orchestrator.router import ProviderExecutionError
from fubot.providers.litellm_provider import LiteLLMProvider
from fubot.providers.openai_codex_provider import _strip_model_prefix
from fubot.providers.registry import find_by_model


def _strip_ansi(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)

runner = CliRunner()


class _StopGatewayError(RuntimeError):
    """Sentinel exception used to stop the gateway during tests."""


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with patch("fubot.config.loader.get_config_path") as mock_cp, \
         patch("fubot.config.loader.save_config") as mock_sc, \
         patch("fubot.config.loader.load_config"), \
         patch("fubot.cli.commands.get_workspace_path") as mock_ws:

        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_sc.side_effect = lambda config: config_file.write_text("{}")

        yield config_file, workspace_dir

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "fubot is ready" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_interactive_quickstart_writes_llm_config(tmp_path, monkeypatch):
    """Interactive onboard should leave behind a usable runtime llm config."""
    config_path = tmp_path / "config.json"
    workspace_dir = tmp_path / "workspace"

    monkeypatch.setattr("fubot.config.loader.get_config_path", lambda: config_path)
    monkeypatch.setattr("fubot.cli.commands.get_workspace_path", lambda: workspace_dir)
    monkeypatch.setattr("fubot.cli.commands._is_interactive_onboard", lambda: True)
    monkeypatch.setattr("fubot.cli.commands._verify_onboard_model_connection", lambda _cfg: (True, "OK"))

    result = runner.invoke(
        app,
        ["onboard"],
        input="\ncustom\nhttps://example.com/v1\nsk-test\nexample-model\n\nn\n",
    )

    assert result.exit_code == 0
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["llm"]["provider"] == "custom"
    assert saved["llm"]["baseUrl"] == "https://example.com/v1"
    assert saved["llm"]["apiKey"] == "sk-test"
    assert saved["llm"]["modelId"] == "example-model"
    assert saved["agents"]["defaults"]["model"] == "example-model"
    assert "Connection OK" in result.stdout
    assert "Config verified" in result.stdout


def test_onboard_interactive_channel_quickstart_writes_telegram_config(tmp_path, monkeypatch):
    """Interactive onboard can configure a chat channel end-to-end."""
    config_path = tmp_path / "config.json"
    workspace_dir = tmp_path / "workspace"

    monkeypatch.setattr("fubot.config.loader.get_config_path", lambda: config_path)
    monkeypatch.setattr("fubot.cli.commands.get_workspace_path", lambda: workspace_dir)
    monkeypatch.setattr("fubot.cli.commands._is_interactive_onboard", lambda: True)

    result = runner.invoke(
        app,
        ["onboard", "--no-verify"],
        input="\ncustom\nhttps://example.com/v1\nsk-test\nexample-model\ny\ntelegram\n123456:ABCDEF\n*\n",
    )

    assert result.exit_code == 0
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["channels"]["telegram"]["enabled"] is True
    assert saved["channels"]["telegram"]["token"] == "123456:ABCDEF"
    assert saved["channels"]["telegram"]["allowFrom"] == ["*"]
    assert "Configured Telegram" in result.stdout
    assert "Configured channels: Telegram" in result.stdout


def test_onboard_no_quickstart_keeps_manual_setup_path(tmp_path, monkeypatch):
    """Users can still opt out of the interactive wizard for scripting or CI."""
    config_path = tmp_path / "config.json"
    workspace_dir = tmp_path / "workspace"

    monkeypatch.setattr("fubot.config.loader.get_config_path", lambda: config_path)
    monkeypatch.setattr("fubot.cli.commands.get_workspace_path", lambda: workspace_dir)
    monkeypatch.setattr("fubot.cli.commands._is_interactive_onboard", lambda: True)

    result = runner.invoke(app, ["onboard", "--no-quickstart"])

    assert result.exit_code == 0
    assert "Add your API key" in result.stdout


def test_config_matches_github_copilot_codex_with_hyphen_prefix():
    config = Config()
    config.agents.defaults.model = "github-copilot/gpt-5.3-codex"

    assert config.get_provider_name() == "github_copilot"


def test_config_matches_openai_codex_with_hyphen_prefix():
    config = Config()
    config.agents.defaults.model = "openai-codex/gpt-5.1-codex"

    assert config.get_provider_name() == "openai_codex"


def test_config_matches_explicit_ollama_prefix_without_api_key():
    config = Config()
    config.agents.defaults.model = "ollama/llama3.2"

    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434"


def test_config_explicit_ollama_provider_uses_default_localhost_api_base():
    config = Config()
    config.agents.defaults.provider = "ollama"
    config.agents.defaults.model = "llama3.2"

    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434"


def test_config_auto_detects_ollama_from_local_api_base():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "llama3.2"}},
            "providers": {"ollama": {"apiBase": "http://localhost:11434"}},
        }
    )

    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434"


def test_config_prefers_ollama_over_vllm_when_both_local_providers_configured():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "llama3.2"}},
            "providers": {
                "vllm": {"apiBase": "http://localhost:8000"},
                "ollama": {"apiBase": "http://localhost:11434"},
            },
        }
    )

    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434"


def test_config_falls_back_to_vllm_when_ollama_not_configured():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "llama3.2"}},
            "providers": {
                "vllm": {"apiBase": "http://localhost:8000"},
            },
        }
    )

    assert config.get_provider_name() == "vllm"
    assert config.get_api_base() == "http://localhost:8000"


def test_find_by_model_prefers_explicit_prefix_over_generic_codex_keyword():
    spec = find_by_model("github-copilot/gpt-5.3-codex")

    assert spec is not None
    assert spec.name == "github_copilot"


def test_litellm_provider_canonicalizes_github_copilot_hyphen_prefix():
    provider = LiteLLMProvider(default_model="github-copilot/gpt-5.3-codex")

    resolved = provider._resolve_model("github-copilot/gpt-5.3-codex")

    assert resolved == "github_copilot/gpt-5.3-codex"


def test_openai_codex_strip_prefix_supports_hyphen_and_underscore():
    assert _strip_model_prefix("openai-codex/gpt-5.1-codex") == "gpt-5.1-codex"
    assert _strip_model_prefix("openai_codex/gpt-5.1-codex") == "gpt-5.1-codex"


@pytest.fixture
def mock_agent_runtime(tmp_path):
    """Mock agent command dependencies for focused CLI tests."""
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "default-workspace")
    cron_dir = tmp_path / "data" / "cron"

    with patch("fubot.config.loader.load_config", return_value=config) as mock_load_config, \
         patch("fubot.config.paths.get_cron_dir", return_value=cron_dir), \
         patch("fubot.cli.commands.sync_workspace_templates") as mock_sync_templates, \
         patch("fubot.cli.commands._make_provider", return_value=object()), \
         patch("fubot.cli.commands._print_agent_response") as mock_print_response, \
         patch("fubot.bus.queue.MessageBus"), \
         patch("fubot.cron.service.CronService"), \
         patch("fubot.agent.loop.AgentLoop") as mock_agent_loop_cls:

        agent_loop = MagicMock()
        agent_loop.channels_config = None
        agent_loop.process_direct = AsyncMock(return_value="mock-response")
        agent_loop.close_mcp = AsyncMock(return_value=None)
        mock_agent_loop_cls.return_value = agent_loop

        yield {
            "config": config,
            "load_config": mock_load_config,
            "sync_templates": mock_sync_templates,
            "agent_loop_cls": mock_agent_loop_cls,
            "agent_loop": agent_loop,
            "print_response": mock_print_response,
        }


def test_agent_help_shows_workspace_and_config_options():
    result = runner.invoke(app, ["agent", "--help"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "--workspace" in stripped_output
    assert "-w" in stripped_output
    assert "--config" in stripped_output
    assert "-c" in stripped_output


def test_agent_uses_default_config_when_no_workspace_or_config_flags(mock_agent_runtime):
    result = runner.invoke(app, ["agent", "-m", "hello"])

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (None,)
    assert mock_agent_runtime["sync_templates"].call_args.args == (
        mock_agent_runtime["config"].workspace_path,
    )
    assert mock_agent_runtime["agent_loop_cls"].call_args.kwargs["workspace"] == (
        mock_agent_runtime["config"].workspace_path
    )
    mock_agent_runtime["agent_loop"].process_direct.assert_awaited_once()
    mock_agent_runtime["print_response"].assert_called_once_with("mock-response", render_markdown=True)


def test_agent_uses_explicit_config_path(mock_agent_runtime, tmp_path: Path):
    config_path = tmp_path / "agent-config.json"
    config_path.write_text("{}")

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_path)])

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (config_path.resolve(),)


def test_agent_config_sets_active_path(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "fubot.config.loader.set_config_path",
        lambda path: seen.__setitem__("config_path", path),
    )
    monkeypatch.setattr("fubot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("fubot.config.paths.get_cron_dir", lambda: config_file.parent / "cron")
    monkeypatch.setattr("fubot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("fubot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("fubot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("fubot.cron.service.CronService", lambda _store: object())

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        async def process_direct(self, *_args, **_kwargs) -> str:
            return "ok"

        async def close_mcp(self) -> None:
            return None

    monkeypatch.setattr("fubot.agent.loop.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("fubot.cli.commands._print_agent_response", lambda *_args, **_kwargs: None)

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert seen["config_path"] == config_file.resolve()


def test_agent_overrides_workspace_path(mock_agent_runtime):
    workspace_path = Path("/tmp/agent-workspace")

    result = runner.invoke(app, ["agent", "-m", "hello", "-w", str(workspace_path)])

    assert result.exit_code == 0
    assert mock_agent_runtime["config"].agents.defaults.workspace == str(workspace_path)
    assert mock_agent_runtime["sync_templates"].call_args.args == (workspace_path,)
    assert mock_agent_runtime["agent_loop_cls"].call_args.kwargs["workspace"] == workspace_path


def test_agent_workspace_override_wins_over_config_workspace(mock_agent_runtime, tmp_path: Path):
    config_path = tmp_path / "agent-config.json"
    config_path.write_text("{}")
    workspace_path = Path("/tmp/agent-workspace")

    result = runner.invoke(
        app,
        ["agent", "-m", "hello", "-c", str(config_path), "-w", str(workspace_path)],
    )

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (config_path.resolve(),)
    assert mock_agent_runtime["config"].agents.defaults.workspace == str(workspace_path)
    assert mock_agent_runtime["sync_templates"].call_args.args == (workspace_path,)
    assert mock_agent_runtime["agent_loop_cls"].call_args.kwargs["workspace"] == workspace_path


def test_agent_warns_about_deprecated_memory_window(mock_agent_runtime):
    mock_agent_runtime["config"].agents.defaults.memory_window = 100

    result = runner.invoke(app, ["agent", "-m", "hello"])

    assert result.exit_code == 0
    assert "memoryWindow" in result.stdout
    assert "contextWindowTokens" in result.stdout


def test_agent_single_message_reports_provider_failure_without_traceback(mock_agent_runtime):
    mock_agent_runtime["agent_loop"].process_direct = AsyncMock(
        side_effect=ProviderExecutionError("Provider fallback exhausted", error_kind="connection")
    )

    result = runner.invoke(app, ["agent", "-m", "hello"])

    assert result.exit_code == 1
    assert "Agent failed:" in result.stdout
    assert "Provider fallback exhausted" in result.stdout
    assert "Traceback" not in result.stdout
    mock_agent_runtime["agent_loop"].close_mcp.assert_awaited_once()


def test_gateway_uses_workspace_from_config_by_default(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "fubot.config.loader.set_config_path",
        lambda path: seen.__setitem__("config_path", path),
    )
    monkeypatch.setattr("fubot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr(
        "fubot.cli.commands.sync_workspace_templates",
        lambda path: seen.__setitem__("workspace", path),
    )
    monkeypatch.setattr(
        "fubot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["config_path"] == config_file.resolve()
    assert seen["workspace"] == Path(config.agents.defaults.workspace)


def test_gateway_workspace_option_overrides_config(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    override = tmp_path / "override-workspace"
    seen: dict[str, Path] = {}

    monkeypatch.setattr("fubot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("fubot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr(
        "fubot.cli.commands.sync_workspace_templates",
        lambda path: seen.__setitem__("workspace", path),
    )
    monkeypatch.setattr(
        "fubot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(
        app,
        ["gateway", "--config", str(config_file), "--workspace", str(override)],
    )

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["workspace"] == override
    assert config.workspace_path == override


def test_gateway_warns_about_deprecated_memory_window(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.memory_window = 100

    monkeypatch.setattr("fubot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("fubot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("fubot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr(
        "fubot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert "memoryWindow" in result.stdout
    assert "contextWindowTokens" in result.stdout

def test_gateway_uses_config_directory_for_cron_store(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    seen: dict[str, Path] = {}

    monkeypatch.setattr("fubot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("fubot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("fubot.config.paths.get_cron_dir", lambda: config_file.parent / "cron")
    monkeypatch.setattr("fubot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("fubot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("fubot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("fubot.session.manager.SessionManager", lambda _workspace: object())

    class _StopCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path
            raise _StopGatewayError("stop")

    monkeypatch.setattr("fubot.cron.service.CronService", _StopCron)

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["cron_store"] == config_file.parent / "cron" / "jobs.json"


def test_gateway_uses_configured_port_when_cli_flag_is_missing(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.gateway.port = 18791

    monkeypatch.setattr("fubot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("fubot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("fubot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr(
        "fubot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert "port 18791" in result.stdout


def test_gateway_cli_port_overrides_configured_port(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.gateway.port = 18791

    monkeypatch.setattr("fubot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("fubot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("fubot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr(
        "fubot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file), "--port", "18792"])

    assert isinstance(result.exception, _StopGatewayError)
    assert "port 18792" in result.stdout


def test_gateway_default_port_matches_compose_and_docs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    port = Config().gateway.port

    dockerfile = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    compose = (repo_root / "docker-compose.yml").read_text(encoding="utf-8")
    regression_doc = (repo_root / "docs" / "regression-audit.md").read_text(encoding="utf-8")

    assert f"EXPOSE {port}" in dockerfile
    assert f"- {port}:{port}" in compose
    assert f"`{port}`" in regression_doc


def test_status_uses_explicit_config_path(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "status-workspace")
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "fubot.config.loader.set_config_path",
        lambda path: seen.__setitem__("config_path", path),
    )
    monkeypatch.setattr("fubot.config.loader.load_config", lambda _path=None: config)

    result = runner.invoke(app, ["status", "--config", str(config_file)])

    assert result.exit_code == 0
    assert seen["config_path"] == config_file.resolve()
    stripped_output = _strip_ansi(result.stdout)
    collapsed_output = stripped_output.replace("\n", "")
    assert str(config_file.resolve()) in collapsed_output
    assert str(config.workspace_path) in collapsed_output


def test_status_reports_runtime_custom_provider_from_llm_config(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.llm.provider = "custom"
    config.llm.base_url = "https://example.com/v1"
    config.llm.api_key = "sk-test"
    config.llm.model_id = "example-model"

    monkeypatch.setattr("fubot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("fubot.config.loader.load_config", lambda _path=None: config)

    result = runner.invoke(app, ["status", "--config", str(config_file)])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "Model: example-model" in stripped_output
    assert "Custom: ✓ https://example.com/v1" in stripped_output
