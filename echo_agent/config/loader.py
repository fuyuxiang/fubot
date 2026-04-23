"""Configuration loader — reads YAML, env vars, and CLI overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from echo_agent.config.schema import Config

_DEFAULT_CONFIG_NAMES = ("echo-agent.yaml", "echo-agent.yml", "config.yaml", "config.yml")
_PACKAGED_DEFAULT_CONFIG = Path(__file__).with_name("default.yaml")


def _find_config_file(search_dir: Path | None = None) -> Path | None:
    base = search_dir or Path.cwd()
    for name in _DEFAULT_CONFIG_NAMES:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def resolve_config_file(config_path: str | Path | None = None, search_dir: str | Path | None = None) -> Path | None:
    if config_path:
        path = Path(config_path).expanduser()
        return path.resolve() if path.exists() else path
    base = Path(search_dir).expanduser() if search_dir else None
    found = _find_config_file(base)
    return found.resolve() if found else None


def _load_yaml_file(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides() -> dict[str, Any]:
    """Collect ECHO_AGENT_ prefixed env vars into a nested dict."""
    prefix = "ECHO_AGENT_"
    result: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    data: dict[str, Any] = _load_yaml_file(_PACKAGED_DEFAULT_CONFIG)

    path = resolve_config_file(config_path)
    if path and path.exists():
        logger.info("Loading config from {}", path)
        data = _deep_merge(data, _load_yaml_file(path))

    env = _env_overrides()
    if env:
        data = _deep_merge(data, env)

    if overrides:
        data = _deep_merge(data, overrides)

    return Config(**data)


def save_config(data: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write configuration dict to a YAML file.

    Returns the path written to.
    """
    target = Path(path) if path else Path.cwd() / "echo-agent.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return target
