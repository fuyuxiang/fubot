"""Tests for echo_agent/config/loader.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from echo_agent.config.loader import (
    _deep_merge,
    _env_overrides,
    _find_config_file_in,
    _load_yaml_file,
    resolve_config_file,
    save_config,
)


# ---------------------------------------------------------------------------
# _find_config_file_in
# ---------------------------------------------------------------------------

class TestFindConfigFileIn:
    def test_finds_yaml(self, tmp_path: Path):
        (tmp_path / "echo-agent.yaml").write_text("key: val\n")
        assert _find_config_file_in(tmp_path) == tmp_path / "echo-agent.yaml"

    def test_finds_yml(self, tmp_path: Path):
        (tmp_path / "echo-agent.yml").write_text("key: val\n")
        assert _find_config_file_in(tmp_path) == tmp_path / "echo-agent.yml"

    def test_prefers_yaml_over_yml(self, tmp_path: Path):
        (tmp_path / "echo-agent.yaml").write_text("a: 1\n")
        (tmp_path / "echo-agent.yml").write_text("b: 2\n")
        result = _find_config_file_in(tmp_path)
        assert result.name == "echo-agent.yaml"

    def test_finds_config_yaml(self, tmp_path: Path):
        (tmp_path / "config.yaml").write_text("x: 1\n")
        assert _find_config_file_in(tmp_path) == tmp_path / "config.yaml"

    def test_returns_none_when_missing(self, tmp_path: Path):
        assert _find_config_file_in(tmp_path) is None


# ---------------------------------------------------------------------------
# _load_yaml_file
# ---------------------------------------------------------------------------

class TestLoadYamlFile:
    def test_valid_yaml(self, tmp_path: Path):
        p = tmp_path / "test.yaml"
        p.write_text("foo: bar\nnested:\n  a: 1\n")
        data = _load_yaml_file(p)
        assert data["foo"] == "bar"
        assert data["nested"]["a"] == 1

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert _load_yaml_file(p) == {}

    def test_missing_file(self, tmp_path: Path):
        assert _load_yaml_file(tmp_path / "nope.yaml") == {}

    def test_none_path(self):
        assert _load_yaml_file(None) == {}


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_nested_dicts(self):
        base = {"a": {"x": 1, "y": 2}, "b": 10}
        override = {"a": {"y": 99, "z": 3}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 3}, "b": 10}

    def test_override_scalar(self):
        base = {"a": 1, "b": 2}
        override = {"a": 100}
        result = _deep_merge(base, override)
        assert result["a"] == 100
        assert result["b"] == 2

    def test_override_dict_with_scalar(self):
        base = {"a": {"nested": True}}
        override = {"a": "flat"}
        result = _deep_merge(base, override)
        assert result["a"] == "flat"

    def test_empty_override(self):
        base = {"a": 1}
        assert _deep_merge(base, {}) == {"a": 1}

    def test_empty_base(self):
        override = {"a": 1}
        assert _deep_merge({}, override) == {"a": 1}


# ---------------------------------------------------------------------------
# _env_overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_simple_prefix(self, monkeypatch):
        monkeypatch.setenv("ECHO_AGENT_MODEL", "gpt-4")
        result = _env_overrides()
        assert result["model"] == "gpt-4"

    def test_nested_with_double_underscore(self, monkeypatch):
        monkeypatch.setenv("ECHO_AGENT_LLM__PROVIDER", "openai")
        result = _env_overrides()
        assert result["llm"]["provider"] == "openai"

    def test_ignores_unrelated_vars(self, monkeypatch):
        monkeypatch.setenv("OTHER_VAR", "nope")
        monkeypatch.delenv("ECHO_AGENT_MODEL", raising=False)
        monkeypatch.delenv("ECHO_AGENT_LLM__PROVIDER", raising=False)
        result = _env_overrides()
        assert "other_var" not in result


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------

class TestSaveConfig:
    def test_writes_yaml(self, tmp_path: Path):
        target = tmp_path / "out.yaml"
        data = {"model": "gpt-4", "nested": {"key": "val"}}
        result = save_config(data, path=target)
        assert result == target
        loaded = yaml.safe_load(target.read_text())
        assert loaded["model"] == "gpt-4"
        assert loaded["nested"]["key"] == "val"

    def test_creates_parent_dirs(self, tmp_path: Path):
        target = tmp_path / "sub" / "dir" / "config.yaml"
        save_config({"a": 1}, path=target)
        assert target.exists()


# ---------------------------------------------------------------------------
# resolve_config_file
# ---------------------------------------------------------------------------

class TestResolveConfigFile:
    def test_explicit_path(self, tmp_path: Path):
        cfg = tmp_path / "my.yaml"
        cfg.write_text("x: 1\n")
        result = resolve_config_file(config_path=cfg)
        assert result == cfg.resolve()

    def test_explicit_path_nonexistent(self, tmp_path: Path):
        cfg = tmp_path / "missing.yaml"
        result = resolve_config_file(config_path=cfg)
        assert result == cfg

    def test_search_dir(self, tmp_path: Path):
        (tmp_path / "echo-agent.yaml").write_text("a: 1\n")
        result = resolve_config_file(search_dir=tmp_path)
        assert result is not None
        assert result.name == "echo-agent.yaml"

