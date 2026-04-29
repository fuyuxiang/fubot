"""Tests for echo_agent/skills/store.py and echo_agent/skills/manager.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from echo_agent.skills.store import SkillStore, parse_frontmatter
from echo_agent.skills.manager import SkillManager, SkillManifest, SkillStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_SKILL_CONTENT = """\
---
name: my-skill
description: A test skill
version: "1.0.0"
---

# My Skill

Body text here.
"""


def _make_skill_on_disk(root: Path, name: str, description: str = "A skill", category: str = "") -> Path:
    """Create a minimal skill directory with SKILL.md."""
    parent = root / category / name if category else root / name
    parent.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n\nBody.\n"
    (parent / "SKILL.md").write_text(content, encoding="utf-8")
    return parent


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        fm, body = parse_frontmatter("---\nname: foo\n---\n\nHello")
        assert fm["name"] == "foo"
        assert "Hello" in body

    def test_missing_frontmatter(self):
        fm, body = parse_frontmatter("Just plain text")
        assert fm == {}
        assert body == "Just plain text"

    def test_invalid_yaml(self):
        fm, body = parse_frontmatter("---\n: [invalid\n---\n\nBody")
        assert fm == {}
        assert "Body" in body

    def test_no_closing_fence(self):
        fm, body = parse_frontmatter("---\nname: bar\nno closing")
        assert fm == {}


# ---------------------------------------------------------------------------
# SkillStore
# ---------------------------------------------------------------------------

class TestSkillStore:
    def test_create_and_list(self, tmp_path: Path):
        store = SkillStore(user_dir=tmp_path / "skills")
        err = store.create_skill("hello", _VALID_SKILL_CONTENT.replace("my-skill", "hello"))
        assert err is None
        skills = store.list_all()
        assert any(s.name == "hello" for s in skills)

    def test_read_skill(self, tmp_path: Path):
        store = SkillStore(user_dir=tmp_path / "skills")
        store.create_skill("reader", _VALID_SKILL_CONTENT.replace("my-skill", "reader"))
        content = store.read_skill("reader")
        assert content is not None
        assert "reader" in content

    def test_delete_skill(self, tmp_path: Path):
        store = SkillStore(user_dir=tmp_path / "skills")
        store.create_skill("doomed", _VALID_SKILL_CONTENT.replace("my-skill", "doomed"))
        err = store.delete_skill("doomed")
        assert err is None
        assert store.read_skill("doomed") is None

    def test_delete_nonexistent(self, tmp_path: Path):
        store = SkillStore(user_dir=tmp_path / "skills")
        err = store.delete_skill("ghost")
        assert err is not None

    def test_invalid_name_rejected(self, tmp_path: Path):
        store = SkillStore(user_dir=tmp_path / "skills")
        err = store.create_skill("INVALID NAME!", _VALID_SKILL_CONTENT)
        assert err is not None

    def test_duplicate_name_rejected(self, tmp_path: Path):
        store = SkillStore(user_dir=tmp_path / "skills")
        store.create_skill("dup", _VALID_SKILL_CONTENT.replace("my-skill", "dup"))
        err = store.create_skill("dup", _VALID_SKILL_CONTENT.replace("my-skill", "dup"))
        assert err is not None
        assert "already exists" in err

    def test_path_traversal_blocked_read_file(self, tmp_path: Path):
        store = SkillStore(user_dir=tmp_path / "skills")
        store.create_skill("safe", _VALID_SKILL_CONTENT.replace("my-skill", "safe"))
        assert store.read_file("safe", "../../etc/passwd") is None
        assert store.read_file("safe", "/etc/passwd") is None

    def test_path_traversal_blocked_write_file(self, tmp_path: Path):
        store = SkillStore(user_dir=tmp_path / "skills")
        store.create_skill("safe2", _VALID_SKILL_CONTENT.replace("my-skill", "safe2"))
        err = store.write_file("safe2", "../evil.txt", "bad")
        assert err is not None

    def test_list_excludes_disabled(self, tmp_path: Path):
        user_dir = tmp_path / "skills"
        _make_skill_on_disk(user_dir, "enabled-skill")
        _make_skill_on_disk(user_dir, "disabled-skill")
        store = SkillStore(user_dir=user_dir, disabled=["disabled-skill"])
        names = [s.name for s in store.list_all()]
        assert "enabled-skill" in names
        assert "disabled-skill" not in names

    def test_create_missing_description(self, tmp_path: Path):
        store = SkillStore(user_dir=tmp_path / "skills")
        content = "---\nname: nodesc\n---\n\nBody"
        err = store.create_skill("nodesc", content)
        assert err is not None
        assert "description" in err


# ---------------------------------------------------------------------------
# SkillManager
# ---------------------------------------------------------------------------

class TestSkillManager:
    @staticmethod
    def _make_source(tmp_path: Path, name: str = "test-skill") -> Path:
        src = tmp_path / "source" / name
        src.mkdir(parents=True, exist_ok=True)
        (src / "SKILL.md").write_text(f"---\nname: {name}\n---\n\nContent.\n", encoding="utf-8")
        return src

    def test_install_and_get(self, tmp_path: Path):
        mgr = SkillManager(tmp_path / "installed")
        src = self._make_source(tmp_path, "alpha")
        skill = mgr.install("alpha", src)
        assert skill.manifest.name == "alpha"
        assert skill.status == SkillStatus.INSTALLED
        assert mgr.get_skill("alpha") is not None

    def test_enable_and_disable(self, tmp_path: Path):
        mgr = SkillManager(tmp_path / "installed")
        src = self._make_source(tmp_path, "beta")
        mgr.install("beta", src)
        assert mgr.enable("beta") is True
        assert mgr.get_skill("beta").status == SkillStatus.ENABLED
        assert mgr.disable("beta") is True
        assert mgr.get_skill("beta").status == SkillStatus.DISABLED

    def test_uninstall(self, tmp_path: Path):
        mgr = SkillManager(tmp_path / "installed")
        src = self._make_source(tmp_path, "gamma")
        mgr.install("gamma", src)
        assert mgr.uninstall("gamma") is True
        assert mgr.get_skill("gamma") is None
        assert mgr.uninstall("gamma") is False

    def test_configure(self, tmp_path: Path):
        mgr = SkillManager(tmp_path / "installed")
        src = self._make_source(tmp_path, "delta")
        mgr.install("delta", src)
        assert mgr.configure("delta", {"key": "value"}) is True
        assert mgr.get_skill("delta").config["key"] == "value"

    def test_list_skills_filter(self, tmp_path: Path):
        mgr = SkillManager(tmp_path / "installed")
        for name in ("s1", "s2", "s3"):
            src = self._make_source(tmp_path, name)
            mgr.install(name, src)
        mgr.enable("s1")
        enabled = mgr.list_skills(status=SkillStatus.ENABLED)
        assert len(enabled) == 1
        assert enabled[0].manifest.name == "s1"
        all_skills = mgr.list_skills()
        assert len(all_skills) == 3

    def test_enable_with_unmet_deps_fails(self, tmp_path: Path):
        mgr = SkillManager(tmp_path / "installed")
        src = self._make_source(tmp_path, "needy")
        manifest = SkillManifest(name="needy", dependencies=["missing-dep"])
        mgr.install("needy", src, manifest=manifest)
        assert mgr.enable("needy") is False
        assert mgr.get_skill("needy").status != SkillStatus.ENABLED

    def test_enable_nonexistent(self, tmp_path: Path):
        mgr = SkillManager(tmp_path / "installed")
        assert mgr.enable("nope") is False

    def test_configure_nonexistent(self, tmp_path: Path):
        mgr = SkillManager(tmp_path / "installed")
        assert mgr.configure("nope", {"a": 1}) is False

