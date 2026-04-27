"""Agent-facing skill install tool — install skills from git, local path, or URL.

Fetches skill sources into a temp directory, validates SKILL.md presence,
and copies into the SkillStore user directory.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from loguru import logger

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult
from echo_agent.skills.store import SkillStore, parse_frontmatter

_TIMEOUT = 60
_GIT_URL_RE = re.compile(
    r"^(https?://[^\s]+\.git|git@[^\s]+\.git|https?://github\.com/[^\s]+|https?://gitlab\.com/[^\s]+)$"
)
_SAFE_URL_RE = re.compile(r"^https?://[^\s]+$")
_SAFE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

_SAFE_PIP_PKG = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(\[[a-zA-Z0-9,._-]+\])?(([><=!~]=?|===?)[a-zA-Z0-9.*_-]+)?$")
_SAFE_BREW_FORMULA = re.compile(r"^[a-z0-9][a-z0-9+._@-]*(\/[a-z0-9][a-z0-9+._@-]*){0,2}$")


async def _run(cmd: list[str], cwd: str | None = None, timeout: int = _TIMEOUT) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"Command timed out after {timeout}s"
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _fetch_git(location: str, tmpdir: str) -> tuple[str | None, str]:
    if not _GIT_URL_RE.match(location):
        return None, f"Invalid git URL: {location}"
    dest = str(Path(tmpdir) / "repo")
    code, _, stderr = await _run(["git", "clone", "--depth", "1", location, dest])
    if code != 0:
        return None, f"git clone failed: {stderr.strip()}"
    return dest, ""


async def _fetch_url(location: str, tmpdir: str) -> tuple[str | None, str]:
    if not _SAFE_URL_RE.match(location):
        return None, f"Invalid URL: {location}"
    dest = Path(tmpdir) / "download"
    dest.mkdir()
    archive = Path(tmpdir) / "archive"
    code, _, stderr = await _run(["curl", "-fsSL", "-o", str(archive), location])
    if code != 0:
        return None, f"Download failed: {stderr.strip()}"
    if location.endswith(".zip"):
        try:
            with zipfile.ZipFile(str(archive)) as zf:
                zf.extractall(str(dest))
        except zipfile.BadZipFile:
            return None, "Downloaded file is not a valid zip"
    else:
        code, _, stderr = await _run(["tar", "xf", str(archive), "-C", str(dest)])
        if code != 0:
            return None, f"Extract failed: {stderr.strip()}"
    return str(dest), ""


def _fetch_local(location: str) -> tuple[str | None, str]:
    p = Path(location).expanduser().resolve()
    if not p.exists():
        return None, f"Path not found: {location}"
    if not p.is_dir():
        return None, f"Not a directory: {location}"
    return str(p), ""


def _find_skill_md(base: str, subdirectory: str) -> tuple[Path | None, str]:
    root = Path(base)
    if subdirectory:
        root = root / subdirectory
    if not root.is_dir():
        return None, f"Subdirectory not found: {subdirectory}"
    skill_md = root / "SKILL.md"
    if skill_md.exists():
        return root, ""
    for candidate in root.rglob("SKILL.md"):
        return candidate.parent, ""
    return None, "No SKILL.md found in source"


def _resolve_name(skill_dir: Path, override: str) -> tuple[str, str]:
    if override:
        if not _SAFE_NAME_RE.match(override):
            return "", f"Invalid skill name: {override}"
        return override, ""
    skill_md = skill_dir / "SKILL.md"
    fm, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    name = fm.get("name", "")
    if not name:
        name = skill_dir.name
    if not _SAFE_NAME_RE.match(name):
        return "", f"Skill name '{name}' from frontmatter is invalid, provide a name override"
    return name, ""


async def _run_install_specs(specs: list[dict], timeout: int = _TIMEOUT) -> list[str]:
    results: list[str] = []
    for spec in specs:
        kind = spec.get("kind", "")
        if kind == "pip":
            pkg = spec.get("package", "")
            if not pkg or not _SAFE_PIP_PKG.match(pkg):
                results.append(f"[pip] skipped unsafe package: {pkg}")
                continue
            code, out, err = await _run(["pip", "install", pkg], timeout=timeout)
            results.append(f"[pip] {pkg}: {'ok' if code == 0 else err.strip()}")
        elif kind == "brew":
            formula = spec.get("formula", "")
            if not formula or not _SAFE_BREW_FORMULA.match(formula):
                results.append(f"[brew] skipped unsafe formula: {formula}")
                continue
            code, out, err = await _run(["brew", "install", formula], timeout=timeout)
            results.append(f"[brew] {formula}: {'ok' if code == 0 else err.strip()}")
        elif kind == "shell":
            cmd = spec.get("command", "")
            if not cmd:
                continue
            results.append(f"[shell] skipped for safety: {cmd[:80]}")
        else:
            results.append(f"[{kind}] unknown install kind, skipped")
    return results


class SkillInstallTool(Tool):
    name = "skill_install"
    description = (
        "Install a skill from an external source into the local skill store. "
        "Supported sources: 'git' (clone a repo), 'local' (copy from filesystem path), "
        "'url' (download tarball/zip). The source must contain a SKILL.md file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["git", "local", "url"],
                "description": "Where to fetch the skill from",
            },
            "location": {
                "type": "string",
                "description": "Git URL, local path, or download URL",
            },
            "name": {
                "type": "string",
                "description": "Override skill name (default: read from SKILL.md frontmatter)",
            },
            "subdirectory": {
                "type": "string",
                "description": "Path within the source to the skill directory (for monorepos)",
            },
            "run_install": {
                "type": "boolean",
                "description": "Run install specs from skill metadata (default true)",
            },
        },
        "required": ["source", "location"],
    }

    def __init__(self, store: SkillStore):
        self._store = store

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        source = params["source"]
        location = params["location"]
        name_override = params.get("name", "")
        subdirectory = params.get("subdirectory", "")
        run_install = params.get("run_install", True)

        tmpdir = tempfile.mkdtemp(prefix="echo_skill_")
        try:
            if source == "git":
                fetched, err = await _fetch_git(location, tmpdir)
            elif source == "url":
                fetched, err = await _fetch_url(location, tmpdir)
            elif source == "local":
                fetched, err = _fetch_local(location)
            else:
                return ToolResult(success=False, error=f"Unknown source: {source}")

            if err:
                return ToolResult(success=False, error=err)

            skill_dir, err = _find_skill_md(fetched, subdirectory)
            if err:
                return ToolResult(success=False, error=err)

            skill_name, err = _resolve_name(skill_dir, name_override)
            if err:
                return ToolResult(success=False, error=err)

            skill_md = skill_dir / "SKILL.md"
            fm, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            category = fm.get("category", "")

            install_err = self._store.create_skill(
                skill_name,
                skill_md.read_text(encoding="utf-8"),
                category=category,
            )
            if install_err and "already exists" in install_err:
                install_err = self._store.update_skill(skill_name, skill_md.read_text(encoding="utf-8"))
            if install_err:
                return ToolResult(success=False, error=install_err)

            for subdir_name in ("references", "templates", "scripts", "assets"):
                src_sub = skill_dir / subdir_name
                if src_sub.is_dir():
                    for f in src_sub.rglob("*"):
                        if f.is_file():
                            rel = str(f.relative_to(skill_dir))
                            self._store.write_file(skill_name, rel, f.read_text(encoding="utf-8"))

            install_results: list[str] = []
            if run_install:
                meta = fm.get("metadata", {}) or {}
                echo_meta = meta.get("echo", {}) or {}
                specs = echo_meta.get("install", [])
                if specs:
                    install_results = await _run_install_specs(specs)

            output = f"Skill '{skill_name}' installed successfully."
            if install_results:
                output += "\n\nInstall results:\n" + "\n".join(f"  {r}" for r in install_results)
            logger.info("Installed skill '{}' from {} ({})", skill_name, source, location)
            return ToolResult(success=True, output=output)

        except Exception as e:
            logger.error("Skill install failed: {}", e)
            return ToolResult(success=False, error=f"Install failed: {e}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
