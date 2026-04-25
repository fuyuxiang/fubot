"""Tests for SessionManager atomic file writes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from echo_agent.session.manager import Session, SessionManager


@pytest.fixture
def manager(tmp_path: Path) -> SessionManager:
    return SessionManager(sessions_dir=tmp_path / "sessions")


@pytest.mark.asyncio
async def test_save_creates_valid_jsonl(manager: SessionManager) -> None:
    session = Session(key="test:1")
    session.add_message("user", "hello")
    session.add_message("assistant", "hi there")
    await manager.save(session)

    path = manager._session_path("test:1")
    assert path.exists()

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    meta = json.loads(lines[0])
    assert meta["_type"] == "metadata"
    assert meta["key"] == "test:1"

    msg1 = json.loads(lines[1])
    assert msg1["role"] == "user"
    assert msg1["content"] == "hello"


@pytest.mark.asyncio
async def test_atomic_write_preserves_original_on_error(manager: SessionManager, monkeypatch) -> None:
    session = Session(key="safe:1")
    session.add_message("user", "original")
    await manager.save(session)

    path = manager._session_path("safe:1")
    original_content = path.read_text(encoding="utf-8")

    original_replace = os.replace

    def failing_replace(src, dst):
        os.unlink(src)
        raise OSError("simulated disk failure")

    monkeypatch.setattr("os.replace", failing_replace)

    session2 = Session(key="safe:1")
    session2.add_message("user", "corrupted")

    with pytest.raises(OSError, match="simulated disk failure"):
        manager._save_to_file(session2)

    monkeypatch.undo()
    assert path.read_text(encoding="utf-8") == original_content


@pytest.mark.asyncio
async def test_no_temp_files_left_after_save(manager: SessionManager) -> None:
    session = Session(key="clean:1")
    session.add_message("user", "test")
    await manager.save(session)

    tmp_files = list(manager.sessions_dir.glob(".sess_*.tmp"))
    assert len(tmp_files) == 0


@pytest.mark.asyncio
async def test_roundtrip_preserves_data(manager: SessionManager) -> None:
    session = Session(key="rt:1")
    session.add_message("user", "msg1")
    session.add_message("assistant", "msg2")
    session.metadata = {"channel": "test"}
    await manager.save(session)

    manager.invalidate("rt:1")
    loaded = await manager.get_or_create("rt:1")
    assert len(loaded.messages) == 2
    assert loaded.messages[0]["content"] == "msg1"
    assert loaded.metadata.get("channel") == "test"
