"""Memory system — user memory, environment memory, CRUD, retrieval, consolidation.

Two-layer design:
  - User memory: preferences, habits, long-term requirements
  - Environment memory: project background, tool docs, process rules, domain knowledge

The implementation keeps the structured memory model used by Echo Agent, while
using built-in safety properties:
  - per-target file locking to avoid stale concurrent writes
  - atomic file replacement for durable persistence
  - prompt-injection / exfiltration scanning before memory is accepted
  - bounded snapshot rendering for system-prompt injection
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from echo_agent.memory.types import MemoryEntry, MemoryTier, MemoryType, Episode, Contradiction
from echo_agent.memory.forgetting import ForgettingCurve

msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "it", "its",
    "this", "that", "and", "or", "but", "not", "no", "if", "so", "than",
})

_MEMORY_THREAT_PATTERNS = [
    (r"ignore\s+(previous|all|above|prior)\s+instructions", "prompt_injection"),
    (r"you\s+are\s+now\s+", "role_hijack"),
    (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
    (r"system\s+prompt\s+override", "sys_prompt_override"),
    (r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)", "disregard_rules"),
    (r"act\s+as\s+(if|though)\s+you\s+(have\s+no|don't\s+have)\s+(restrictions|limits|rules)", "bypass_restrictions"),
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_curl"),
    (r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_wget"),
    (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)", "read_secrets"),
    (r"authorized_keys", "ssh_backdoor"),
    (r"\$HOME/\.ssh|\~/\.ssh", "ssh_access"),
    (r"\$HOME/\.echo-agent|\~/\.echo-agent", "agent_secret_path"),
]

_INVISIBLE_CHARS = {
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
}


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _normalize_tags(tags: Iterable[str]) -> list[str]:
    normalized = [tag.strip() for tag in tags if tag and tag.strip()]
    return _dedupe_keep_order(normalized)


def _scan_memory_content(content: str) -> str | None:
    for char in _INVISIBLE_CHARS:
        if char in content:
            return (
                f"Blocked: content contains invisible unicode character U+{ord(char):04X} "
                "(possible injection)."
            )

    for pattern, threat_id in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return (
                f"Blocked: content matches threat pattern '{threat_id}'. "
                "Memory entries are injected into prompts and must not contain "
                "injection or exfiltration payloads."
            )
    return None


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".mem_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class MemoryStore:
    """Persistent memory store with file-based storage, safety checks, and scored search."""

    def __init__(
        self,
        memory_dir: Path,
        max_user: int = 1000,
        max_env: int = 500,
        decay_half_life_days: float = 30.0,
        user_snapshot_char_limit: int = 1375,
        env_snapshot_char_limit: int = 2200,
        storage: Any = None,
    ):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._user_file = memory_dir / "user_memory.json"
        self._env_file = memory_dir / "env_memory.json"
        self._history_file = memory_dir / "HISTORY.md"
        self._long_term_file = memory_dir / "MEMORY.md"
        self._max_user = max_user
        self._max_env = max_env
        self._decay_half_life = decay_half_life_days
        self._user_snapshot_char_limit = user_snapshot_char_limit
        self._env_snapshot_char_limit = env_snapshot_char_limit
        self._entries: dict[str, MemoryEntry] = {}
        self._storage = storage
        self._load()
        self._forgetting = ForgettingCurve(
            base_half_life_days=decay_half_life_days,
            archive_threshold=0.05,
            forget_threshold=0.01,
        )
        self._vector_index = None  # set externally via set_vector_index()
        self._retriever = None  # set externally via set_retriever()
        self._embed_fn = None  # async callable: str -> list[float]
        self._pending_embeds: list[tuple[str, str]] = []  # (entry_id, text) pairs awaiting embedding

    def set_vector_index(self, index):
        self._vector_index = index

    def set_embed_fn(self, fn):
        self._embed_fn = fn

    def set_retriever(self, retriever):
        self._retriever = retriever

    def _queue_embed(self, entry: MemoryEntry) -> None:
        if self._embed_fn and self._vector_index:
            text = f"{entry.key} {entry.content}" if entry.key else entry.content
            self._pending_embeds.append((entry.id, text))

    async def flush_pending_embeds(self) -> int:
        """Generate embeddings for queued entries and add to vector index. Returns count."""
        if not self._pending_embeds or not self._embed_fn or not self._vector_index:
            return 0
        batch = list(self._pending_embeds)
        self._pending_embeds.clear()
        count = 0
        for entry_id, text in batch:
            if entry_id not in self._entries:
                continue
            try:
                embedding = await self._embed_fn(text)
                if embedding:
                    vec_id = await self._vector_index.add(entry_id, embedding)
                    if vec_id:
                        self._entries[entry_id].embedding_id = vec_id
                        count += 1
            except Exception as e:
                logger.debug("Embedding generation failed for {}: {}", entry_id, e)
        if count:
            logger.info("Generated {} embeddings", count)
        return count

    @property
    def forgetting_curve(self) -> ForgettingCurve:
        return self._forgetting

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """跨平台文件锁（Unix 用 fcntl，Windows 用 msvcrt）。"""
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
            lock_path.write_text(" ", encoding="utf-8")

        fd = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    def _path_for(self, mem_type: MemoryType) -> Path:
        return self._user_file if mem_type == MemoryType.USER else self._env_file

    def _typed_entries(self, mem_type: MemoryType) -> list[MemoryEntry]:
        return [entry for entry in self._entries.values() if entry.type == mem_type]

    def _filtered_entries(
        self, mem_type: MemoryType | None = None, session_key: str | None = None,
    ) -> list[MemoryEntry]:
        """按类型和会话可见性过滤记忆条目。"""
        entries = list(self._entries.values())
        if mem_type is not None:
            entries = [entry for entry in entries if entry.type == mem_type]
        if session_key:
            entries = [entry for entry in entries if self._visible_in_session(entry, session_key)]
        return entries

    def _visible_in_session(self, entry: MemoryEntry, session_key: str | None = None) -> bool:
        if not session_key:
            return True
        if entry.type == MemoryType.ENVIRONMENT:
            return True
        if "global" in entry.tags:
            return True
        return entry.source_session == session_key

    def _same_scope(self, existing: MemoryEntry, incoming: MemoryEntry) -> bool:
        if existing.type != incoming.type:
            return False
        if incoming.type == MemoryType.ENVIRONMENT:
            return True
        return existing.source_session == incoming.source_session

    def _load_type_from_disk(self, mem_type: MemoryType) -> list[MemoryEntry]:
        path = self._path_for(mem_type)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load memory from {}: {}", path, exc)
            return []

        entries: list[MemoryEntry] = []
        seen_ids: set[str] = set()
        for item in raw:
            try:
                entry = MemoryEntry.from_dict(item)
            except Exception as exc:
                logger.warning("Failed to parse memory entry from {}: {}", path, exc)
                continue
            entry.type = mem_type
            if entry.id in seen_ids:
                continue
            seen_ids.add(entry.id)
            entries.append(entry)
        return entries

    def _reload_type(self, mem_type: MemoryType) -> None:
        for entry_id in [entry.id for entry in self._typed_entries(mem_type)]:
            self._entries.pop(entry_id, None)
        for entry in self._load_type_from_disk(mem_type):
            self._entries[entry.id] = entry

    def _save_type(self, mem_type: MemoryType) -> None:
        """将指定类型的记忆条目原子写入磁盘。"""
        entries = self._typed_entries(mem_type)
        entries.sort(key=lambda entry: (entry.created_at or "", entry.updated_at or "", entry.id))
        payload = [entry.to_dict() for entry in entries]
        _atomic_write_text(
            self._path_for(mem_type),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        if self._storage:
            import asyncio
            for entry in entries:
                try:
                    loop = asyncio.get_event_loop()
                    coro = self._storage.store_memory(entry.id, entry.to_dict())
                    if loop.is_running():
                        asyncio.ensure_future(coro)
                    else:
                        loop.run_until_complete(coro)
                except Exception as e:
                    logger.warning("Failed to sync memory {} to storage: {}", entry.id, e)

    def _load(self) -> None:
        self._reload_type(MemoryType.USER)
        self._reload_type(MemoryType.ENVIRONMENT)

    def _find_conflict(self, entry: MemoryEntry) -> MemoryEntry | None:
        if not entry.key:
            return None
        for existing in self._typed_entries(entry.type):
            if existing.key == entry.key and self._same_scope(existing, entry):
                return existing
        return None

    def _validate_content(self, content: str, *, field_name: str = "content") -> str:
        normalized = content.strip()
        if not normalized:
            raise ValueError(f"{field_name} cannot be empty")
        scan_error = _scan_memory_content(normalized)
        if scan_error:
            raise ValueError(scan_error)
        return normalized

    def _evict_oldest(self, mem_type: MemoryType) -> None:
        """淘汰有效重要性最低的记忆条目，为新条目腾出空间。"""
        typed = sorted(
            self._typed_entries(mem_type),
            key=lambda entry: (self._forgetting.effective_importance(entry), entry.updated_at or "", entry.id),
        )
        if typed:
            self._entries.pop(typed[0].id, None)

    def _merge_locked(self, existing_id: str, new_entry: MemoryEntry) -> MemoryEntry:
        existing = self._entries[existing_id]
        existing.content = self._validate_content(new_entry.content)
        existing.tags = _normalize_tags([*existing.tags, *new_entry.tags])
        existing.importance = max(existing.importance, new_entry.importance)
        existing.updated_at = datetime.now().isoformat()
        return existing

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add(self, entry: MemoryEntry) -> MemoryEntry:
        """添加记忆条目。自动去重、冲突合并、容量淘汰。"""
        entry.content = self._validate_content(entry.content)
        entry.key = entry.key.strip()
        entry.tags = _normalize_tags(entry.tags)

        path = self._path_for(entry.type)
        with self._file_lock(path):
            self._reload_type(entry.type)

            duplicate = next(
                (
                    existing
                    for existing in self._typed_entries(entry.type)
                    if (
                        existing.key == entry.key
                        and existing.content == entry.content
                        and self._same_scope(existing, entry)
                    )
                ),
                None,
            )
            if duplicate:
                return duplicate

            existing = self._find_conflict(entry)
            if existing:
                merged = self._merge_locked(existing.id, entry)
                self._save_type(entry.type)
                return merged

            limit = self._max_user if entry.type == MemoryType.USER else self._max_env
            if len(self._typed_entries(entry.type)) >= limit:
                self._evict_oldest(entry.type)

            self._entries[entry.id] = entry
            self._save_type(entry.type)
            self._queue_embed(entry)
            return entry

    def update(
        self,
        entry_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry | None:
        entry = self._entries.get(entry_id)
        if not entry:
            return None

        normalized_content = self._validate_content(content) if content is not None else None
        normalized_tags = _normalize_tags(tags) if tags is not None else None

        path = self._path_for(entry.type)
        with self._file_lock(path):
            self._reload_type(entry.type)
            entry = self._entries.get(entry_id)
            if not entry:
                return None
            if normalized_content is not None:
                entry.content = normalized_content
            if normalized_tags is not None:
                entry.tags = normalized_tags
            entry.updated_at = datetime.now().isoformat()
            self._save_type(entry.type)
            if normalized_content is not None:
                self._queue_embed(entry)
            return entry

    def delete(self, entry_id: str) -> bool:
        entry = self._entries.get(entry_id)
        if not entry:
            return False

        path = self._path_for(entry.type)
        with self._file_lock(path):
            self._reload_type(entry.type)
            entry = self._entries.get(entry_id)
            if not entry:
                return False
            self._entries.pop(entry_id, None)
            self._save_type(entry.type)
            return True

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def list_all(self, mem_type: MemoryType | None = None, session_key: str | None = None) -> list[MemoryEntry]:
        entries = self._filtered_entries(mem_type, session_key)
        return sorted(entries, key=lambda entry: entry.updated_at or "", reverse=True)

    # ── Search ───────────────────────────────────────────────────────────────

    def search_keyword(
        self,
        query: str,
        mem_type: MemoryType | None = None,
        limit: int = 20,
        session_key: str | None = None,
    ) -> list[MemoryEntry]:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results: list[MemoryEntry] = []
        for entry in self._filtered_entries(mem_type, session_key):
            matched = (
                pattern.search(entry.content)
                or pattern.search(entry.key)
                or any(pattern.search(tag) for tag in entry.tags)
            )
            if matched:
                entry.touch()
                results.append(entry)
        results.sort(key=lambda entry: self._forgetting.effective_importance(entry), reverse=True)
        return results[:limit]

    def search_scored(
        self,
        query: str,
        mem_type: MemoryType | None = None,
        limit: int = 10,
        session_key: str | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Multi-keyword scored search. Returns (entry, score) pairs sorted by score."""
        if self._retriever is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    pass  # Can't await in sync context, fall through to keyword search
                else:
                    results = loop.run_until_complete(
                        self._retriever.retrieve(query, limit=limit, session_key=session_key or "", mem_type=mem_type)
                    )
                    return results
            except Exception as e:
                logger.debug("Vector retrieval unavailable, falling back to keyword search: {}", e)

        words = [
            word.lower() for word in re.findall(r"\w+", query)
            if len(word) > 1 and word.lower() not in _STOP_WORDS
        ]
        if not words:
            return [
                (entry, self._forgetting.effective_importance(entry))
                for entry in self.search_keyword(query, mem_type, limit, session_key=session_key)
            ]

        scored: list[tuple[MemoryEntry, float]] = []
        for entry in self._filtered_entries(mem_type, session_key):
            haystack = f"{entry.key} {entry.content} {' '.join(entry.tags)}".lower()
            word_hits = sum(1 for word in words if word in haystack)
            if word_hits == 0:
                continue
            coverage = word_hits / len(words)
            eff_imp = self._forgetting.effective_importance(entry)
            score = coverage * 0.7 + eff_imp * 0.3
            entry.touch()
            scored.append((entry, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def find_by_key(
        self,
        key: str,
        mem_type: MemoryType | None = None,
        session_key: str | None = None,
    ) -> MemoryEntry | None:
        normalized_key = key.strip()
        if not normalized_key:
            return None
        for entry in self._filtered_entries(mem_type, session_key):
            if entry.key == normalized_key:
                return entry
        return None

    def find_by_content(
        self,
        substring: str,
        mem_type: MemoryType | None = None,
        session_key: str | None = None,
    ) -> MemoryEntry | None:
        matches = self.find_by_content_matches(substring, mem_type=mem_type, limit=1, session_key=session_key)
        return matches[0] if matches else None

    def find_by_content_matches(
        self,
        substring: str,
        mem_type: MemoryType | None = None,
        limit: int | None = 10,
        session_key: str | None = None,
    ) -> list[MemoryEntry]:
        normalized = substring.strip()
        if not normalized:
            return []
        results: list[MemoryEntry] = []
        for entry in self._filtered_entries(mem_type, session_key):
            if normalized in entry.content or normalized in entry.key:
                results.append(entry)
                if limit is not None and len(results) >= limit:
                    break
        return results

    def search_by_time(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        mem_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        results: list[MemoryEntry] = []
        for entry in self._entries.values():
            if mem_type and entry.type != mem_type:
                continue
            try:
                ts = datetime.fromisoformat(entry.updated_at)
            except ValueError:
                continue
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            results.append(entry)
        return sorted(results, key=lambda entry: entry.updated_at or "", reverse=True)

    # ── Context injection ────────────────────────────────────────────────────

    def get_context(
        self,
        mem_type: MemoryType | None = None,
        max_entries: int = 50,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> str:
        entries = sorted(
            self.list_all(mem_type, session_key=session_key)[:max_entries],
            key=lambda entry: self._forgetting.effective_importance(entry),
            reverse=True,
        )
        if not entries:
            return ""

        lines: list[str] = []
        used = 0
        for entry in entries:
            tags = f" [{', '.join(entry.tags)}]" if entry.tags else ""
            line = f"- **{entry.key}**{tags}: {entry.content}"
            delta = len(line) + (1 if lines else 0)
            if max_chars is not None and used + delta > max_chars:
                if not lines and max_chars > 3:
                    truncated = line[: max_chars - 3].rstrip()
                    if truncated:
                        lines.append(truncated + "...")
                break
            lines.append(line)
            used += delta
        return "\n".join(lines)

    def get_snapshot(self, session_key: str | None = None) -> str:
        """构建有界记忆快照，用于注入系统提示。包含长期记忆、用户记忆和环境记忆三部分。"""
        parts: list[str] = []
        long_term = self.read_long_term()
        if long_term:
            parts.append(f"## Long-term Memory\n\n{long_term}")

        user_ctx = self.get_context(
            MemoryType.USER,
            max_entries=30,
            max_chars=self._user_snapshot_char_limit,
            session_key=session_key,
        )
        if user_ctx:
            parts.append(f"## Session User Memory\n\n{user_ctx}")

        env_ctx = self.get_context(
            MemoryType.ENVIRONMENT,
            max_entries=30,
            max_chars=self._env_snapshot_char_limit,
        )
        if env_ctx:
            parts.append(f"## Environment Memory\n\n{env_ctx}")

        return "\n\n".join(parts)

    # ── Long-term memory file (MEMORY.md) ────────────────────────────────────

    def read_long_term(self) -> str:
        if self._long_term_file.exists():
            return self._long_term_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        normalized = content.strip()
        if normalized:
            self._validate_content(normalized, field_name="memory_update")
        with self._file_lock(self._long_term_file):
            _atomic_write_text(self._long_term_file, normalized)

    def append_history(self, entry: str) -> None:
        with self._file_lock(self._history_file):
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, "a", encoding="utf-8") as handle:
                handle.write(entry.rstrip() + "\n\n")

    def search_history(self, query: str, limit: int = 20) -> list[str]:
        if not self._history_file.exists():
            return []
        content = self._history_file.read_text(encoding="utf-8")
        entries = content.split("\n\n")
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        return [entry.strip() for entry in entries if entry.strip() and pattern.search(entry)][:limit]
