"""SQLite storage backend — async implementation with error recovery."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
from loguru import logger

from echo_agent.storage.backend import StorageBackend

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sessions (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    key TEXT,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    workflow_id TEXT,
    status TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_workflow ON tasks(workflow_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    name TEXT,
    status TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_trace ON logs(trace_id);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    checksum TEXT,
    size INTEGER,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    embedding BLOB,
    metadata TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""

_MIGRATIONS: list[tuple[int, str]] = [
    (1, "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at)"),
    (2, "CREATE INDEX IF NOT EXISTS idx_logs_created ON logs(created_at)"),
    (3, "CREATE INDEX IF NOT EXISTS idx_vectors_source ON vectors(source_id)"),
    (4, """CREATE TABLE IF NOT EXISTS memory_episodes (
        id TEXT PRIMARY KEY,
        session_key TEXT NOT NULL,
        summary TEXT NOT NULL,
        message_range_start INTEGER NOT NULL DEFAULT 0,
        message_range_end INTEGER NOT NULL DEFAULT 0,
        entities TEXT DEFAULT '[]',
        importance REAL DEFAULT 0.5,
        created_at TEXT NOT NULL
    )"""),
    (5, "CREATE INDEX IF NOT EXISTS idx_episodes_session ON memory_episodes(session_key)"),
    (6, """CREATE TABLE IF NOT EXISTS memory_graph_nodes (
        id TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        node_type TEXT NOT NULL DEFAULT 'concept',
        properties TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )"""),
    (7, "CREATE INDEX IF NOT EXISTS idx_graph_nodes_label ON memory_graph_nodes(label COLLATE NOCASE)"),
    (8, """CREATE TABLE IF NOT EXISTS memory_graph_edges (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        relation TEXT NOT NULL,
        weight REAL DEFAULT 1.0,
        valid_from TEXT,
        valid_to TEXT,
        source_memory_id TEXT,
        created_at TEXT NOT NULL
    )"""),
    (9, "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON memory_graph_edges(source_id)"),
    (10, "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON memory_graph_edges(target_id)"),
    (11, """CREATE TABLE IF NOT EXISTS memory_contradictions (
        id TEXT PRIMARY KEY,
        memory_id_a TEXT NOT NULL,
        memory_id_b TEXT NOT NULL,
        description TEXT NOT NULL,
        resolution TEXT,
        resolved_at TEXT,
        created_at TEXT NOT NULL
    )"""),
    (12, """CREATE TABLE IF NOT EXISTS memory_access_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id TEXT NOT NULL,
        accessed_at TEXT NOT NULL,
        context_query TEXT
    )"""),
    (13, "CREATE INDEX IF NOT EXISTS idx_access_log_memory ON memory_access_log(memory_id)"),
]


class SQLiteBackend(StorageBackend):
    """Async SQLite storage with error recovery and auto-reconnect."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._connect()
        logger.info("SQLite storage initialized at {}", self._db_path)

    async def _connect(self) -> None:
        self._db = await aiosqlite.connect(str(self._db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.executescript(_SCHEMA_SQL)
        await self._run_migrations()

    async def _ensure_connection(self) -> aiosqlite.Connection:
        if self._db is None:
            await self._connect()
        try:
            await self._db.execute("SELECT 1")  # type: ignore[union-attr]
        except Exception:
            logger.warning("SQLite connection lost, reconnecting")
            try:
                await self._db.close()  # type: ignore[union-attr]
            except Exception:
                pass
            await self._connect()
        return self._db  # type: ignore[return-value]

    async def _run_migrations(self) -> None:
        db = self._db
        assert db
        rows = await db.execute_fetchall("SELECT version FROM schema_migrations")
        applied = {row[0] for row in rows}
        now = datetime.now().isoformat()
        for version, sql in _MIGRATIONS:
            if version in applied:
                continue
            await db.execute(sql)
            await db.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, now),
            )
        await db.commit()

    async def close(self) -> None:
        if self._db:
            try:
                await self._db.close()
            except Exception:
                pass
            self._db = None

    # ── Session ────────────────────────────────────────────────────────────

    async def store_session(self, key: str, data: dict[str, Any]) -> None:
        db = await self._ensure_connection()
        now = datetime.now().isoformat()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO sessions (key, data, created_at, updated_at) "
                "VALUES (?, ?, COALESCE((SELECT created_at FROM sessions WHERE key=?), ?), ?)",
                (key, json.dumps(data, ensure_ascii=False), key, now, now),
            )
            await db.commit()
        except Exception as e:
            logger.error("Failed to store session '{}': {}", key, e)
            raise

    async def load_session(self, key: str) -> dict[str, Any] | None:
        db = await self._ensure_connection()
        try:
            row = await db.execute_fetchall("SELECT data FROM sessions WHERE key=?", (key,))
            return json.loads(row[0][0]) if row else None
        except Exception as e:
            logger.error("Failed to load session '{}': {}", key, e)
            return None

    async def delete_session(self, key: str) -> bool:
        db = await self._ensure_connection()
        try:
            cursor = await db.execute("DELETE FROM sessions WHERE key=?", (key,))
            await db.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error("Failed to delete session '{}': {}", key, e)
            return False

    # ── Memory ─────────────────────────────────────────────────────────────

    async def store_memory(self, entry_id: str, data: dict[str, Any]) -> None:
        db = await self._ensure_connection()
        now = datetime.now().isoformat()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO memories (id, type, key, data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM memories WHERE id=?), ?), ?)",
                (entry_id, data.get("type", "user"), data.get("key", ""),
                 json.dumps(data, ensure_ascii=False), entry_id, now, now),
            )
            await db.commit()
        except Exception as e:
            logger.error("Failed to store memory '{}': {}", entry_id, e)
            raise

    async def load_memories(self, mem_type: str | None = None) -> list[dict[str, Any]]:
        db = await self._ensure_connection()
        try:
            if mem_type:
                rows = await db.execute_fetchall(
                    "SELECT data FROM memories WHERE type=? ORDER BY updated_at DESC", (mem_type,),
                )
            else:
                rows = await db.execute_fetchall("SELECT data FROM memories ORDER BY updated_at DESC")
            return [json.loads(r[0]) for r in rows]
        except Exception as e:
            logger.error("Failed to load memories: {}", e)
            return []

    async def delete_memory(self, entry_id: str) -> bool:
        db = await self._ensure_connection()
        try:
            cursor = await db.execute("DELETE FROM memories WHERE id=?", (entry_id,))
            await db.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error("Failed to delete memory '{}': {}", entry_id, e)
            return False

    # ── Task ───────────────────────────────────────────────────────────────

    async def store_task(self, task_id: str, data: dict[str, Any]) -> None:
        db = await self._ensure_connection()
        now = datetime.now().isoformat()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO tasks (id, workflow_id, status, data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM tasks WHERE id=?), ?), ?)",
                (task_id, data.get("workflow_id", ""), data.get("status", "pending"),
                 json.dumps(data, ensure_ascii=False), task_id, now, now),
            )
            await db.commit()
        except Exception as e:
            logger.error("Failed to store task '{}': {}", task_id, e)
            raise

    async def load_task(self, task_id: str) -> dict[str, Any] | None:
        db = await self._ensure_connection()
        try:
            row = await db.execute_fetchall("SELECT data FROM tasks WHERE id=?", (task_id,))
            return json.loads(row[0][0]) if row else None
        except Exception as e:
            logger.error("Failed to load task '{}': {}", task_id, e)
            return None

    async def list_tasks(self, workflow_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        db = await self._ensure_connection()
        try:
            clauses: list[str] = []
            params: list[str] = []
            if workflow_id:
                clauses.append("workflow_id=?")
                params.append(workflow_id)
            if status:
                clauses.append("status=?")
                params.append(status)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = await db.execute_fetchall(
                f"SELECT data FROM tasks{where} ORDER BY updated_at DESC", params,
            )
            return [json.loads(r[0]) for r in rows]
        except Exception as e:
            logger.error("Failed to list tasks: {}", e)
            return []

    # ── Workflow ────────────────────────────────────────────────────────────

    async def store_workflow(self, workflow_id: str, data: dict[str, Any]) -> None:
        db = await self._ensure_connection()
        now = datetime.now().isoformat()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO workflows (id, name, status, data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM workflows WHERE id=?), ?), ?)",
                (workflow_id, data.get("name", ""), data.get("status", "pending"),
                 json.dumps(data, ensure_ascii=False), workflow_id, now, now),
            )
            await db.commit()
        except Exception as e:
            logger.error("Failed to store workflow '{}': {}", workflow_id, e)
            raise

    async def load_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        db = await self._ensure_connection()
        try:
            row = await db.execute_fetchall("SELECT data FROM workflows WHERE id=?", (workflow_id,))
            return json.loads(row[0][0]) if row else None
        except Exception as e:
            logger.error("Failed to load workflow '{}': {}", workflow_id, e)
            return None

    async def list_workflows(self, status: str | None = None) -> list[dict[str, Any]]:
        db = await self._ensure_connection()
        try:
            if status:
                rows = await db.execute_fetchall(
                    "SELECT data FROM workflows WHERE status=? ORDER BY updated_at DESC", (status,),
                )
            else:
                rows = await db.execute_fetchall("SELECT data FROM workflows ORDER BY updated_at DESC")
            return [json.loads(r[0]) for r in rows]
        except Exception as e:
            logger.error("Failed to list workflows: {}", e)
            return []

    # ── Log ────────────────────────────────────────────────────────────────

    async def store_log(self, trace_id: str, spans: list[dict[str, Any]]) -> None:
        db = await self._ensure_connection()
        now = datetime.now().isoformat()
        try:
            await db.execute(
                "INSERT INTO logs (trace_id, data, created_at) VALUES (?, ?, ?)",
                (trace_id, json.dumps(spans, ensure_ascii=False), now),
            )
            await db.commit()
        except Exception as e:
            logger.error("Failed to store log for trace '{}': {}", trace_id, e)

    async def query_logs(self, filters: dict[str, Any] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        db = await self._ensure_connection()
        try:
            rows = await db.execute_fetchall(
                "SELECT trace_id, data, created_at FROM logs ORDER BY id DESC LIMIT ?", (limit,),
            )
            return [{"trace_id": r[0], "spans": json.loads(r[1]), "created_at": r[2]} for r in rows]
        except Exception as e:
            logger.error("Failed to query logs: {}", e)
            return []

    # ── File & Vector ──────────────────────────────────────────────────────

    async def store_file_meta(self, path: str, checksum: str, size: int) -> None:
        db = await self._ensure_connection()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO files (path, checksum, size, updated_at) VALUES (?, ?, ?, ?)",
                (path, checksum, size, datetime.now().isoformat()),
            )
            await db.commit()
        except Exception as e:
            logger.error("Failed to store file meta '{}': {}", path, e)

    async def store_vector(self, vec_id: str, source_id: str, embedding: bytes, metadata: dict[str, Any] | None = None) -> None:
        db = await self._ensure_connection()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO vectors (id, source_id, embedding, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (vec_id, source_id, embedding, json.dumps(metadata or {}), datetime.now().isoformat()),
            )
            await db.commit()
        except Exception as e:
            logger.error("Failed to store vector '{}': {}", vec_id, e)

    async def load_vectors_all(self) -> list[dict[str, Any]]:
        db = await self._ensure_connection()
        try:
            rows = await db.execute_fetchall("SELECT id, source_id, embedding, metadata FROM vectors")
            return [{"id": r[0], "source_id": r[1], "embedding": r[2], "metadata": r[3]} for r in rows]
        except Exception as e:
            logger.error("Failed to load vectors: {}", e)
            return []

    async def load_vector_by_source(self, source_id: str) -> dict[str, Any] | None:
        db = await self._ensure_connection()
        try:
            rows = await db.execute_fetchall("SELECT id, source_id, embedding, metadata FROM vectors WHERE source_id=?", (source_id,))
            if rows:
                r = rows[0]
                return {"id": r[0], "source_id": r[1], "embedding": r[2], "metadata": r[3]}
            return None
        except Exception as e:
            logger.error("Failed to load vector for source '{}': {}", source_id, e)
            return None

    async def delete_vector(self, vec_id: str) -> None:
        db = await self._ensure_connection()
        try:
            await db.execute("DELETE FROM vectors WHERE id=?", (vec_id,))
            await db.commit()
        except Exception as e:
            logger.error("Failed to delete vector '{}': {}", vec_id, e)

    async def execute_sql(self, sql: str, params: tuple = ()) -> None:
        db = await self._ensure_connection()
        try:
            await db.execute(sql, params)
            await db.commit()
        except Exception as e:
            logger.error("SQL execute failed: {}", e)
            raise

    async def fetch_sql(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        db = await self._ensure_connection()
        try:
            cursor = await db.execute(sql, params)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = await cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error("SQL fetch failed: {}", e)
            return []
