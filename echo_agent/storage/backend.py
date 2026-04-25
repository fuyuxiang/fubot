"""Storage layer — abstract backend interface.

Separates: session, memory, task, workflow, file, log, and vector stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @abstractmethod
    async def initialize(self) -> None:
        """Set up storage (create tables, directories, etc.)."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""

    @abstractmethod
    async def store_session(self, key: str, data: dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def load_session(self, key: str) -> dict[str, Any] | None:
        pass

    @abstractmethod
    async def delete_session(self, key: str) -> bool:
        pass

    @abstractmethod
    async def store_memory(self, entry_id: str, data: dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def load_memories(self, mem_type: str | None = None) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_memory(self, entry_id: str) -> bool:
        pass

    @abstractmethod
    async def store_task(self, task_id: str, data: dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def load_task(self, task_id: str) -> dict[str, Any] | None:
        pass

    @abstractmethod
    async def list_tasks(self, workflow_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def store_workflow(self, workflow_id: str, data: dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def load_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        pass

    @abstractmethod
    async def list_workflows(self, status: str | None = None) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def store_log(self, trace_id: str, spans: list[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    async def query_logs(self, filters: dict[str, Any] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def store_file_meta(self, path: str, checksum: str, size: int) -> None:
        pass

    @abstractmethod
    async def store_vector(self, vec_id: str, source_id: str, embedding: bytes, metadata: dict[str, Any] | None = None) -> None:
        pass

    async def load_vectors_all(self) -> list[dict[str, Any]]:
        return []

    async def load_vector_by_source(self, source_id: str) -> dict[str, Any] | None:
        return None

    async def delete_vector(self, vec_id: str) -> None:
        pass

    async def execute_sql(self, sql: str, params: tuple = ()) -> None:
        pass

    async def fetch_sql(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return []
