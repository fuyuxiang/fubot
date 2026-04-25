"""FAISS vector index — embedding storage, similarity search, graceful degradation."""

from __future__ import annotations

import json
import uuid
from typing import Any, TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from echo_agent.storage.backend import StorageBackend

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None
    _HAS_FAISS = False


class VectorIndex:
    """FAISS-backed vector index with SQLite persistence.

    Falls back gracefully when faiss is not installed — all search methods
    return empty results and add() is a no-op.
    """

    def __init__(self, storage: StorageBackend, dimensions: int = 1536):
        self._storage = storage
        self._dimensions = dimensions
        self._index: Any | None = None
        self._id_map: list[str] = []
        self._initialized = False

    @property
    def available(self) -> bool:
        return _HAS_FAISS

    @property
    def count(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    async def initialize(self) -> None:
        if not _HAS_FAISS:
            logger.info("FAISS not installed — vector search disabled")
            return
        self._index = faiss.IndexFlatIP(self._dimensions)
        rows = await self._storage.load_vectors_all()
        if rows:
            embeddings = []
            for row in rows:
                vec_id = row["id"]
                emb = row.get("embedding")
                if emb is not None:
                    arr = np.frombuffer(emb, dtype=np.float32)
                    if arr.shape[0] == self._dimensions:
                        embeddings.append(arr)
                        self._id_map.append(vec_id)
            if embeddings:
                matrix = np.vstack(embeddings).astype(np.float32)
                faiss.normalize_L2(matrix)
                self._index.add(matrix)
        self._initialized = True
        logger.info("Vector index loaded: {} vectors, {} dims", self.count, self._dimensions)

    async def add(self, memory_id: str, embedding: list[float]) -> str:
        vec_id = uuid.uuid4().hex[:12]
        arr = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != self._dimensions:
            logger.warning("Embedding dimension mismatch: {} vs {}", arr.shape[1], self._dimensions)
            return ""
        await self._storage.store_vector(vec_id, memory_id, arr.tobytes(), {})
        if self._index is not None:
            faiss.normalize_L2(arr)
            self._index.add(arr)
            self._id_map.append(vec_id)
        return vec_id

    async def search(self, query_embedding: list[float], limit: int = 10) -> list[tuple[str, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        arr = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != self._dimensions:
            return []
        faiss.normalize_L2(arr)
        k = min(limit, self._index.ntotal)
        scores, indices = self._index.search(arr, k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    async def remove(self, vec_id: str) -> None:
        if vec_id in self._id_map:
            self._id_map = [v for v in self._id_map if v != vec_id]
        await self._storage.delete_vector(vec_id)

    async def rebuild(self) -> None:
        self._id_map.clear()
        if self._index is not None:
            self._index.reset()
        await self.initialize()

    async def get_embedding_by_memory_id(self, memory_id: str) -> list[float] | None:
        row = await self._storage.load_vector_by_source(memory_id)
        if row and row.get("embedding"):
            arr = np.frombuffer(row["embedding"], dtype=np.float32)
            return arr.tolist()
        return None
