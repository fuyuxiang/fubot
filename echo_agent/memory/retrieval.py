"""Hybrid retrieval — BM25 + vector + Resonance Scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Awaitable, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from echo_agent.memory.vectors import VectorIndex

from echo_agent.memory.types import MemoryEntry, MemoryType
from echo_agent.memory.forgetting import ForgettingCurve


_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "it", "its",
    "this", "that", "and", "or", "but", "not", "no", "if", "so", "than",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HybridRetriever:
    """Multi-signal retrieval with query-adaptive Resonance Scoring.

    Weights adapt per-query based on query entropy (Shannon entropy of token distribution).
    - High entropy (vague/broad query) -> upweight vector similarity
    - Low entropy (specific query) -> upweight keyword match
    """

    def __init__(
        self,
        entries_fn: Callable[[], list[MemoryEntry]],
        vector_index: VectorIndex | None = None,
        forgetting: ForgettingCurve | None = None,
        embed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
    ):
        self._entries_fn = entries_fn
        self._vector_index = vector_index
        self._forgetting = forgetting or ForgettingCurve()
        self._embed_fn = embed_fn

    async def retrieve(
        self, query: str, limit: int = 10,
        session_key: str = "", mem_type: MemoryType | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """混合检索管线：BM25 + 向量相似度 + 知识图谱扩展 + 共振评分。

        Args:
            query: 检索查询文本
            limit: 返回结果数量上限
            session_key: 会话标识，用于记忆可见性过滤
            mem_type: 可选的记忆类型过滤
        Returns:
            按共振评分降序排列的 (记忆条目, 分数) 列表
        """
        entries = self._entries_fn()
        if mem_type is not None:
            entries = [e for e in entries if e.type == mem_type]
        if session_key:
            entries = [e for e in entries if not e.source_session or e.source_session == session_key]
        if not entries:
            return []

        entry_map = {e.id: e for e in entries}
        entropy = self._query_entropy(query)
        pool = limit * 3

        keyword_scores: dict[str, float] = dict(self._bm25_search(query, entries, pool))
        vector_scores: dict[str, float] = {}
        if self._vector_index and self._embed_fn:
            vector_scores = {eid: sc for eid, sc in await self._vector_search(query, pool) if eid in entry_map}

        all_ids = set(keyword_scores) | set(vector_scores)
        kw_max = max(keyword_scores.values(), default=1.0) or 1.0
        vec_max = max(vector_scores.values(), default=1.0) or 1.0
        scored: list[tuple[MemoryEntry, float]] = []
        for memory_id in all_ids:
            entry = entry_map.get(memory_id)
            if entry is None or entry.is_superseded:
                continue
            score = self._resonance_score(
                entry, keyword_scores.get(memory_id, 0.0) / kw_max,
                vector_scores.get(memory_id, 0.0) / vec_max, entropy,
            )
            if score > 0:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        logger.debug("hybrid retrieve: {} candidates, entropy={:.3f}", len(scored), entropy)
        return scored[:limit]

    # -- tokenizer -----------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]

    # -- BM25 ----------------------------------------------------------------

    def _bm25_search(self, query: str, entries: list[MemoryEntry], limit: int) -> list[tuple[str, float]]:
        k1, b = 1.5, 0.75
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        doc_tokens = [self._tokenize(f"{e.key} {e.content} {' '.join(e.tags)}") for e in entries]
        n = len(entries)
        avgdl = sum(len(d) for d in doc_tokens) / max(n, 1)
        df: Counter[str] = Counter()
        for dt in doc_tokens:
            seen = set(dt)
            for qt in q_tokens:
                if qt in seen:
                    df[qt] += 1
        results: list[tuple[str, float]] = []
        for i, e in enumerate(entries):
            dl = len(doc_tokens[i])
            freq = Counter(doc_tokens[i])
            score = 0.0
            for qt in q_tokens:
                if qt not in freq:
                    continue
                idf = math.log((n - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
                tf = (freq[qt] * (k1 + 1)) / (freq[qt] + k1 * (1 - b + b * dl / max(avgdl, 1e-9)))
                score += idf * tf
            if score > 0:
                results.append((e.id, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # -- vector search -------------------------------------------------------

    async def _vector_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        if not self._embed_fn or not self._vector_index:
            return []
        try:
            embedding = await self._embed_fn(query)
        except Exception:
            logger.warning("embed_fn failed for vector search")
            return []
        return await self._vector_index.search(embedding, limit)

    # -- query entropy -------------------------------------------------------

    @staticmethod
    def _query_entropy(query: str) -> float:
        """Shannon entropy of token freq distribution, normalized to [0, 1]."""
        tokens = [t for t in _TOKEN_RE.findall(query.lower()) if t not in _STOP_WORDS]
        if not tokens:
            return 0.5
        freq = Counter(tokens)
        total = len(tokens)
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
        max_ent = math.log2(len(freq)) if len(freq) > 1 else 1.0
        return min(entropy / max_ent, 1.0) if max_ent > 0 else 0.0

    # -- resonance scoring ---------------------------------------------------

    def _resonance_score(
        self, entry: MemoryEntry, keyword_score: float,
        vector_score: float, query_entropy: float,
    ) -> float:
        """Query-adaptive Resonance Scoring — weights shift with entropy."""
        e = max(0.0, min(1.0, query_entropy))
        w_keyword = 0.5 * (1 - e)
        w_vector = 0.5 + 0.5 * e
        raw = w_keyword * keyword_score + w_vector * vector_score
        return raw * self._forgetting.effective_importance(entry)

