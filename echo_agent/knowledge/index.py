"""Local keyword knowledge index for enterprise/internal-doc retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


_LATIN_OR_NUM_RE = re.compile(r"[a-z0-9_]+", re.I)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class KnowledgeSearchResult:
    citation_id: str
    path: str
    title: str
    text: str
    score: float
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_path(workspace: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else workspace / path


def _tokenize(text: str) -> list[str]:
    lower = text.lower()
    tokens = _LATIN_OR_NUM_RE.findall(lower)
    cjk_chars = _CJK_RE.findall(lower)
    tokens.extend(cjk_chars)
    tokens.extend("".join(cjk_chars[i:i + 2]) for i in range(max(0, len(cjk_chars) - 1)))
    return tokens


def _extract_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    raw = text[4:end].strip()
    body = text[end + 4:].lstrip()
    metadata: dict[str, Any] = {}
    current_key = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") and current_key:
            metadata.setdefault(current_key, []).append(stripped[2:].strip().strip("'\""))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        current_key = key.strip()
        value = value.strip()
        if not value:
            metadata[current_key] = []
        elif value.startswith("[") and value.endswith("]"):
            metadata[current_key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
        else:
            metadata[current_key] = value.strip("'\"")
    return metadata, body


class KnowledgeIndex:
    """Persistent local index with deterministic keyword ranking and citations."""

    def __init__(
        self,
        *,
        workspace: Path,
        docs_dir: str,
        index_path: str,
        chunk_size: int = 1200,
        chunk_overlap: int = 120,
        allowed_extensions: list[str] | None = None,
    ):
        self.workspace = workspace
        self.docs_dir = _resolve_path(workspace, docs_dir)
        self.index_path = _resolve_path(workspace, index_path)
        self.chunk_size = max(200, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.allowed_extensions = {ext.lower() for ext in (allowed_extensions or [".md", ".txt"])}
        self._chunks: list[dict[str, Any]] = []
        self._df: Counter[str] = Counter()
        self._loaded = False

    @property
    def chunk_count(self) -> int:
        self._ensure_loaded()
        return len(self._chunks)

    @property
    def doc_count(self) -> int:
        self._ensure_loaded()
        return len({chunk["path"] for chunk in self._chunks})

    def ensure_ready(self, *, auto_index: bool = True) -> None:
        if self.index_path.exists() and not self._is_stale():
            self.load()
            return
        if auto_index:
            self.rebuild()
        elif self.index_path.exists():
            self.load()

    def rebuild(self) -> dict[str, Any]:
        self._chunks = []
        self._df = Counter()
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        files = [
            path for path in self.docs_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.allowed_extensions
        ]
        for path in sorted(files):
            self._index_file(path)
        self._recompute_stats()
        self._save()
        self._loaded = True
        summary = {"documents": len(files), "chunks": len(self._chunks), "index_path": str(self.index_path)}
        logger.info("Knowledge index rebuilt: {} documents, {} chunks", summary["documents"], summary["chunks"])
        return summary

    def load(self) -> None:
        if not self.index_path.exists():
            self._chunks = []
            self._df = Counter()
            self._loaded = True
            return
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        self._chunks = list(data.get("chunks", []))
        self._recompute_stats()
        self._loaded = True

    def search(self, query: str, *, limit: int = 5, user_id: str = "") -> list[KnowledgeSearchResult]:
        self._ensure_loaded()
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)
        total_chunks = max(1, len(self._chunks))
        scored: list[tuple[float, dict[str, Any]]] = []
        query_lower = query.lower()
        for chunk in self._chunks:
            if not self._allowed_for_user(chunk.get("metadata", {}), user_id):
                continue
            terms = Counter(chunk.get("terms", {}))
            if not terms:
                continue
            score = 0.0
            length = max(1, sum(terms.values()))
            for term, query_tf in query_counts.items():
                tf = terms.get(term, 0)
                if tf <= 0:
                    continue
                idf = math.log((1 + total_chunks) / (1 + self._df.get(term, 0))) + 1
                score += (tf / length) * idf * query_tf
            text_lower = chunk.get("text", "").lower()
            if query_lower and query_lower in text_lower:
                score += 1.5
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[KnowledgeSearchResult] = []
        for idx, (score, chunk) in enumerate(scored[:limit], 1):
            results.append(KnowledgeSearchResult(
                citation_id=f"K{idx}",
                path=chunk.get("path", ""),
                title=chunk.get("title", "") or Path(chunk.get("path", "")).name,
                text=chunk.get("text", ""),
                score=round(score, 6),
                chunk_id=chunk.get("id", ""),
                metadata=chunk.get("metadata", {}),
            ))
        return results

    def format_results(self, results: list[KnowledgeSearchResult]) -> str:
        if not results:
            return ""
        lines = [
            "Internal knowledge context. Use these citations when answering claims sourced from internal docs."
        ]
        for result in results:
            excerpt = re.sub(r"\s+", " ", result.text).strip()
            if len(excerpt) > 900:
                excerpt = excerpt[:900] + "..."
            lines.append(
                f"[{result.citation_id}] {result.title} ({result.path})\n"
                f"{excerpt}"
            )
        return "\n\n".join(lines)

    def status(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "docs_dir": str(self.docs_dir),
            "index_path": str(self.index_path),
            "documents": self.doc_count,
            "chunks": self.chunk_count,
            "allowed_extensions": sorted(self.allowed_extensions),
            "stale": self._is_stale(),
        }

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _is_stale(self) -> bool:
        if not self.index_path.exists():
            return True
        index_mtime = self.index_path.stat().st_mtime
        if not self.docs_dir.exists():
            return False
        for path in self.docs_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.allowed_extensions and path.stat().st_mtime > index_mtime:
                return True
        return False

    def _index_file(self, path: Path) -> None:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Failed to read knowledge document {}: {}", path, e)
            return
        metadata, body = _extract_frontmatter(raw)
        rel_path = str(path.relative_to(self.workspace)) if path.is_relative_to(self.workspace) else str(path)
        title = self._title_for(path, body)
        for idx, text in enumerate(self._chunk_text(body)):
            terms = Counter(_tokenize(text))
            self._chunks.append({
                "id": f"{rel_path}#{idx}",
                "path": rel_path,
                "title": title,
                "text": text,
                "terms": dict(terms),
                "metadata": metadata,
                "mtime": path.stat().st_mtime,
            })

    def _chunk_text(self, text: str) -> list[str]:
        clean = text.strip()
        if not clean:
            return []
        if len(clean) <= self.chunk_size:
            return [clean]
        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        while start < len(clean):
            end = min(len(clean), start + self.chunk_size)
            chunks.append(clean[start:end].strip())
            if end >= len(clean):
                break
            start += step
        return [c for c in chunks if c]

    def _title_for(self, path: Path, text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or path.name
            if stripped:
                return stripped[:80]
        return path.name

    def _allowed_for_user(self, metadata: dict[str, Any], user_id: str) -> bool:
        allowed = metadata.get("allowed_users") or metadata.get("allow_users") or metadata.get("users")
        if not allowed:
            return True
        if isinstance(allowed, str):
            allowed_values = {allowed}
        else:
            allowed_values = {str(value) for value in allowed}
        return "*" in allowed_values or user_id in allowed_values

    def _recompute_stats(self) -> None:
        self._df = Counter()
        for chunk in self._chunks:
            self._df.update(set(chunk.get("terms", {}).keys()))

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": "echo-agent-knowledge-v1",
            "generated_at": datetime.now().isoformat(),
            "docs_dir": str(self.docs_dir),
            "chunks": self._chunks,
        }
        self.index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
