"""SQLite-backed in-process knowledge graph — entity/relation CRUD, BFS, PageRank, LLM extraction."""
from __future__ import annotations

import json, uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger
from echo_agent.memory.types import GraphEdge, GraphNode

if TYPE_CHECKING:
    from echo_agent.storage.backend import StorageBackend

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY, label TEXT NOT NULL, label_lower TEXT NOT NULL,
    node_type TEXT NOT NULL DEFAULT 'concept', properties TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS idx_gn_lower ON graph_nodes(label_lower);
CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
    relation TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
    valid_from TEXT, valid_to TEXT, source_memory_id TEXT, created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
    FOREIGN KEY (target_id) REFERENCES graph_nodes(id));
CREATE INDEX IF NOT EXISTS idx_ge_src ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_ge_tgt ON graph_edges(target_id);"""

_EXTRACT_TOOL = {"type": "function", "function": {
    "name": "extract_entities",
    "description": "Extract named entities and their relations from text.",
    "parameters": {"type": "object", "required": ["entities", "relations"], "properties": {
        "entities": {"type": "array", "description": "Entities found in the text.", "items": {
            "type": "object", "required": ["name", "type"],
            "properties": {"name": {"type": "string"}, "type": {"type": "string"}}}},
        "relations": {"type": "array", "description": "(subject, relation, object) triples.", "items": {
            "type": "object", "required": ["subject", "relation", "object"],
            "properties": {"subject": {"type": "string"}, "relation": {"type": "string"},
                           "object": {"type": "string"}}}}}}}}

_EXTRACT_PROMPT = (
    "Extract all named entities (people, places, organisations, concepts, technologies) "
    "and the relations between them from the following text. "
    "Return them using the extract_entities tool.\n\n{text}")

# ---------------------------------------------------------------------------


class MemoryGraph:
    """SQLite-backed knowledge graph with entity/relation CRUD, PageRank, and LLM entity extraction."""

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage
        self._init = False

    async def _ensure(self) -> None:
        if self._init:
            return
        for s in _SCHEMA.strip().split(";"):
            s = s.strip()
            if s:
                await self._storage.execute_sql(s, [])
        self._init = True

    # -- node CRUD -----------------------------------------------------------

    async def add_node(self, label: str, node_type: str = "concept",
                       properties: dict | None = None) -> GraphNode:
        """Add or get existing node by label (case-insensitive dedup)."""
        await self._ensure()
        existing = await self.get_node_by_label(label)
        if existing is not None:
            return existing
        now = datetime.now().isoformat()
        node = GraphNode(id=uuid.uuid4().hex[:12], label=label, node_type=node_type,
                         properties=properties or {}, created_at=now, updated_at=now)
        await self._storage.execute_sql(
            "INSERT INTO graph_nodes (id,label,label_lower,node_type,properties,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            [node.id, node.label, label.lower(), node.node_type,
             json.dumps(node.properties), node.created_at, node.updated_at])
        logger.debug("added graph node {}", node.label)
        return node

    async def get_node_by_label(self, label: str) -> GraphNode | None:
        """Case-insensitive lookup."""
        await self._ensure()
        rows = await self._storage.fetch_sql(
            "SELECT * FROM graph_nodes WHERE label_lower = ?", [label.lower()])
        return self._to_node(rows[0]) if rows else None

    async def get_all_nodes(self) -> list[GraphNode]:
        """Load all nodes."""
        await self._ensure()
        return [self._to_node(r) for r in await self._storage.fetch_sql("SELECT * FROM graph_nodes", [])]

    async def find_nodes(self, query: str, limit: int = 10) -> list[GraphNode]:
        """Substring search on node labels."""
        await self._ensure()
        rows = await self._storage.fetch_sql(
            "SELECT * FROM graph_nodes WHERE label_lower LIKE ? LIMIT ?",
            [f"%{query.lower()}%", limit])
        return [self._to_node(r) for r in rows]

    # -- edge CRUD -----------------------------------------------------------

    async def add_edge(self, source_label: str, target_label: str, relation: str,
                       weight: float = 1.0, valid_from: str | None = None,
                       source_memory_id: str | None = None) -> GraphEdge:
        """Add edge between nodes (creates nodes if needed)."""
        await self._ensure()
        src = await self.add_node(source_label)
        tgt = await self.add_node(target_label)
        edge = GraphEdge(id=uuid.uuid4().hex[:12], source_id=src.id, target_id=tgt.id,
                         relation=relation, weight=weight, valid_from=valid_from,
                         source_memory_id=source_memory_id, created_at=datetime.now().isoformat())
        await self._storage.execute_sql(
            "INSERT INTO graph_edges (id,source_id,target_id,relation,weight,"
            "valid_from,valid_to,source_memory_id,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            [edge.id, edge.source_id, edge.target_id, edge.relation, edge.weight,
             edge.valid_from, edge.valid_to, edge.source_memory_id, edge.created_at])
        logger.debug("added edge {} -[{}]-> {}", source_label, relation, target_label)
        return edge

    async def get_all_edges(self) -> list[GraphEdge]:
        """Load all edges."""
        await self._ensure()
        return [self._to_edge(r) for r in await self._storage.fetch_sql("SELECT * FROM graph_edges", [])]

    # -- traversal -----------------------------------------------------------

    async def get_neighbors(self, label: str, max_depth: int = 2) -> list[tuple[GraphNode, GraphEdge, int]]:
        """BFS traversal returning (node, edge, depth) tuples."""
        await self._ensure()
        start = await self.get_node_by_label(label)
        if start is None:
            return []
        visited: set[str] = {start.id}
        queue: deque[tuple[str, int]] = deque([(start.id, 0)])
        results: list[tuple[GraphNode, GraphEdge, int]] = []
        while queue:
            current_node_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            rows = await self._storage.fetch_sql(
                "SELECT e.*, n.id AS n_id, n.label, n.label_lower, n.node_type, "
                "n.properties, n.created_at AS n_created, n.updated_at AS n_updated "
                "FROM graph_edges e JOIN graph_nodes n ON n.id = e.target_id WHERE e.source_id = ? "
                "UNION ALL "
                "SELECT e.*, n.id AS n_id, n.label, n.label_lower, n.node_type, "
                "n.properties, n.created_at AS n_created, n.updated_at AS n_updated "
                "FROM graph_edges e JOIN graph_nodes n ON n.id = e.source_id WHERE e.target_id = ?",
                [current_node_id, current_node_id])
            for row in rows:
                neighbor_id = row["n_id"]
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                node = self._to_node({"id": neighbor_id, "label": row["label"], "label_lower": row["label_lower"],
                                      "node_type": row["node_type"], "properties": row["properties"],
                                      "created_at": row["n_created"], "updated_at": row["n_updated"]})
                results.append((node, self._to_edge(row), depth + 1))
                queue.append((neighbor_id, depth + 1))
        return results

    # -- subgraph ------------------------------------------------------------

    async def get_subgraph(self, node_ids: list[str]) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Get nodes and edges for a set of node IDs."""
        await self._ensure()
        if not node_ids:
            return [], []
        ph = ",".join("?" for _ in node_ids)
        nodes = [self._to_node(r) for r in await self._storage.fetch_sql(
            f"SELECT * FROM graph_nodes WHERE id IN ({ph})", node_ids)]
        edges = [self._to_edge(r) for r in await self._storage.fetch_sql(
            f"SELECT * FROM graph_edges WHERE source_id IN ({ph}) AND target_id IN ({ph})",
            node_ids + node_ids)]
        return nodes, edges

    # -- PageRank ------------------------------------------------------------

    async def pagerank(self, iterations: int = 20, damping: float = 0.85) -> dict[str, float]:
        """计算知识图谱中所有节点的 PageRank 分数。

        Args:
            iterations: 迭代次数，越多越精确
            damping: 阻尼系数，通常为 0.85
        Returns:
            节点 ID 到 PageRank 分数的映射
        """
        await self._ensure()
        nodes = await self.get_all_nodes()
        edges = await self.get_all_edges()
        if not nodes:
            return {}
        n = len(nodes)
        scores: dict[str, float] = {node.id: 1.0 / n for node in nodes}
        out_deg: dict[str, float] = defaultdict(float)
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for e in edges:
            out_deg[e.source_id] += e.weight
            adj[e.target_id].append((e.source_id, e.weight))
        for _ in range(iterations):
            new: dict[str, float] = {}
            for node in nodes:
                rank_sum = sum(scores[src] * w / out_deg[src] for src, w in adj[node.id] if out_deg[src] > 0)
                new[node.id] = (1 - damping) / n + damping * rank_sum
            scores = new
        return scores

    # -- LLM entity extraction ------------------------------------------------

    async def extract_entities_and_relations(
            self, text: str, llm_call: Callable[..., Any]) -> list[tuple[str, str, str]]:
        """Use LLM to extract (subject, relation, object) triples from text."""
        try:
            resp = await llm_call(
                messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(text=text)}],
                tools=[_EXTRACT_TOOL],
                tool_choice={"type": "function", "function": {"name": "extract_entities"}})
        except Exception:
            logger.exception("LLM entity extraction failed")
            return []
        triples: list[tuple[str, str, str]] = []
        for tc in getattr(resp, "tool_calls", None) or []:
            args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
            for ent in args.get("entities", []):
                await self.add_node(ent["name"], ent.get("type", "concept"))
            for rel in args.get("relations", []):
                subj, relation, obj = rel["subject"], rel["relation"], rel["object"]
                await self.add_edge(subj, obj, relation)
                triples.append((subj, relation, obj))
        logger.info("extracted {} triples from text", len(triples))
        return triples

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _to_node(row: dict[str, Any]) -> GraphNode:
        props = row.get("properties", "{}")
        if isinstance(props, str):
            props = json.loads(props)
        return GraphNode(id=row["id"], label=row["label"],
                         node_type=row.get("node_type", "concept"), properties=props,
                         created_at=row.get("created_at", ""), updated_at=row.get("updated_at", ""))

    @staticmethod
    def _to_edge(row: dict[str, Any]) -> GraphEdge:
        return GraphEdge(id=row["id"], source_id=row["source_id"], target_id=row["target_id"],
                         relation=row["relation"], weight=row.get("weight", 1.0),
                         valid_from=row.get("valid_from"), valid_to=row.get("valid_to"),
                         source_memory_id=row.get("source_memory_id"),
                         created_at=row.get("created_at", ""))