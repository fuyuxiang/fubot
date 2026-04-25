from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchNode:
    id: str
    state: str
    score: float = 0.0
    visits: int = 0
    parent: SearchNode | None = None
    children: list[SearchNode] = field(default_factory=list)
    action: str = ""
    reward: float = 0.0

    def ucb1(self, exploration: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.score / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else 0
        return exploit + explore

    def best_child(self) -> SearchNode | None:
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1())

    def add_child(self, child: SearchNode) -> None:
        child.parent = self
        self.children.append(child)

    def backpropagate(self, reward: float) -> None:
        node: SearchNode | None = self
        while node is not None:
            node.visits += 1
            node.score += reward
            node = node.parent


class SearchTree:
    def __init__(self, root_state: str):
        self.root = SearchNode(id="root", state=root_state)
        self._node_count = 1

    def select(self) -> SearchNode:
        node = self.root
        while node.children:
            best = node.best_child()
            if best is None:
                break
            node = best
        return node

    def expand(self, parent: SearchNode, action: str, state: str, score: float = 0.0) -> SearchNode:
        self._node_count += 1
        child = SearchNode(id=f"n_{self._node_count}", state=state, action=action, score=score)
        parent.add_child(child)
        return child

    @property
    def size(self) -> int:
        return self._node_count

    def best_path(self) -> list[SearchNode]:
        path = []
        node = self.root
        while node.children:
            best = max(node.children, key=lambda c: c.score / max(c.visits, 1))
            path.append(best)
            node = best
        return path
