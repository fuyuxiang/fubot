"""Evaluation dataset — test case loading from YAML/JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalCase:
    id: str = ""
    input: str = ""
    expected_tools: list[str] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    expected_output: str = ""
    max_iterations: int = 10
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        return cls(
            id=data.get("id", ""),
            input=data.get("input", ""),
            expected_tools=data.get("expected_tools", []),
            expected_contains=data.get("expected_contains", []),
            expected_output=data.get("expected_output", ""),
            max_iterations=data.get("max_iterations", 10),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


class EvalDataset:
    def __init__(self, cases: list[EvalCase] | None = None):
        self.cases = cases or []

    @classmethod
    def from_yaml(cls, path: Path) -> EvalDataset:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cases = [EvalCase.from_dict(c) for c in (data if isinstance(data, list) else data.get("cases", []))]
        return cls(cases)

    @classmethod
    def from_json(cls, path: Path) -> EvalDataset:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cases = [EvalCase.from_dict(c) for c in (data if isinstance(data, list) else data.get("cases", []))]
        return cls(cases)

    @classmethod
    def from_path(cls, path: Path) -> EvalDataset:
        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        return cls.from_json(path)

    def filter_by_tag(self, tag: str) -> list[EvalCase]:
        return [c for c in self.cases if tag in c.tags]

    def __len__(self) -> int:
        return len(self.cases)
