"""JSONL audit sink for multi-agent dispatch decisions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class DispatchAuditLog:
    def __init__(self, path: Path):
        self._path = path

    def write(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.time(), **record}
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
