"""Base tool class and execution context for the tool framework."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


def _validate_json_schema(schema: Any, path: str = "parameters") -> None:
    if not isinstance(schema, dict):
        return

    schema_type = schema.get("type")
    if schema_type == "array":
        if "items" not in schema:
            raise ValueError(f"Invalid schema at '{path}': array schema missing items")
        items_schema = schema["items"]
        if isinstance(items_schema, list):
            for index, entry in enumerate(items_schema):
                _validate_json_schema(entry, f"{path}.items[{index}]")
        else:
            _validate_json_schema(items_schema, f"{path}.items")

    if schema_type == "object":
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for name, prop_schema in properties.items():
                _validate_json_schema(prop_schema, f"{path}.properties.{name}")
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            _validate_json_schema(additional, f"{path}.additionalProperties")

    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for index, entry in enumerate(variants):
                _validate_json_schema(entry, f"{path}.{key}[{index}]")


class ToolPermission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    ADMIN = "admin"


@dataclass(frozen=True)
class ToolExecutionContext:
    execution_id: str = ""
    trace_id: str = ""
    session_key: str = ""
    user_id: str = ""
    attempt_index: int = 0
    idempotency_key: str = ""
    is_replay: bool = False
    parent_execution_id: str | None = None
    credentials: dict[str, str] = field(default_factory=dict)

    def log_fields(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "trace_id": self.trace_id,
            "attempt_index": self.attempt_index,
            "idempotency_key": self.idempotency_key[:16] if self.idempotency_key else "",
            "is_replay": self.is_replay,
        }


def build_idempotency_key(trace_id: str, tool_name: str, index: int, params: Mapping[str, Any]) -> str:
    payload = json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(f"{trace_id}:{tool_name}:{index}:{payload}".encode()).hexdigest()
    return digest[:24]


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.output if self.success else f"Error: {self.error}"


class Tool(ABC):
    """Abstract base class for all agent tools.

    Subclasses define name, description, parameters schema, required permissions,
    and the execute method.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    required_permissions: list[ToolPermission] = []
    timeout_seconds: int = 30
    max_retries: int = 0
    stream_capable: bool = False

    @abstractmethod
    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        """Execute the tool with given parameters."""

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate parameters against the JSON schema."""
        errors = []
        required = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})
        for key in required:
            if key not in params:
                errors.append(f"missing required parameter: {key}")
        for key, value in params.items():
            if key in properties:
                prop = properties[key]
                expected_type = prop.get("type")
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"{key} must be a string")
                elif expected_type == "integer" and not isinstance(value, int):
                    errors.append(f"{key} must be an integer")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"{key} must be a boolean")
                if "enum" in prop and value not in prop["enum"]:
                    errors.append(f"{key} must be one of {prop['enum']}")
        return errors

    def execution_mode(self, params: dict[str, Any]) -> str:
        """Classify as 'read_only' or 'side_effect' for replay guards."""
        return "side_effect"

    def to_schema(self) -> dict[str, Any]:
        _validate_json_schema(self.parameters)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
