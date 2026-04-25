"""MCP security — prompt injection scanning and collision guards."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("instruction_override", re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I)),
    ("role_injection", re.compile(r"<\s*(system|assistant|admin)\s*>", re.I)),
    ("prompt_leak", re.compile(r"(reveal|show|print|output)\s+(your|the)\s+(system\s+)?prompt", re.I)),
    ("code_exec_ref", re.compile(r"(exec|eval|subprocess|os\.system|__import__)\s*\(", re.I)),
    ("jailbreak_attempt", re.compile(r"(DAN|do\s+anything\s+now|bypass\s+safety)", re.I)),
    ("hidden_instruction", re.compile(r"(you\s+must|always\s+respond|never\s+refuse)", re.I)),
    ("base64_payload", re.compile(r"base64[.\s]*(decode|encode)", re.I)),
    ("data_exfil", re.compile(r"(send|post|upload)\s+(to|data|content)\s+(http|url|endpoint)", re.I)),
]


def scan_tool_description(description: str) -> list[str]:
    findings: list[str] = []
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(description):
            findings.append(name)
    return findings


def check_tool_collision(
    tool_name: str,
    builtin_names: set[str],
) -> bool:
    if tool_name in builtin_names:
        logger.warning("MCP tool '{}' collides with built-in tool — skipping", tool_name)
        return True
    return False


def validate_mcp_tools(
    server_name: str,
    tools: list[dict[str, Any]],
    builtin_names: set[str],
    include_filter: list[str] | None = None,
    exclude_filter: list[str] | None = None,
    policy: str = "block",
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []

    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        if not name:
            logger.warning("MCP server '{}' exposed a tool without a name — skipping", server_name)
            continue

        if include_filter and name not in include_filter:
            continue
        if exclude_filter and name in exclude_filter:
            continue

        sanitized_name = f"mcp_{server_name}_{name}"
        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", sanitized_name)

        if check_tool_collision(sanitized_name, builtin_names):
            continue

        findings = scan_tool_description(desc)
        if findings:
            logger.warning(
                "MCP tool '{}/{}' description has suspicious patterns: {}",
                server_name, name, ", ".join(findings),
            )
            if policy == "block":
                continue

        accepted.append(tool)

    return accepted
