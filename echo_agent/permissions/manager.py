"""Permission, approval, and credential management.

Covers:
  - Multi-level permissions (user, channel, tool, file, workspace, admin)
  - Pre-execution approval workflow (approve/deny/whitelist/blacklist)
  - Credential management (agent identity, token storage, isolation, rotation, audit)
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class PermissionLevel(str, Enum):
    USER = "user"
    CHANNEL = "channel"
    TOOL = "tool"
    FILE = "file"
    WORKSPACE = "workspace"
    ADMIN = "admin"


class Effect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class PermissionRule:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    level: PermissionLevel = PermissionLevel.USER
    subject: str = "*"  # user_id, channel, tool_name, file_path, etc.
    action: str = "*"   # read, write, execute, etc.
    effect: Effect = Effect.ALLOW
    scope: str = "*"    # workspace, global, etc.
    priority: int = 0

    def matches(self, level: PermissionLevel, subject: str, action: str) -> bool:
        if self.level != level:
            return False
        if self.subject != "*" and self.subject != subject:
            return False
        if self.action != "*" and self.action != action:
            return False
        return True


class PermissionManager:
    """Evaluates permission rules across all levels."""

    def __init__(self, admin_users: list[str] | None = None):
        self._rules: list[PermissionRule] = []
        self._admin_users = set(admin_users or [])

    def add_rule(self, rule: PermissionRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.id != rule_id]
        return len(self._rules) < before

    def is_admin(self, user_id: str) -> bool:
        return user_id in self._admin_users

    def check(self, level: PermissionLevel, subject: str, action: str, user_id: str = "") -> Effect:
        if user_id and self.is_admin(user_id):
            return Effect.ALLOW
        for rule in self._rules:
            if rule.matches(level, subject, action):
                return rule.effect
        return Effect.ALLOW if not self._rules else Effect.DENY

    def check_tool(self, tool_name: str, user_id: str = "") -> bool:
        return self.check(PermissionLevel.TOOL, tool_name, "execute", user_id) == Effect.ALLOW

    def check_file(self, path: str, action: str, user_id: str = "") -> bool:
        return self.check(PermissionLevel.FILE, path, action, user_id) == Effect.ALLOW

    def check_channel(self, channel: str, user_id: str) -> bool:
        return self.check(PermissionLevel.CHANNEL, channel, "access", user_id) == Effect.ALLOW


@dataclass
class ApprovalRequest:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    action: str = ""
    tool_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    user_id: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    reason: str = ""
    decided_by: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    decided_at: str = ""


class ApprovalManager:
    """Manages pre-execution approval for high-risk actions."""

    def __init__(
        self,
        require_approval: list[str] | None = None,
        auto_approve: list[str] | None = None,
        auto_deny: list[str] | None = None,
        default_policy: str = "ask",
    ):
        self._require = set(require_approval or [])
        self._auto_approve = set(auto_approve or [])
        self._auto_deny = set(auto_deny or [])
        self._default_policy = default_policy
        self._pending: dict[str, ApprovalRequest] = {}
        self._history: list[ApprovalRequest] = []
        self._approved_signatures: set[str] = set()

    def needs_approval(self, action: str) -> bool:
        if action in self._auto_approve:
            return False
        if action in self._auto_deny:
            return True
        if action in self._require:
            return True
        return self._default_policy == "ask"

    def request_approval(self, action: str, tool_name: str = "", params: dict[str, Any] | None = None, user_id: str = "") -> ApprovalRequest:
        params = params or {}
        signature = self._signature(action, tool_name, params, user_id)
        if signature in self._approved_signatures:
            self._approved_signatures.remove(signature)
            req = ApprovalRequest(action=action, tool_name=tool_name, params=params, user_id=user_id, status=ApprovalStatus.APPROVED, reason="approved by prior request")
            self._history.append(req)
            return req
        if action in self._auto_approve:
            req = ApprovalRequest(action=action, tool_name=tool_name, params=params, user_id=user_id, status=ApprovalStatus.APPROVED)
            self._history.append(req)
            return req
        if action in self._auto_deny:
            req = ApprovalRequest(action=action, tool_name=tool_name, params=params, user_id=user_id, status=ApprovalStatus.DENIED, reason="auto-denied by policy")
            self._history.append(req)
            return req
        if action in self._require:
            req = ApprovalRequest(action=action, tool_name=tool_name, params=params, user_id=user_id)
            self._pending[req.id] = req
            return req
        if self._default_policy == "approve":
            req = ApprovalRequest(action=action, tool_name=tool_name, params=params, user_id=user_id, status=ApprovalStatus.APPROVED)
            self._history.append(req)
            return req
        if self._default_policy == "deny":
            req = ApprovalRequest(action=action, tool_name=tool_name, params=params, user_id=user_id, status=ApprovalStatus.DENIED, reason="denied by default policy")
            self._history.append(req)
            return req

        req = ApprovalRequest(action=action, tool_name=tool_name, params=params, user_id=user_id)
        self._pending[req.id] = req
        return req

    def approve(self, request_id: str, decided_by: str = "") -> bool:
        req = self._pending.pop(request_id, None)
        if not req:
            return False
        req.status = ApprovalStatus.APPROVED
        req.decided_by = decided_by
        req.decided_at = datetime.now().isoformat()
        self._approved_signatures.add(self._signature(req.action, req.tool_name, req.params, req.user_id))
        self._history.append(req)
        return True

    def deny(self, request_id: str, reason: str = "", decided_by: str = "") -> bool:
        req = self._pending.pop(request_id, None)
        if not req:
            return False
        req.status = ApprovalStatus.DENIED
        req.reason = reason
        req.decided_by = decided_by
        req.decided_at = datetime.now().isoformat()
        self._history.append(req)
        return True

    def get_pending(self) -> list[ApprovalRequest]:
        return list(self._pending.values())

    def get(self, request_id: str) -> ApprovalRequest | None:
        return self._pending.get(request_id)

    def _signature(self, action: str, tool_name: str, params: dict[str, Any], user_id: str) -> str:
        payload = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(f"{user_id}:{action}:{tool_name}:{payload}".encode()).hexdigest()


@dataclass
class Credential:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    tool_scope: str = "*"
    value_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    rotated_at: str = ""
    _value: str = field(default="", repr=False)


class CredentialManager:
    """Manages agent credentials with isolation per tool and rotation support."""

    def __init__(
        self,
        store_path: Path,
        encryption_key_env: str = "ECHO_AGENT_CREDENTIAL_KEY",
        require_encryption: bool = False,
    ):
        self._store_path = store_path
        self._encryption_key_env = encryption_key_env
        self._require_encryption = require_encryption
        self._credentials: dict[str, Credential] = {}
        self._audit: list[dict[str, Any]] = []
        self._load()

    def _fernet(self) -> Any | None:
        secret = os.environ.get(self._encryption_key_env, "")
        if not secret:
            if self._require_encryption:
                raise RuntimeError(
                    f"Credential encryption required but {self._encryption_key_env} is not set"
                )
            return None
        try:
            import base64
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise RuntimeError("cryptography is required for encrypted credential storage") from exc
        key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
        return Fernet(key)

    def _encode_secret(self, value: str) -> tuple[str, str]:
        fernet = self._fernet()
        if not fernet:
            logger.warning(
                "Credential encryption key not configured; storing credential metadata with plaintext fallback"
            )
            return "plain", value
        return "fernet", fernet.encrypt(value.encode()).decode()

    def _decode_secret(self, item: dict[str, Any]) -> str:
        encoding = item.get("encoding", "plain")
        if encoding == "plain":
            return item.get("_value", "")
        if encoding == "fernet":
            fernet = self._fernet()
            if not fernet:
                raise RuntimeError(
                    f"Credential file is encrypted but {self._encryption_key_env} is not set"
                )
            return fernet.decrypt(item.get("value_enc", "").encode()).decode()
        raise RuntimeError(f"Unsupported credential encoding: {encoding}")

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            for item in data.get("credentials", []):
                value = self._decode_secret(item)
                cred = Credential(
                    id=item["id"], name=item["name"],
                    tool_scope=item.get("tool_scope", "*"),
                    value_hash=item.get("value_hash", ""),
                    created_at=item.get("created_at", ""),
                    rotated_at=item.get("rotated_at", ""),
                    _value=value,
                )
                self._credentials[cred.id] = cred
        except Exception as e:
            logger.warning("Failed to load credentials: {}", e)

    def _save(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        items: list[dict[str, Any]] = []
        for c in self._credentials.values():
            encoding, encoded = self._encode_secret(c._value)
            item = {
                "id": c.id,
                "name": c.name,
                "tool_scope": c.tool_scope,
                "value_hash": c.value_hash,
                "created_at": c.created_at,
                "rotated_at": c.rotated_at,
                "encoding": encoding,
            }
            if encoding == "fernet":
                item["value_enc"] = encoded
            else:
                item["_value"] = encoded
            items.append(item)
        data = {
            "format": "echo-agent-credentials-v2",
            "encryption_key_env": self._encryption_key_env,
            "credentials": items,
        }
        self._store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def store(self, name: str, value: str, tool_scope: str = "*") -> Credential:
        cred = Credential(
            name=name, tool_scope=tool_scope, _value=value,
            value_hash=hashlib.sha256(value.encode()).hexdigest()[:16],
        )
        self._credentials[cred.id] = cred
        self._audit_log("store", cred)
        self._save()
        return cred

    def get(self, name: str, tool_scope: str = "*") -> str | None:
        for cred in self._credentials.values():
            if cred.name == name and (cred.tool_scope == "*" or cred.tool_scope == tool_scope):
                self._audit_log("access", cred)
                return cred._value
        return None

    def rotate(self, name: str, new_value: str) -> bool:
        for cred in self._credentials.values():
            if cred.name == name:
                cred._value = new_value
                cred.value_hash = hashlib.sha256(new_value.encode()).hexdigest()[:16]
                cred.rotated_at = datetime.now().isoformat()
                self._audit_log("rotate", cred)
                self._save()
                return True
        return False

    def delete(self, name: str) -> bool:
        to_remove = [cid for cid, c in self._credentials.items() if c.name == name]
        for cid in to_remove:
            self._audit_log("delete", self._credentials[cid])
            del self._credentials[cid]
        if to_remove:
            self._save()
        return bool(to_remove)

    def get_for_tool(self, tool_name: str) -> dict[str, str]:
        result = {}
        for cred in self._credentials.values():
            if cred.tool_scope == "*" or cred.tool_scope == tool_name:
                result[cred.name] = cred._value
        return result

    def _audit_log(self, action: str, cred: Credential) -> None:
        self._audit.append({
            "action": action, "credential": cred.name,
            "tool_scope": cred.tool_scope,
            "timestamp": datetime.now().isoformat(),
        })

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._audit[-limit:]
