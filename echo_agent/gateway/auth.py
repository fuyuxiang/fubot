"""Gateway authentication — allowlist and pairing code authorization."""

from __future__ import annotations

import json
import hmac
import secrets
import time
from pathlib import Path
from typing import Any

from loguru import logger

from echo_agent.config.schema import GatewayAuthConfig


class GatewayAuth:

    def __init__(self, config: GatewayAuthConfig, data_dir: Path):
        self._mode = config.mode
        self._allowed = set(config.allowed_users)
        self._admins = set(config.admin_users)
        self._api_tokens = list(config.api_tokens)
        self.token_header = config.token_header
        self._pairing_ttl = config.pairing_ttl_seconds
        self._data_dir = data_dir / "gateway_auth"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._audit_path = self._data_dir / "audit.jsonl"

        self._approved: dict[str, set[str]] = {}
        self._pending_codes: dict[str, dict[str, Any]] = {}
        self._load_approved()
        self._load_pending()

    def is_authorized(self, platform: str, user_id: str) -> bool:
        if self._mode == "open":
            return True

        if self._mode == "allowlist":
            return user_id in self._allowed or f"{platform}:{user_id}" in self._allowed

        if self._mode == "pairing":
            if user_id in self._allowed or f"{platform}:{user_id}" in self._allowed:
                return True
            approved = self._approved.get(platform, set())
            return user_id in approved

        return False

    def authenticate_token(self, token: str) -> bool:
        if not self._api_tokens:
            return True
        if not token:
            return False
        return any(hmac.compare_digest(token, configured) for configured in self._api_tokens)

    def token_from_headers(self, headers: Any) -> str:
        token = headers.get(self.token_header, "")
        if token:
            return token.strip()
        auth = headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""

    def authenticate_headers(self, headers: Any) -> bool:
        return self.authenticate_token(self.token_from_headers(headers))

    def is_admin(self, platform: str, user_id: str, token: str = "") -> bool:
        if token and self._api_tokens and self.authenticate_token(token):
            return True
        return user_id in self._admins or f"{platform}:{user_id}" in self._admins

    def audit(self, action: str, *, platform: str = "", user_id: str = "", ok: bool = True, reason: str = "") -> None:
        record = {
            "ts": time.time(),
            "action": action,
            "platform": platform,
            "user_id": user_id,
            "ok": ok,
            "reason": reason,
        }
        with self._audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate_pairing_code(self, platform: str) -> str:
        code = secrets.token_hex(3).upper()
        self._pending_codes[code] = {
            "platform": platform,
            "created_at": time.time(),
        }
        self._save_pending()
        logger.info("Pairing code generated for {}", platform)
        self.audit("pair_generate", platform=platform)
        return code

    def verify_pairing(self, platform: str, user_id: str, code: str) -> bool:
        code = code.upper().strip()
        entry = self._pending_codes.get(code)
        if entry is None:
            return False

        if time.time() - entry["created_at"] > self._pairing_ttl:
            del self._pending_codes[code]
            self._save_pending()
            return False

        if entry["platform"] != platform:
            return False

        del self._pending_codes[code]
        self._save_pending()

        if platform not in self._approved:
            self._approved[platform] = set()
        self._approved[platform].add(user_id)
        self._save_approved(platform)

        logger.info("User {}:{} paired successfully", platform, user_id)
        self.audit("pair_verify", platform=platform, user_id=user_id)
        return True

    def _load_approved(self) -> None:
        for path in self._data_dir.glob("*_approved.json"):
            platform = path.stem.replace("_approved", "")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._approved[platform] = set(data)
            except Exception as e:
                logger.debug("Failed to load approved list for {}: {}", platform, e)

    def _save_approved(self, platform: str) -> None:
        path = self._data_dir / f"{platform}_approved.json"
        users = sorted(self._approved.get(platform, set()))
        path.write_text(json.dumps(users, indent=2), encoding="utf-8")

    def _load_pending(self) -> None:
        path = self._data_dir / "pending_codes.json"
        if path.exists():
            try:
                self._pending_codes = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.debug("Failed to load pending codes: {}", e)
                self._pending_codes = {}
        now = time.time()
        expired = [k for k, v in self._pending_codes.items()
                   if now - v.get("created_at", 0) > self._pairing_ttl]
        for k in expired:
            del self._pending_codes[k]

    def _save_pending(self) -> None:
        path = self._data_dir / "pending_codes.json"
        path.write_text(json.dumps(self._pending_codes, indent=2), encoding="utf-8")
