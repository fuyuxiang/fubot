"""Session reset policies — daily, idle, both, or none."""

from __future__ import annotations

from datetime import datetime, timedelta

from echo_agent.config.schema import GatewaySessionPolicyConfig
from echo_agent.session.manager import Session, SessionManager


class SessionResetPolicy:

    def __init__(self, config: GatewaySessionPolicyConfig):
        self._mode = config.mode
        self._daily_hour = config.daily_reset_hour
        self._idle_delta = timedelta(minutes=config.idle_timeout_minutes)

    def should_reset(self, session: Session) -> bool:
        if self._mode == "none":
            return False

        now = datetime.now()

        if self._mode in ("idle", "both"):
            if (now - session.updated_at) >= self._idle_delta:
                return True

        if self._mode in ("daily", "both"):
            if self._crossed_daily_boundary(session.updated_at, now):
                return True

        return False

    async def reset(self, session: Session, manager: SessionManager) -> None:
        session.clear()
        session.metadata["last_reset_at"] = datetime.now().isoformat()
        session.metadata["reset_count"] = session.metadata.get("reset_count", 0) + 1
        await manager.save(session)

    def _crossed_daily_boundary(self, last_active: datetime, now: datetime) -> bool:
        reset_today = now.replace(
            hour=self._daily_hour, minute=0, second=0, microsecond=0,
        )
        if now.hour < self._daily_hour:
            reset_today -= timedelta(days=1)

        return last_active < reset_today <= now
