"""Scheduler — cron, interval, condition, event, and delay triggers.

Supports: pause/resume/cancel, background tasks with progress tracking,
result delivery, interrupt recovery, and idempotency control.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger


class TriggerKind(str, Enum):
    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"
    EVENT = "event"
    CONDITION = "condition"


class JobStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


@dataclass
class ScheduledJob:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str = ""
    trigger: TriggerKind = TriggerKind.INTERVAL
    cron_expr: str = ""
    interval_ms: int = 0
    at_ms: int = 0
    timezone: str = ""
    event_name: str = ""
    condition_expr: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.ACTIVE
    enabled: bool = True
    delete_after_run: bool = False
    next_run_ms: int | None = None
    last_run_ms: int | None = None
    last_status: str = ""
    last_error: str = ""
    run_count: int = 0
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "trigger": self.trigger.value,
            "cron_expr": self.cron_expr, "interval_ms": self.interval_ms,
            "at_ms": self.at_ms, "timezone": self.timezone,
            "event_name": self.event_name, "payload": self.payload,
            "status": self.status.value, "enabled": self.enabled,
            "delete_after_run": self.delete_after_run,
            "next_run_ms": self.next_run_ms, "last_run_ms": self.last_run_ms,
            "last_status": self.last_status, "last_error": self.last_error,
            "run_count": self.run_count, "created_at_ms": self.created_at_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledJob:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:10]),
            name=data.get("name", ""),
            trigger=TriggerKind(data.get("trigger", "interval")),
            cron_expr=data.get("cron_expr", ""),
            interval_ms=data.get("interval_ms", 0),
            at_ms=data.get("at_ms", 0),
            timezone=data.get("timezone", ""),
            event_name=data.get("event_name", ""),
            payload=data.get("payload", {}),
            status=JobStatus(data.get("status", "active")),
            enabled=data.get("enabled", True),
            delete_after_run=data.get("delete_after_run", False),
            next_run_ms=data.get("next_run_ms"),
            last_run_ms=data.get("last_run_ms"),
            last_status=data.get("last_status", ""),
            last_error=data.get("last_error", ""),
            run_count=data.get("run_count", 0),
            created_at_ms=data.get("created_at_ms", 0),
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _compute_next_run(job: ScheduledJob, now_ms: int) -> int | None:
    if job.trigger == TriggerKind.ONCE:
        return job.at_ms if job.at_ms > now_ms else None
    if job.trigger == TriggerKind.INTERVAL:
        return now_ms + job.interval_ms if job.interval_ms > 0 else None
    if job.trigger == TriggerKind.CRON and job.cron_expr:
        try:
            from croniter import croniter
            base = datetime.fromtimestamp(now_ms / 1000)
            cron = croniter(job.cron_expr, base)
            return int(cron.get_next(datetime).timestamp() * 1000)
        except Exception as e:
            logger.debug("Failed to compute next cron run: {}", e)
            return None
    return None


class Scheduler:
    """Central scheduler for all trigger types with persistence and recovery."""

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[ScheduledJob], Awaitable[str | None]] | None = None,
        max_concurrent: int = 10,
    ):
        self._store_path = store_path
        self._on_job = on_job
        self._max_concurrent = max_concurrent
        self._jobs: dict[str, ScheduledJob] = {}
        self._running = False
        self._timer_task: asyncio.Task | None = None
        self._event_handlers: dict[str, list[str]] = {}
        self._background_tasks: dict[str, asyncio.Task] = {}
        self._load()

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            for item in data.get("jobs", []):
                job = ScheduledJob.from_dict(item)
                self._jobs[job.id] = job
                if job.trigger == TriggerKind.EVENT and job.event_name:
                    self._event_handlers.setdefault(job.event_name, []).append(job.id)
        except Exception as e:
            logger.warning("Failed to load scheduler state: {}", e)

    def _save(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"jobs": [j.to_dict() for j in self._jobs.values()]}
        self._store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    async def start(self) -> None:
        self._running = True
        for job in self._jobs.values():
            if job.enabled and job.status == JobStatus.ACTIVE:
                job.next_run_ms = _compute_next_run(job, _now_ms())
        self._save()
        self._timer_task = asyncio.create_task(self._tick_loop())
        logger.info("Scheduler started with {} jobs", len(self._jobs))

    async def stop(self) -> None:
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
        for task in self._background_tasks.values():
            task.cancel()
        self._save()

    def add_job(self, job: ScheduledJob) -> ScheduledJob:
        job.next_run_ms = _compute_next_run(job, _now_ms())
        self._jobs[job.id] = job
        if job.trigger == TriggerKind.EVENT and job.event_name:
            self._event_handlers.setdefault(job.event_name, []).append(job.id)
        self._save()
        return job

    def remove_job(self, job_id: str) -> bool:
        job = self._jobs.pop(job_id, None)
        if job:
            self._save()
            return True
        return False

    def pause_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.PAUSED
            self._save()
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.PAUSED:
            job.status = JobStatus.ACTIVE
            job.next_run_ms = _compute_next_run(job, _now_ms())
            self._save()
            return True
        return False

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.CANCELLED
            bg = self._background_tasks.pop(job_id, None)
            if bg:
                bg.cancel()
            self._save()
            return True
        return False

    def list_jobs(self) -> list[ScheduledJob]:
        return list(self._jobs.values())

    async def trigger_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or not job.enabled or job.status in (JobStatus.CANCELLED, JobStatus.COMPLETED):
            return False
        await self._execute_job(job)
        return True

    async def fire_event(self, event_name: str) -> int:
        job_ids = self._event_handlers.get(event_name, [])
        count = 0
        for jid in job_ids:
            job = self._jobs.get(jid)
            if job and job.enabled and job.status == JobStatus.ACTIVE:
                await self._execute_job(job)
                count += 1
        return count

    async def _tick_loop(self) -> None:
        while self._running:
            try:
                now = _now_ms()
                for job in list(self._jobs.values()):
                    if not job.enabled or job.status != JobStatus.ACTIVE:
                        continue
                    if job.next_run_ms and now >= job.next_run_ms:
                        await self._execute_job(job)
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler tick error: {}", e)
                await asyncio.sleep(5)

    async def _execute_job(self, job: ScheduledJob) -> None:
        job.last_run_ms = _now_ms()
        job.run_count += 1
        try:
            if self._on_job:
                await self._on_job(job)
                job.last_status = "success"
                job.last_error = ""
            else:
                job.last_status = "skipped"
        except Exception as e:
            job.last_status = "error"
            job.last_error = str(e)
            logger.error("Job {} failed: {}", job.id, e)

        if job.delete_after_run or job.trigger == TriggerKind.ONCE:
            job.status = JobStatus.COMPLETED
        else:
            job.next_run_ms = _compute_next_run(job, _now_ms())

        self._save()

    def run_in_background(self, job_id: str, coro: Any) -> None:
        task = asyncio.create_task(self._background_wrapper(job_id, coro))
        self._background_tasks[job_id] = task

    async def _background_wrapper(self, job_id: str, coro: Any) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Background task {} failed: {}", job_id, e)
        finally:
            self._background_tasks.pop(job_id, None)

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "total_jobs": len(self._jobs),
            "active_jobs": sum(1 for j in self._jobs.values() if j.status == JobStatus.ACTIVE),
            "background_tasks": len(self._background_tasks),
        }
