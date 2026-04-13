"""Reminder scheduler for Memclaw.

Reminders are stored in the same SQLite database as the memory index. A
single background asyncio task polls for due reminders and invokes a
platform-specific delivery callback. One-shot reminders are marked done
after firing; recurring reminders have their ``fire_at`` advanced by
``interval_seconds``.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from loguru import logger

from .config import MemclawConfig

Delivery = Callable[[str, str], Awaitable[None]]


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


class ReminderScheduler:
    """Polls a SQLite-backed reminder table and fires due reminders."""

    def __init__(self, config: MemclawConfig, poll_interval: float = 20.0):
        self.config = config
        self.poll_interval = poll_interval
        self._db = sqlite3.connect(str(config.db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self._task: asyncio.Task | None = None
        self._delivery: dict[str, Delivery] = {}

    def _init_schema(self):
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                text TEXT NOT NULL,
                fire_at TEXT NOT NULL,
                interval_seconds INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"""
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_due "
            "ON reminders (status, fire_at)"
        )
        self._db.commit()

    # ── CRUD ─────────────────────────────────────────────────────────

    def create(
        self,
        *,
        platform: str,
        chat_id: str,
        text: str,
        fire_at: datetime,
        interval_seconds: int | None = None,
    ) -> int:
        cur = self._db.execute(
            "INSERT INTO reminders (platform, chat_id, text, fire_at, interval_seconds) "
            "VALUES (?, ?, ?, ?, ?)",
            (platform, chat_id, text, _iso(fire_at), interval_seconds),
        )
        self._db.commit()
        return cur.lastrowid or 0

    def list_for(self, platform: str, chat_id: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, text, fire_at, interval_seconds, status "
            "FROM reminders WHERE platform = ? AND chat_id = ? AND status = 'pending' "
            "ORDER BY fire_at ASC",
            (platform, chat_id),
        ).fetchall()
        return [
            {
                "id": r[0],
                "text": r[1],
                "fire_at": r[2],
                "interval_seconds": r[3],
                "status": r[4],
            }
            for r in rows
        ]

    def cancel(self, reminder_id: int, *, platform: str, chat_id: str) -> bool:
        cur = self._db.execute(
            "UPDATE reminders SET status = 'cancelled' "
            "WHERE id = ? AND platform = ? AND chat_id = ? AND status = 'pending'",
            (reminder_id, platform, chat_id),
        )
        self._db.commit()
        return cur.rowcount > 0

    # ── Runner ───────────────────────────────────────────────────────

    def register_delivery(self, platform: str, delivery: Delivery):
        self._delivery[platform] = delivery

    def start(self):
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        logger.info("Reminder scheduler started (poll every {s}s)", s=self.poll_interval)
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Reminder tick failed: {exc}", exc=exc)
            try:
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                raise

    async def _tick(self):
        now = datetime.now()
        rows = self._db.execute(
            "SELECT id, platform, chat_id, text, fire_at, interval_seconds "
            "FROM reminders WHERE status = 'pending' AND fire_at <= ?",
            (_iso(now),),
        ).fetchall()

        for rid, platform, chat_id, text, fire_at_str, interval in rows:
            delivery = self._delivery.get(platform)
            if delivery is None:
                continue

            fire_at = datetime.fromisoformat(fire_at_str)
            late_seconds = (now - fire_at).total_seconds()
            message = f"⏰ Reminder: {text}"
            if late_seconds > 60:
                message += f"\n(was scheduled for {fire_at_str})"

            try:
                await delivery(chat_id, message)
                logger.info(
                    "Reminder {id} delivered ({p}:{c})",
                    id=rid, p=platform, c=chat_id,
                )
            except Exception as exc:
                logger.exception(
                    "Reminder {id} delivery failed: {exc}", id=rid, exc=exc,
                )
                continue

            if interval:
                next_fire = fire_at
                while next_fire <= now:
                    next_fire += timedelta(seconds=interval)
                self._db.execute(
                    "UPDATE reminders SET fire_at = ? WHERE id = ?",
                    (_iso(next_fire), rid),
                )
            else:
                self._db.execute(
                    "UPDATE reminders SET status = 'done' WHERE id = ?", (rid,),
                )
            self._db.commit()

    def stop(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def close(self):
        self.stop()
        self._db.close()
