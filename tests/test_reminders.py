"""Tests for the reminder scheduler."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from memclaw.reminders import ReminderScheduler


@pytest.fixture
def scheduler(tmp_config):
    s = ReminderScheduler(tmp_config, poll_interval=0.05)
    yield s
    s.close()


def test_create_and_list(scheduler):
    fire_at = datetime.now() + timedelta(hours=1)
    rid = scheduler.create(
        platform="telegram",
        chat_id="42",
        text="call Alex",
        fire_at=fire_at,
    )
    assert rid > 0
    items = scheduler.list_for("telegram", "42")
    assert len(items) == 1
    assert items[0]["text"] == "call Alex"
    assert items[0]["interval_seconds"] is None


def test_list_is_scoped_per_chat(scheduler):
    fire_at = datetime.now() + timedelta(hours=1)
    scheduler.create(platform="telegram", chat_id="a", text="one", fire_at=fire_at)
    scheduler.create(platform="telegram", chat_id="b", text="two", fire_at=fire_at)
    assert len(scheduler.list_for("telegram", "a")) == 1
    assert len(scheduler.list_for("telegram", "b")) == 1
    assert len(scheduler.list_for("slack", "a")) == 0


def test_cancel(scheduler):
    rid = scheduler.create(
        platform="telegram",
        chat_id="42",
        text="x",
        fire_at=datetime.now() + timedelta(hours=1),
    )
    assert scheduler.cancel(rid, platform="telegram", chat_id="42") is True
    assert scheduler.list_for("telegram", "42") == []
    # Second cancel is a no-op
    assert scheduler.cancel(rid, platform="telegram", chat_id="42") is False


@pytest.mark.asyncio
async def test_one_shot_fires_and_completes(scheduler):
    received: list[tuple[str, str]] = []

    async def deliver(chat_id: str, text: str):
        received.append((chat_id, text))

    scheduler.register_delivery("telegram", deliver)
    scheduler.create(
        platform="telegram",
        chat_id="42",
        text="ping",
        fire_at=datetime.now() - timedelta(seconds=1),
    )

    scheduler.start()
    await asyncio.wait_for(_wait_for(lambda: received), timeout=2.0)
    scheduler.stop()

    assert received[0][0] == "42"
    assert "ping" in received[0][1]
    # Status flipped to done -> no longer pending
    assert scheduler.list_for("telegram", "42") == []


@pytest.mark.asyncio
async def test_recurring_reschedules(scheduler):
    received: list[str] = []

    async def deliver(chat_id: str, text: str):
        received.append(text)

    scheduler.register_delivery("telegram", deliver)
    rid = scheduler.create(
        platform="telegram",
        chat_id="42",
        text="tick",
        fire_at=datetime.now() - timedelta(seconds=1),
        interval_seconds=3600,
    )

    scheduler.start()
    await asyncio.wait_for(_wait_for(lambda: received), timeout=2.0)
    scheduler.stop()

    # Still pending — fire_at advanced by one interval
    items = scheduler.list_for("telegram", "42")
    assert len(items) == 1
    assert items[0]["id"] == rid
    new_fire = datetime.fromisoformat(items[0]["fire_at"])
    assert new_fire > datetime.now()


async def _wait_for(predicate, poll=0.02):
    while not predicate():
        await asyncio.sleep(poll)
