"""Tests for Consolidator — distilling the daily notes into MEMORY.md.

They run against a fake one-shot backend, so they pin what is sent, what is
written (MEMORY.md, .history/, meta.json) and when it runs, without an SDK.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from memclaw.config import ClaudeSettings, MemclawConfig, OpenAISettings
from memclaw.memory import Consolidator, MemoryIndex, MemoryStore
from memclaw.memory.consolidation import _CONSOLIDATION_BUDGET_CHARS as BUDGET

# The smallest reply that is a MEMORY.md (the prompt's required headings).
VALID = "# Personal Memory\n\n## Index\n"


@pytest.fixture
def cfg(tmp_path: Path) -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai=OpenAISettings(api_key="test-openai-key"),
        claude=ClaudeSettings(api_key="test-anthropic-key"),
    )


@dataclass
class FakeBackend:
    """One-shot stub that records calls and returns a canned reply."""

    one_shot_response: str = ""
    one_shot_calls: list[dict[str, Any]] = field(default_factory=list)

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str:
        self.one_shot_calls.append({"system": system_prompt, "user": user_message})
        return self.one_shot_response


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def index(cfg: MemclawConfig):
    idx = MemoryIndex(cfg)
    yield idx
    idx.close()


@pytest.fixture
def consolidator(cfg: MemclawConfig, index: MemoryIndex, fake_backend: FakeBackend) -> Consolidator:
    return Consolidator(cfg, MemoryStore(cfg), index, fake_backend)


@pytest.fixture
def warnings_logged():
    """The messages logged at WARNING or above while the test runs."""
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink)


def _daily(cfg: MemclawConfig, d: date, text: str) -> Path:
    path = cfg.daily_file(d)
    path.write_text(text)
    return path


def _consolidated_through(cfg: MemclawConfig) -> str:
    return json.loads((cfg.memory_dir / "meta.json").read_text())["consolidated_through"]


# ────────────────────────────────────────────────────────────────────
# When it runs
# ────────────────────────────────────────────────────────────────────

class TestThreshold:
    @pytest.mark.asyncio
    async def test_skips_when_below_threshold(self, cfg, fake_backend, consolidator):
        """Consolidation should not run when file count < threshold."""
        cfg.memory.consolidation_threshold = 7
        for i in range(3):
            _daily(cfg, date(2025, 3, i + 1), f"# Day {i}\nSome content")

        assert await consolidator.consolidate() is False
        assert fake_backend.one_shot_calls == []

    @pytest.mark.asyncio
    async def test_runs_when_above_threshold(self, cfg, fake_backend, consolidator, index):
        """Consolidation should run when file count >= threshold."""
        cfg.memory.consolidation_threshold = 3
        fake_backend.one_shot_response = "# Personal Memory\n\n## Key Facts\n\n- Fact 0\n- Fact 1\n\n## Index\n"
        for i in range(5):
            _daily(cfg, date(2025, 3, i + 1), f"# Day {i}\nImportant fact {i}")
        index.index_file = AsyncMock()

        assert await consolidator.consolidate() is True
        assert "Key Facts" in cfg.memory_file.read_text()
        assert _consolidated_through(cfg) == "2025-03-05"
        index.index_file.assert_awaited_once_with(cfg.memory_file)

    @pytest.mark.asyncio
    async def test_force_ignores_threshold(self, cfg, fake_backend, consolidator, index):
        """force=True should run consolidation even with 1 file."""
        cfg.memory.consolidation_threshold = 100
        fake_backend.one_shot_response = "# Personal Memory\n\n## Notes\n- One note\n\n## Index"
        _daily(cfg, date(2025, 3, 1), "# Single day\nJust one note")
        index.index_file = AsyncMock()

        assert await consolidator.consolidate(force=True) is True

    @pytest.mark.asyncio
    async def test_since_overrides_meta(self, cfg, fake_backend, consolidator, index):
        """since= takes the files after that date, whatever meta.json says."""
        fake_backend.one_shot_response = VALID
        for i in range(1, 11):
            _daily(cfg, date(2025, 3, i), f"Content for day {i}")
        (cfg.memory_dir / "meta.json").write_text(json.dumps({"consolidated_through": "2025-03-01"}))
        index.index_file = AsyncMock()

        assert await consolidator.consolidate(force=True, since=date(2025, 3, 8)) is True
        user_msg = fake_backend.one_shot_calls[-1]["user"]
        assert "2025-03-09" in user_msg
        assert "2025-03-10" in user_msg
        assert "2025-03-05" not in user_msg

    @pytest.mark.asyncio
    async def test_no_files_returns_false(self, fake_backend, consolidator):
        assert await consolidator.consolidate(force=True) is False
        assert fake_backend.one_shot_calls == []

    @pytest.mark.asyncio
    async def test_by_default_it_runs_once_a_day_has_passed(self, cfg, fake_backend, consolidator):
        fake_backend.one_shot_response = "# Personal Memory\n- x\n## Index"
        _daily(cfg, date.today(), "today")
        assert await consolidator.consolidate() is False  # today isn't over
        _daily(cfg, date.today() - timedelta(days=1), "yesterday")
        assert await consolidator.consolidate() is True

    @pytest.mark.asyncio
    async def test_todays_file_is_left_for_later(self, cfg, fake_backend, consolidator):
        fake_backend.one_shot_response = "# Personal Memory\n## Facts\n- x\n## Index"
        _daily(cfg, date.today() - timedelta(days=1), "yesterday's note")
        _daily(cfg, date.today(), "today's note")

        assert await consolidator.consolidate(force=True)
        assert "today's note" not in fake_backend.one_shot_calls[0]["user"]
        assert _consolidated_through(cfg) == (date.today() - timedelta(days=1)).isoformat()

    @pytest.mark.asyncio
    async def test_only_today_means_nothing_to_do(self, cfg, fake_backend, consolidator):
        _daily(cfg, date.today(), "today's note")
        assert await consolidator.consolidate(force=True) is False
        assert fake_backend.one_shot_calls == []

    @pytest.mark.asyncio
    async def test_empty_daily_files_are_marked_done_without_a_call(self, cfg, fake_backend, consolidator):
        _daily(cfg, date(2025, 3, 1), "   ")
        _daily(cfg, date(2025, 3, 2), "")
        assert await consolidator.consolidate(force=True) is False
        assert fake_backend.one_shot_calls == []
        assert _consolidated_through(cfg) == "2025-03-02"


# ────────────────────────────────────────────────────────────────────
# What is sent
# ────────────────────────────────────────────────────────────────────

class TestMessage:
    @pytest.mark.asyncio
    async def test_content_limit(self, cfg, fake_backend, consolidator):
        """Content gathering should stop at the consolidation budget."""
        fake_backend.one_shot_response = VALID
        for i in range(1, 6):
            _daily(cfg, date(2025, 3, i), "x" * (BUDGET // 3))

        await consolidator.consolidate(force=True)
        assert len(fake_backend.one_shot_calls[-1]["user"]) < BUDGET + 5000

    @pytest.mark.asyncio
    async def test_files_past_the_budget_stay_unconsolidated(self, cfg, fake_backend, consolidator):
        fake_backend.one_shot_response = "# Personal Memory\n## Facts\n- x\n## Index"
        for day in range(1, 6):
            _daily(cfg, date(2025, 3, day), f"day {day} " + "x" * (BUDGET // 3))

        assert await consolidator.consolidate(force=True)
        user_msg = fake_backend.one_shot_calls[0]["user"]
        assert "day 2 " in user_msg and "day 3 " not in user_msg
        assert _consolidated_through(cfg) == "2025-03-02"

        # The next run picks up where this one stopped.
        assert await consolidator.consolidate(force=True)
        assert "day 3 " in fake_backend.one_shot_calls[1]["user"]

    @pytest.mark.asyncio
    async def test_single_oversized_file_is_truncated_not_stuck(self, cfg, fake_backend, consolidator):
        fake_backend.one_shot_response = "# Personal Memory\n## Facts\n- x\n## Index"
        _daily(cfg, date(2025, 3, 1), "y" * (BUDGET + 20000))
        assert await consolidator.consolidate(force=True)
        assert _consolidated_through(cfg) == "2025-03-01"

    @pytest.mark.asyncio
    async def test_consolidation_is_told_todays_date(self, cfg, fake_backend, consolidator):
        fake_backend.one_shot_response = VALID
        _daily(cfg, date(2025, 3, 1), "note")
        await consolidator.consolidate(force=True)
        assert f"Today's date: {date.today().isoformat()}" in fake_backend.one_shot_calls[0]["user"]

    @pytest.mark.asyncio
    async def test_prompt_and_message_layout(self, cfg, fake_backend, consolidator):
        fake_backend.one_shot_response = VALID
        cfg.memory_file.write_text("# Personal Memory\n- old fact\n")
        _daily(cfg, date(2025, 3, 1), "note one")
        await consolidator.consolidate(force=True)
        call = fake_backend.one_shot_calls[0]
        assert call["system"].startswith("You maintain MEMORY.md, the user's permanent memory.")
        assert call["user"] == (
            f"Today's date: {date.today().isoformat()}\n\n## Daily Memory Files\n\n"
            "\n### 2025-03-01\n\nnote one"
            "\n\n## Current MEMORY.md\n\n# Personal Memory\n- old fact\n"
        )

    @pytest.mark.asyncio
    async def test_other_notes_are_listed_for_the_index(self, cfg, fake_backend, consolidator):
        fake_backend.one_shot_response = VALID
        (cfg.memory_dir / "todos.md").write_text("# Todos\n- renew passport")
        _daily(cfg, date(2025, 3, 1), "note")
        await consolidator.consolidate(force=True)
        user_msg = fake_backend.one_shot_calls[0]["user"]
        assert "## Other Notes" in user_msg
        assert "- todos.md (" in user_msg and "Todos" in user_msg
        assert "AGENTS.md" not in user_msg.split("## Other Notes")[1]


# ────────────────────────────────────────────────────────────────────
# Writing MEMORY.md
# ────────────────────────────────────────────────────────────────────

class TestWriting:
    @pytest.mark.asyncio
    async def test_save_during_consolidation_is_kept(self, cfg, fake_backend, consolidator):
        _daily(cfg, date(2025, 3, 1), "note")

        async def one_shot(**kwargs):
            with open(cfg.memory_file, "a") as f:  # appended by hand meanwhile
                f.write("- Blood type 0 negative\n")
            return "# Personal Memory\n- consolidated\n## Index"

        fake_backend.run_one_shot = one_shot
        assert await consolidator.consolidate(force=True)
        memory = cfg.memory_file.read_text()
        assert memory.startswith("# Personal Memory\n- consolidated\n## Index\n")
        assert "Blood type 0 negative" in memory
        history = list((cfg.memory_dir / ".history").glob("MEMORY-*.md"))
        assert [h.read_text() for h in history] == ["# Personal Memory\n"]

    @pytest.mark.asyncio
    async def test_manual_edit_during_consolidation_aborts(self, cfg, fake_backend, consolidator):
        _daily(cfg, date(2025, 3, 1), "note")

        async def one_shot(**kwargs):
            cfg.memory_file.write_text("# Edited by hand\n")
            return "# Personal Memory\n- consolidated\n## Index"

        fake_backend.run_one_shot = one_shot
        assert await consolidator.consolidate(force=True) is False
        assert cfg.memory_file.read_text() == "# Edited by hand\n"
        assert not (cfg.memory_dir / "meta.json").exists()

    @pytest.mark.asyncio
    async def test_failed_backend_call_leaves_memory_untouched(self, cfg, fake_backend, consolidator):
        _daily(cfg, date(2025, 3, 1), "note")
        before = cfg.memory_file.read_text()

        async def one_shot(**kwargs):
            raise RuntimeError("Claude one-shot failed (unknown): API Error: Connection refused")

        fake_backend.run_one_shot = one_shot
        with pytest.raises(RuntimeError):
            await consolidator.consolidate(force=True)
        assert cfg.memory_file.read_text() == before
        assert not (cfg.memory_dir / "meta.json").exists()
        assert not (cfg.memory_dir / ".history").exists()

    @pytest.mark.parametrize("reply", [
        "API Error: Connection refused",  # a backend's error text, returned instead of raised
        "I couldn't generate a response.",  # Cursor's placeholder for an empty result
        "  \n",  # empty
        "# Personal Memory\n\n## About\n- Ana, lives in",  # cut off before the index
        "Here is the updated MEMORY.md:\n\n# Personal Memory\n\n## Index\n",  # preamble
    ], ids=["error", "placeholder", "empty", "cut-off", "preamble"])
    @pytest.mark.asyncio
    async def test_a_reply_that_is_not_a_memory_file_is_rejected(
        self, cfg, fake_backend, consolidator, reply, warnings_logged,
    ):
        """Raised like a failed backend call (so the background run waits an
        hour), with MEMORY.md, .history/ and meta.json untouched."""
        _daily(cfg, date(2025, 3, 1), "note")
        cfg.memory_file.write_text("# Personal Memory\n- old fact\n\n## Index\n")
        before = cfg.memory_file.read_text()
        fake_backend.one_shot_response = reply
        with pytest.raises(RuntimeError, match="not a MEMORY.md"):
            await consolidator.consolidate(force=True)
        assert cfg.memory_file.read_text() == before
        assert not (cfg.memory_dir / "meta.json").exists()
        assert not (cfg.memory_dir / ".history").exists()
        [warning] = [m for m in warnings_logged if "not a MEMORY.md" in m]
        assert repr(reply.strip()[:80]) in warning

    @pytest.mark.asyncio
    async def test_a_valid_reply_is_written(self, cfg, fake_backend, consolidator):
        _daily(cfg, date(2025, 3, 1), "note")
        fake_backend.one_shot_response = "\n# Personal Memory\n\n## About\n- Ana\n\n## Index\n### Notes\n"
        assert await consolidator.consolidate(force=True)
        assert cfg.memory_file.read_text() == "# Personal Memory\n\n## About\n- Ana\n\n## Index\n### Notes\n"
        assert _consolidated_through(cfg) == "2025-03-01"

    @pytest.mark.asyncio
    async def test_code_fence_is_stripped(self, cfg, fake_backend, consolidator):
        fake_backend.one_shot_response = "```markdown\n# Personal Memory\n- x\n## Index\n```"
        _daily(cfg, date(2025, 3, 1), "note")
        assert await consolidator.consolidate(force=True)
        assert cfg.memory_file.read_text() == "# Personal Memory\n- x\n## Index\n"

    @pytest.mark.asyncio
    async def test_references_to_missing_notes_are_dropped(self, cfg, fake_backend, consolidator):
        fake_backend.one_shot_response = (
            "# Personal Memory\n\n## Index\n"
            "- Lion8 login → memory/2025-03-01.md, memory/2024-01-01.md\n"
        )
        _daily(cfg, date(2025, 3, 1), "Lion8 login: admin")
        await consolidator.consolidate(force=True)
        assert "- Lion8 login → memory/2025-03-01.md\n" in cfg.memory_file.read_text()

    @pytest.mark.asyncio
    async def test_history_keeps_the_last_60(self, cfg, fake_backend, consolidator):
        history = cfg.memory_dir / ".history"
        history.mkdir()
        for i in range(65):
            (history / f"MEMORY-2025-01-01-{i:06d}.md").write_text(str(i))
        cfg.memory_file.write_text("latest")
        fake_backend.one_shot_response = "# Personal Memory\n- next\n## Index"
        _daily(cfg, date(2025, 3, 1), "note")
        assert await consolidator.consolidate(force=True)
        kept = sorted(history.glob("MEMORY-*.md"))
        assert len(kept) == 60
        assert kept[-1].read_text() == "latest"
        assert not (history / "MEMORY-2025-01-01-000000.md").exists()

    @pytest.mark.asyncio
    async def test_indexing_failure_still_counts_as_done(self, cfg, fake_backend, consolidator, index):
        fake_backend.one_shot_response = "# Personal Memory\n- x\n## Index"
        _daily(cfg, date(2025, 3, 1), "note")
        index.index_file = AsyncMock(side_effect=RuntimeError("OpenAI down"))
        assert await consolidator.consolidate(force=True)
        assert cfg.memory_file.read_text() == "# Personal Memory\n- x\n## Index\n"
        assert _consolidated_through(cfg) == "2025-03-01"

    @pytest.mark.asyncio
    async def test_unreadable_meta_starts_over(self, cfg, fake_backend, consolidator):
        (cfg.memory_dir / "meta.json").write_text("[1, 2]")
        fake_backend.one_shot_response = "# Personal Memory\n- x\n## Index"
        _daily(cfg, date(2025, 3, 1), "note")
        assert await consolidator.consolidate()
        assert json.loads((cfg.memory_dir / "meta.json").read_text()) == {"consolidated_through": "2025-03-01"}


# ────────────────────────────────────────────────────────────────────
# Background schedule
# ────────────────────────────────────────────────────────────────────

class TestSchedule:
    @pytest.mark.asyncio
    async def test_at_most_one_runs_at_a_time(self, consolidator):
        release = asyncio.Event()

        async def slow(**kwargs):
            await release.wait()
            return True

        consolidator.consolidate = AsyncMock(side_effect=slow)
        consolidator.schedule()
        await asyncio.sleep(0)
        consolidator.schedule()  # one already running: no second one
        release.set()
        await consolidator._task
        assert consolidator.consolidate.await_count == 1

    @pytest.mark.asyncio
    async def test_failure_is_logged_and_not_retried_for_an_hour(self, consolidator):
        consolidator.consolidate = AsyncMock(side_effect=RuntimeError("bad model"))
        consolidator.schedule()
        await consolidator._task  # does not raise
        consolidator.schedule()
        await consolidator._task
        assert consolidator.consolidate.await_count == 1
        assert 3590 < consolidator._retry_at - time.monotonic() <= 3600

        consolidator._retry_at = time.monotonic() - 1  # the hour has passed
        consolidator.schedule()
        await consolidator._task
        assert consolidator.consolidate.await_count == 2

    @pytest.mark.asyncio
    async def test_aclose_cancels_a_running_consolidation(self, consolidator):
        async def forever(**kwargs):
            await asyncio.sleep(60)

        consolidator.consolidate = AsyncMock(side_effect=forever)
        consolidator.schedule()
        await asyncio.sleep(0)
        task = consolidator._task
        await consolidator.aclose()
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_stops_without_waiting(self, consolidator):
        async def forever(**kwargs):
            await asyncio.sleep(60)

        consolidator.consolidate = AsyncMock(side_effect=forever)
        consolidator.schedule()
        await asyncio.sleep(0)
        task = consolidator._task
        consolidator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_aclose_without_a_run_is_a_no_op(self, consolidator):
        await consolidator.aclose()
        consolidator.cancel()
