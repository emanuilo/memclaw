"""Tests for MemoryVault — the notes and the index opened together for the
one-off `memclaw` commands (save, search, index, consolidate, status)."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from memclaw.config import MemclawConfig
from memclaw.memory import MemoryVault


class _OneShot:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str:
        self.calls += 1
        return self.reply


@pytest.mark.asyncio
async def test_save_writes_the_daily_note_and_indexes_it(tmp_config: MemclawConfig):
    with MemoryVault(tmp_config) as vault:
        path = await vault.save("Alex's birthday is March 3")

        assert path == tmp_config.daily_file()
        assert "Alex's birthday is March 3" in path.read_text()
        assert vault.index.get_stats()["files"] == 1


@pytest.mark.asyncio
async def test_search_finds_a_saved_note(tmp_config: MemclawConfig):
    with MemoryVault(tmp_config) as vault:
        await vault.save("The spare key is under the blue flowerpot")
        results = await vault.search("spare key flowerpot", limit=3)

    assert results
    assert "blue flowerpot" in results[0].content


@pytest.mark.asyncio
async def test_reindex_reports_whether_anything_changed(tmp_config: MemclawConfig):
    with MemoryVault(tmp_config) as vault:
        vault.store.save("A note")
        changed, stats = await vault.reindex()
        assert changed is True
        assert stats["files"] >= 1 and stats["chunks"] >= 1

        changed, again = await vault.reindex()
        assert changed is False
        assert again == stats


@pytest.mark.asyncio
async def test_sync_or_warn_swallows_a_failing_sync(tmp_config: MemclawConfig):
    with MemoryVault(tmp_config) as vault:
        with patch.object(vault.index, "sync", AsyncMock(side_effect=RuntimeError("offline"))):
            await vault.sync_or_warn()


@pytest.mark.asyncio
async def test_consolidate_writes_memory_md_with_the_backend_given(tmp_config: MemclawConfig):
    backend = _OneShot("## Key Facts\n- Alex's birthday is March 3")
    yesterday = date.today() - timedelta(days=1)
    with MemoryVault(tmp_config) as vault:
        # Today's note is still being written; consolidation takes the days before.
        tmp_config.daily_file(yesterday).write_text("# Yesterday\n\nAlex's birthday is March 3\n")

        assert await vault.consolidate(backend) is True

    assert backend.calls == 1
    assert "Alex's birthday is March 3" in tmp_config.memory_file.read_text()


@pytest.mark.asyncio
async def test_consolidate_with_no_daily_notes_does_nothing(tmp_config: MemclawConfig):
    backend = _OneShot("unused")
    with MemoryVault(tmp_config) as vault:
        assert await vault.consolidate(backend) is False
    assert backend.calls == 0


def test_status_rows(tmp_config: MemclawConfig):
    with MemoryVault(tmp_config) as vault:
        vault.store.save("A note")
        store_rows = dict(vault.store.status_rows())
        index_rows = dict(vault.index.status_rows())

    assert store_rows["Memory directory"] == tmp_config.memory_dir
    assert store_rows["Memory files"] == 2  # MEMORY.md and today's note
    assert index_rows == {"Indexed chunks": 0, "Stored images": 0, "Database": tmp_config.db_path}


def test_closing_the_vault_closes_the_index(tmp_config: MemclawConfig):
    with MemoryVault(tmp_config) as vault:
        pass
    with pytest.raises(Exception, match="closed"):
        vault.index.get_stats()
