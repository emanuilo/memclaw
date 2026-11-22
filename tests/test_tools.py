"""The agent's tools, called through ToolExecutor: every tool's replies,
including the error and edge-case wording the agent sees."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from memclaw.memory import SearchResult
from memclaw.tools import ToolContext, ToolExecutor


def _executor(config, *, platform="telegram", scheduler=None, chat_id="42", found_images=None):
    store = MagicMock()
    store.save.return_value = config.memory_dir / "memory" / "2026-01-01.md"
    index = MagicMock()
    index.index_file = AsyncMock()
    index.store_platform_image = AsyncMock()
    search = MagicMock()
    search.search = AsyncMock(return_value=[])
    search.search_images = AsyncMock(return_value=[])
    executor = ToolExecutor(ToolContext(
        config=config, store=store, index=index, search=search,
        found_images=[] if found_images is None else found_images,
        platform=platform, scheduler=scheduler, chat_id=chat_id,
    ))
    return executor, store, index, search


@pytest.fixture
def config(tmp_config):
    return tmp_config


class TestDispatch:
    @pytest.mark.asyncio
    async def test_unknown_tool(self, config):
        executor, *_ = _executor(config)
        assert await executor.execute("nope", {}) == "Error: Unknown tool 'nope'"

    @pytest.mark.asyncio
    async def test_a_failing_tool_is_reported_not_raised(self, config):
        executor, *_ = _executor(config)
        assert await executor.execute("memory_save", {}) == "Error executing memory_save: 'content'"


class TestMemoryTools:
    @pytest.mark.asyncio
    async def test_memory_save(self, config):
        executor, store, index, _ = _executor(config)
        out = await executor.execute(
            "memory_save", {"content": "Max is a beagle", "entry_type": "fact", "tags": ["dog"]},
        )
        assert out == "Memory saved to 2026-01-01.md"
        store.save.assert_called_once_with("Max is a beagle", entry_type="fact", tags=["dog"])
        index.index_file.assert_awaited_once_with(store.save.return_value)

    @pytest.mark.asyncio
    async def test_memory_save_with_indexing_down(self, config):
        executor, _, index, _ = _executor(config)
        index.index_file.side_effect = RuntimeError("offline")
        out = await executor.execute("memory_save", {"content": "x"})
        assert out == (
            "Memory saved to 2026-01-01.md"
            " (search indexing is delayed; it will be searchable soon)"
        )

    @pytest.mark.asyncio
    async def test_memory_search_formats_results(self, config):
        executor, _, _, search = _executor(config)
        search.search.return_value = [
            SearchResult(str(config.memory_file), 1, 2, " Likes tea \n", 0.912, "hybrid"),
            SearchResult("/m/memory/2026-01-01.md", 3, 4, "Max is a beagle", 0.5, "vector"),
        ]
        out = await executor.execute("memory_search", {"query": "dog", "limit": 99})
        search.search.assert_awaited_once_with("dog", limit=25)
        assert out == (
            "**[1]** (score: 0.91, source: permanent memory)\nLikes tea"
            "\n\n---\n\n"
            "**[2]** (score: 0.50, source: 2026-01-01)\nMax is a beagle"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("limit", "expected"), [(None, 10), ("x", 10), (0, 10), (-3, 1), (7, 7)])
    async def test_memory_search_limit(self, config, limit, expected):
        executor, _, _, search = _executor(config)
        out = await executor.execute("memory_search", {"query": "q", "limit": limit})
        assert out == "No matching memories found."
        assert search.search.await_args.kwargs["limit"] == expected


class TestImageTools:
    @pytest.mark.asyncio
    async def test_image_save(self, config):
        executor, store, index, _ = _executor(config, platform=None)
        out = await executor.execute(
            "image_save", {"description": "A cat " * 30, "media_ref": "f1", "caption": "Tom"},
        )
        assert out == f"Image saved: {('A cat ' * 30)[:100]}"
        index.store_platform_image.assert_awaited_once_with(
            platform="local", media_ref="f1",
            description=f"Image: {'A cat ' * 30} Caption: Tom", caption="Tom",
        )
        store.save.assert_called_once_with(f"Image: {'A cat ' * 30} Caption: Tom", entry_type="image")

    @pytest.mark.asyncio
    async def test_image_search(self, config):
        executor, _, _, search = _executor(config)
        search.search_images.return_value = [
            {"id": 3, "score": 0.81234, "description": "word " * 100},
        ]
        out = await executor.execute("image_search", {"query": "cat"})
        search.search_images.assert_awaited_once_with("cat", limit=5, platform="telegram")
        description = " ".join(("word " * 100).split())[:300] + "…"
        assert out == (
            "Closest stored images, best first (nothing sent yet):\n"
            f"- id 3 (match 0.81): {description}"
            "\nCall image_send with the ids of the ones the user asked for."
        )

    @pytest.mark.asyncio
    async def test_image_search_nothing_found(self, config):
        executor, *_ = _executor(config)
        assert await executor.execute("image_search", {"query": "cat"}) == "No matching images found."

    @pytest.mark.asyncio
    async def test_image_send(self, config):
        found: list[dict] = [{"id": 1}]
        executor, _, index, _ = _executor(config, found_images=found)
        index.get_platform_images.return_value = [{"id": 1}, {"id": 2}]
        out = await executor.execute("image_send", {"ids": ["1", 2, 5, 4]})
        index.get_platform_images.assert_called_once_with([1, 2, 5, 4], platform="telegram")
        assert out == "Sending 2 image(s) with your reply. No stored image with id 4, 5."
        assert found == [{"id": 1}, {"id": 2}]

    @pytest.mark.asyncio
    async def test_image_send_nothing(self, config):
        executor, _, index, _ = _executor(config)
        index.get_platform_images.return_value = []
        assert await executor.execute("image_send", {}) == "Nothing to send."


class TestFileTools:
    @pytest.mark.asyncio
    async def test_update_instructions(self, config):
        config.agent_file.write_text("# Rules\n")
        executor, *_ = _executor(config)
        out = await executor.execute("update_instructions", {"instruction": "  Be brief \n"})
        assert out == "Instruction saved: Be brief"
        assert config.agent_file.read_text() == "# Rules\n\n- Be brief\n"

    @pytest.mark.asyncio
    async def test_file_write_and_read(self, config):
        executor, _, index, _ = _executor(config)
        index.is_indexable.return_value = False
        target = config.memory_dir.resolve() / "lists" / "todo.txt"
        assert await executor.execute(
            "file_write", {"file_path": "lists/todo.txt", "content": "milk"},
        ) == f"File written: {target}"
        index.index_file.assert_not_awaited()
        assert await executor.execute("file_read", {"file_path": "lists/todo.txt"}) == "milk"
        assert await executor.execute("file_read", {"file_path": "none.md"}) == (
            f"File not found: {config.memory_dir.resolve() / 'none.md'}"
        )

    @pytest.mark.asyncio
    async def test_blocked_paths(self, config):
        executor, *_ = _executor(config)
        root = config.memory_dir.resolve()
        assert await executor.execute("file_write", {"file_path": "/etc/x", "content": ""}) == (
            f"Blocked: path is outside {root} or is one of Memclaw's "
            "internal files. Files must be under ~/.memclaw/"
        )
        assert await executor.execute("file_read", {"file_path": "../x"}) == (
            f"Blocked: path is outside {root} or is one of Memclaw's internal files."
        )
        assert await executor.execute("file_write", {"file_path": "MEMORY.md", "content": ""}) == (
            "Blocked: MEMORY.md and the daily notes in memory/ are maintained "
            "by Memclaw. Use memory_save to add a memory."
        )


class TestReminderTools:
    @pytest.mark.asyncio
    async def test_needs_a_scheduler_and_a_chat(self, config):
        executor, *_ = _executor(config, scheduler=None)
        for tool in ("reminder_create", "reminder_list", "reminder_cancel"):
            assert await executor.execute(tool, {}) == (
                "Reminders are only available when running in a messaging bot "
                "(telegram/whatsapp/slack)."
            )
        executor, *_ = _executor(config, scheduler=MagicMock(), chat_id=None)
        assert await executor.execute("reminder_list", {}) == (
            "Reminders require a messaging context (missing platform/chat_id)."
        )

    @pytest.mark.asyncio
    async def test_create_one_shot(self, config):
        scheduler = MagicMock()
        scheduler.create.return_value = 7
        executor, *_ = _executor(config, scheduler=scheduler)
        out = await executor.execute(
            "reminder_create", {"text": " call Alex ", "fire_at": "2026-04-13T21:30:00"},
        )
        assert out == "Reminder #7 scheduled for 2026-04-13T21:30."
        scheduler.create.assert_called_once_with(
            platform="telegram", chat_id="42", text="call Alex",
            fire_at=datetime(2026, 4, 13, 21, 30), interval_seconds=None,
        )

    @pytest.mark.asyncio
    async def test_create_converts_an_aware_time_to_local(self, config):
        scheduler = MagicMock()
        scheduler.create.return_value = 1
        executor, *_ = _executor(config, scheduler=scheduler)
        await executor.execute("reminder_create", {"text": "t", "fire_at": "2026-04-13T21:30:00+00:00"})
        expected = datetime(2026, 4, 13, 21, 30, tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
        assert scheduler.create.call_args.kwargs["fire_at"] == expected

    @pytest.mark.asyncio
    async def test_create_recurring(self, config):
        scheduler = MagicMock()
        scheduler.create.return_value = 2
        executor, *_ = _executor(config, scheduler=scheduler)
        before = datetime.now()
        out = await executor.execute("reminder_create", {"text": "water", "interval_seconds": 18000})
        fire_at = scheduler.create.call_args.kwargs["fire_at"]
        assert before + timedelta(seconds=18000) <= fire_at <= datetime.now() + timedelta(seconds=18000)
        assert out == (
            f"Reminder #2 scheduled for {fire_at.isoformat(timespec='minutes')} "
            "and every 18000 seconds after."
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("args", "expected"), [
        ({"text": "t", "fire_at": "soon"}, "Invalid fire_at: 'soon'. Use local ISO 8601."),
        ({"text": "t"}, "Provide either fire_at or interval_seconds (or both)."),
        ({"text": "t", "interval_seconds": 30}, "interval_seconds must be at least 60."),
    ])
    async def test_create_rejects(self, config, args, expected):
        scheduler = MagicMock()
        executor, *_ = _executor(config, scheduler=scheduler)
        assert await executor.execute("reminder_create", args) == expected
        scheduler.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_list(self, config):
        scheduler = MagicMock()
        scheduler.list_for.return_value = [
            {"id": 1, "fire_at": "2026-04-13T21:30:00", "text": "call", "interval_seconds": None},
            {"id": 2, "fire_at": "2026-04-14T09:00:00", "text": "water", "interval_seconds": 3600},
        ]
        executor, *_ = _executor(config, scheduler=scheduler)
        assert await executor.execute("reminder_list", {}) == (
            "#1 at 2026-04-13T21:30:00: call\n#2 at 2026-04-14T09:00:00: water (every 3600s)"
        )
        scheduler.list_for.assert_called_once_with("telegram", "42")
        scheduler.list_for.return_value = []
        assert await executor.execute("reminder_list", {}) == "No pending reminders."

    @pytest.mark.asyncio
    async def test_cancel(self, config):
        scheduler = MagicMock()
        scheduler.cancel.side_effect = [True, False]
        executor, *_ = _executor(config, scheduler=scheduler)
        assert await executor.execute("reminder_cancel", {"id": "3"}) == "Reminder #3 cancelled."
        assert await executor.execute("reminder_cancel", {"id": 4}) == "No pending reminder #4."
        scheduler.cancel.assert_called_with(4, platform="telegram", chat_id="42")
