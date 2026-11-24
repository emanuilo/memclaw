"""Tests for MemclawAgent — sessions, context, sync, fs guardrail.

MEMORY.md consolidation itself is tested in tests/memory/test_consolidation.py;
here only that the agent schedules and stops it.

These tests run against a backend-agnostic `FakeBackend` so they exercise the
orchestration in `MemclawAgent` without depending on any SDK. Per-backend
behavior (env scrubbing, MCP wrapping, …) is tested separately in
`tests/backends/`.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from dataclasses import dataclass, field

import pytest

from memclaw.backends.base import TurnRequest, TurnResult
from memclaw.config import ClaudeSettings, MemclawConfig, OpenAISettings
from memclaw.memory import SearchResult


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_config(tmp_path: Path) -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai=OpenAISettings(api_key="test-openai-key"),
        claude=ClaudeSettings(api_key="test-anthropic-key"),
    )


@pytest.fixture
def cfg(tmp_path: Path) -> MemclawConfig:
    return _make_config(tmp_path)


@dataclass
class FakeBackend:
    """Backend stub that records calls and returns canned responses."""

    name: str = "fake"
    display_name: str = "Fake backend"
    bills_per_token: bool = False
    one_shot_response: str = ""
    turn_response: TurnResult = field(default_factory=lambda: TurnResult(text=""))
    one_shot_calls: list[dict[str, Any]] = field(default_factory=list)
    turn_calls: list[dict[str, Any]] = field(default_factory=list)
    reset_calls: list[str] = field(default_factory=list)
    # Session id each turn reports; None mimics a backend without sessions.
    session_id: str | None = "sess-1"
    turn_error: Exception | None = None

    async def on_agent_start(self, tool_executor) -> None:
        pass

    async def on_agent_shutdown(self) -> None:
        pass

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str:
        self.one_shot_calls.append({"system": system_prompt, "user": user_message})
        return self.one_shot_response

    async def run_turn(self, request: TurnRequest, tool_executor) -> TurnResult:
        self.turn_calls.append({**vars(request), "tool_executor": tool_executor})
        if self.turn_error is not None:
            raise self.turn_error
        result = TurnResult(**vars(self.turn_response))
        result.session_id = self.session_id
        return result

    async def reset_session(self, session_key: str) -> None:
        self.reset_calls.append(session_key)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


def _make_agent(cfg: MemclawConfig, backend: FakeBackend, platform: str | None = None):
    from memclaw.agent import MemclawAgent
    return MemclawAgent(cfg, platform=platform, backend=backend)


def _chat_agent(cfg: MemclawConfig, backend: FakeBackend, platform: str | None = None):
    """An agent ready for handle(): no search, no consolidation."""
    agent = _make_agent(cfg, backend, platform)
    agent.search.search = AsyncMock(return_value=[])
    agent.consolidator.consolidate = AsyncMock(return_value=False)
    return agent


# ────────────────────────────────────────────────────────────────────
# Conversation sessions
# ────────────────────────────────────────────────────────────────────

class TestSessions:
    @pytest.mark.asyncio
    async def test_first_turn_starts_a_session(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        fake_backend.turn_response = TurnResult(text="Hello! I'm Memclaw.")
        agent = _chat_agent(cfg, fake_backend)

        reply, _ = await agent.handle("Hello")

        assert reply == "Hello! I'm Memclaw."
        call = fake_backend.turn_calls[0]
        assert call["user_message"] == "Hello"
        assert call["session_key"] == "cli"
        assert call["resume_session_id"] is None
        stored = json.loads((cfg.memory_dir / "sessions.json").read_text())
        assert stored["cli"]["session_id"] == "sess-1"
        agent.close()

    @pytest.mark.asyncio
    async def test_session_key_is_per_chat_and_platform(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend, platform="telegram")
        await agent.handle("hi", chat_id="42")
        assert fake_backend.turn_calls[0]["session_key"] == "telegram:42"
        agent.close()

    @pytest.mark.asyncio
    async def test_next_turn_and_restart_resume_the_session(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("one")
        await agent.handle("two")
        assert fake_backend.turn_calls[1]["resume_session_id"] == "sess-1"
        agent.close()

        restarted = _chat_agent(cfg, fake_backend)
        await restarted.handle("three")
        assert fake_backend.turn_calls[2]["resume_session_id"] == "sess-1"
        restarted.close()

    @pytest.mark.asyncio
    async def test_new_session_id_after_a_failed_resume_is_stored(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("one")
        fake_backend.session_id = "sess-2"  # backend couldn't resume, started fresh
        await agent.handle("two")
        await agent.handle("three")
        assert fake_backend.turn_calls[2]["resume_session_id"] == "sess-2"
        agent.close()

    @pytest.mark.asyncio
    async def test_session_from_other_instructions_is_not_resumed(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        (cfg.memory_dir / "sessions.json").write_text(json.dumps({
            "cli": {"session_id": "old", "fingerprint": "stale",
                    "agents_hash": "x", "memory_hash": "y"},
        }))
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("hi")

        call = fake_backend.turn_calls[0]
        assert call["resume_session_id"] is None
        # A fresh session gets the files in its system prompt, not as updates.
        assert "Updated AGENTS.md" not in call["context"]
        stored = json.loads((cfg.memory_dir / "sessions.json").read_text())
        assert stored["cli"]["session_id"] == "sess-1"
        agent.close()

    @pytest.mark.asyncio
    async def test_backend_without_sessions_stores_nothing(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        fake_backend.session_id = None
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("hi")
        assert not (cfg.memory_dir / "sessions.json").exists()
        agent.close()

    @pytest.mark.asyncio
    async def test_reset_conversation(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _chat_agent(cfg, fake_backend, platform="telegram")
        await agent.handle("one", chat_id="42")
        await agent.handle("other chat", chat_id="7")
        agent.record_reminder_fired("⏰ Reminder: stretch", "42")

        await agent.reset_conversation("42")

        assert fake_backend.reset_calls == ["telegram:42"]
        stored = json.loads((cfg.memory_dir / "sessions.json").read_text())
        assert "telegram:42" not in stored
        assert "telegram:7" in stored
        await agent.handle("two", chat_id="42")
        call = fake_backend.turn_calls[-1]
        assert call["resume_session_id"] is None
        assert "stretch" not in call["context"]
        agent.close()


class TestSystemPromptAndContext:
    @pytest.mark.asyncio
    async def test_system_prompt_is_static(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        cfg.agent_file.write_text("Be terse.")
        cfg.memory_file.write_text("## Key Facts\n- Name: Test")
        agent = _chat_agent(cfg, fake_backend)

        await agent.handle("one")
        await agent.handle("two")

        first, second = (c["system_prompt"] for c in fake_backend.turn_calls)
        assert first == second
        assert "Be terse." in first
        assert "- Name: Test" in first
        assert "Current local time" not in first
        agent.close()

    @pytest.mark.asyncio
    async def test_context_block_carries_time_and_memories(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend)
        agent.search.search = AsyncMock(return_value=[
            SearchResult(file_path="memory/2025-03-01.md", line_start=0, line_end=1,
                         content="Bought   a red\nbike", score=0.9, match_type="hybrid"),
            SearchResult(file_path=str(cfg.memory_file), line_start=0, line_end=1,
                         content="already in the system prompt", score=0.8,
                         match_type="hybrid"),
            SearchResult(file_path="memory/2025-03-02.md", line_start=0, line_end=1,
                         content="x" * 2000, score=0.7, match_type="hybrid"),
        ])

        await agent.handle("what bike?")

        context = fake_backend.turn_calls[0]["context"]
        assert context.startswith("<context>\nCurrent local time: ")
        assert context.endswith("</context>")
        assert "- [2025-03-01] Bought a red bike" in context
        assert "already in the system prompt" not in context
        assert "x" * 501 not in context  # long snippets are cut
        agent.close()

    @pytest.mark.asyncio
    async def test_changed_agents_md_is_sent_once(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        cfg.agent_file.write_text("Be terse.")
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("one")
        assert "Updated AGENTS.md" not in fake_backend.turn_calls[0]["context"]

        cfg.agent_file.write_text("Be terse.\nAlways answer in Spanish.")
        await agent.handle("two")
        await agent.handle("three")

        second = fake_backend.turn_calls[1]["context"]
        assert "Updated AGENTS.md" in second
        assert "Always answer in Spanish." in second
        assert "Updated AGENTS.md" not in fake_backend.turn_calls[2]["context"]
        agent.close()

    @pytest.mark.asyncio
    async def test_changed_memory_md_is_sent_once_even_after_restart(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        cfg.memory_file.write_text("## Key Facts\n- Old fact")
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("one")
        agent.close()

        cfg.memory_file.write_text("## Key Facts\n- New fact")
        restarted = _chat_agent(cfg, fake_backend)
        await restarted.handle("two")
        await restarted.handle("three")

        second = fake_backend.turn_calls[1]["context"]
        assert "Updated permanent memory" in second
        assert "- New fact" in second
        assert "Updated permanent memory" not in fake_backend.turn_calls[2]["context"]
        restarted.close()

    @pytest.mark.asyncio
    async def test_own_writes_during_a_turn_are_not_sent_back(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        cfg.memory_file.write_text("## Key Facts\n- Old fact")
        agent = _chat_agent(cfg, fake_backend)

        async def turn_that_saves(request, tool_executor):
            # update_instructions mid-turn.
            cfg.agent_file.write_text("Be terse.")
            return TurnResult(text="saved", session_id="sess-1")

        agent.backend.run_turn = turn_that_saves
        await agent.handle("be terse")
        agent.backend.run_turn = FakeBackend.run_turn.__get__(fake_backend)
        await agent.handle("next")

        context = fake_backend.turn_calls[-1]["context"]
        assert "Updated AGENTS.md" not in context
        agent.close()

    @pytest.mark.asyncio
    async def test_consolidation_during_a_turn_is_still_sent(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        cfg.memory_file.write_text("## Key Facts\n- Old fact")
        agent = _chat_agent(cfg, fake_backend)

        async def turn_racing_consolidation(request, tool_executor):
            cfg.memory_file.write_text("## Key Facts\n- Consolidated")
            return TurnResult(text="ok", session_id="sess-1")

        agent.backend.run_turn = turn_racing_consolidation
        await agent.handle("one")
        agent.backend.run_turn = FakeBackend.run_turn.__get__(fake_backend)
        await agent.handle("two")

        context = fake_backend.turn_calls[-1]["context"]
        assert "Updated permanent memory" in context
        assert "- Consolidated" in context
        agent.close()

    @pytest.mark.asyncio
    async def test_reminder_notes_go_into_the_next_message_of_that_chat(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend, platform="telegram")
        agent.record_reminder_fired("⏰ Reminder: call Alex", "42")

        await agent.handle("other chat", chat_id="7")
        await agent.handle("done!", chat_id="42")
        await agent.handle("and now?", chat_id="42")

        other, first, second = (c["context"] for c in fake_backend.turn_calls)
        assert "call Alex" not in other
        assert "- ⏰ Reminder: call Alex" in first
        assert "call Alex" not in second
        agent.close()

    @pytest.mark.asyncio
    async def test_reminder_note_survives_a_failed_turn(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend, platform="telegram")
        agent.record_reminder_fired("⏰ Reminder: call Alex", "42")
        fake_backend.turn_error = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await agent.handle("hi", chat_id="42")
        fake_backend.turn_error = None
        await agent.handle("hi again", chat_id="42")
        assert "call Alex" in fake_backend.turn_calls[-1]["context"]
        agent.close()


class TestBackgroundConsolidation:
    @pytest.mark.asyncio
    async def test_runs_after_the_reply_without_blocking_it(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend)
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_consolidate(**kwargs):
            started.set()
            await release.wait()
            return True

        agent.consolidator.consolidate = AsyncMock(side_effect=slow_consolidate)

        await agent.handle("hi")  # returns while consolidation is still pending
        await asyncio.wait_for(started.wait(), 1)
        await agent.handle("again")  # one already running: no second one
        release.set()
        await agent.consolidator._task
        assert agent.consolidator.consolidate.await_count == 1
        agent.close()

    @pytest.mark.asyncio
    async def test_failure_is_logged_not_raised(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _chat_agent(cfg, fake_backend)
        agent.consolidator.consolidate = AsyncMock(side_effect=RuntimeError("API down"))
        reply, _ = await agent.handle("hi")
        await agent.consolidator._task  # does not raise
        assert reply
        agent.close()

    @pytest.mark.asyncio
    async def test_aclose_cancels_a_running_consolidation(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend)
        async def forever(**kwargs):
            await asyncio.sleep(60)

        agent.consolidator.consolidate = AsyncMock(side_effect=forever)
        await agent.handle("hi")
        await asyncio.sleep(0)
        task = agent.consolidator._task
        await agent.aclose()
        assert task.cancelled()


# ────────────────────────────────────────────────────────────────────
# Spec #9: Startup and Background Sync
# ────────────────────────────────────────────────────────────────────

class TestSyncOptimization:
    @pytest.mark.asyncio
    async def test_start_calls_sync(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """start() should call index.sync() once."""
        agent = _make_agent(cfg, fake_backend)
        agent.index.sync = AsyncMock(return_value=False)

        await agent.start()
        agent.index.sync.assert_called_once()
        agent.close()

    @pytest.mark.asyncio
    async def test_start_without_backend_skips_lifecycle(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """start(include_backend=False) should sync without backend startup."""
        agent = _make_agent(cfg, fake_backend)
        agent.index.sync = AsyncMock(return_value=False)
        fake_backend.on_agent_start = AsyncMock()
        fake_backend.on_agent_shutdown = AsyncMock()

        await agent.start(include_backend=False)
        agent.index.sync.assert_called_once()
        fake_backend.on_agent_start.assert_not_called()
        assert not agent._backend_started

        await agent.aclose()
        fake_backend.on_agent_shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_background_sync_creates_task(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """start_background_sync() should create an asyncio task."""
        import asyncio
        agent = _make_agent(cfg, fake_backend)
        agent.index.sync = AsyncMock(return_value=False)

        await agent.start_background_sync(interval=1)
        assert hasattr(agent, "_sync_task")
        assert isinstance(agent._sync_task, asyncio.Task)

        agent._sync_task.cancel()
        try:
            await agent._sync_task
        except asyncio.CancelledError:
            pass
        agent.close()


# ────────────────────────────────────────────────────────────────────
# Filesystem Guardrail (now in ToolExecutor)
# ────────────────────────────────────────────────────────────────────

class TestFilesystemGuardrail:
    @pytest.mark.asyncio
    async def test_allows_write_inside_memory_dir(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """file_write to a path under memory_dir should succeed."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": "todos.md", "content": "- Buy milk"},
        )
        assert "File written" in result
        assert (cfg.memory_dir / "todos.md").exists()
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_write_outside_memory_dir(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """file_write to a path outside memory_dir should be blocked."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": "/tmp/evil.md", "content": "bad"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_write_to_home_dir(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """file_write to ~/something.md should be blocked."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": str(Path.home() / "todos.md"), "content": "bad"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_path_traversal(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """Path traversal attempts (../../etc) should be blocked."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": str(cfg.memory_dir / ".." / ".." / "etc" / "passwd"), "content": "bad"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_allows_nested_path_inside_memory_dir(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """Writing to a subdirectory of memory_dir should work."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": "subdir/file.md", "content": "nested"},
        )
        assert "File written" in result
        assert (cfg.memory_dir / "subdir" / "file.md").exists()
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_read_outside(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """file_read outside memory_dir should be blocked."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_read",
            {"file_path": "/etc/hosts"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_file_read_returns_content(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """file_read should return content for files inside memory_dir."""
        agent = _make_agent(cfg, fake_backend)

        (cfg.memory_dir / "test.md").write_text("hello world")
        result = await agent._tools.execute(
            "file_read",
            {"file_path": "test.md"},
        )
        assert result == "hello world"
        agent.close()


class TestSandboxedFileTools:
    def test_tool_definitions_contain_file_tools(self):
        """TOOL_DEFINITIONS should include file_write and file_read."""
        from memclaw.tools import TOOL_DEFINITIONS

        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert "file_write" in names
        assert "file_read" in names

    def test_tool_definitions_contain_all_tools(self):
        """TOOL_DEFINITIONS should expose the full catalog."""
        from memclaw.tools import TOOL_DEFINITIONS

        names = [t["name"] for t in TOOL_DEFINITIONS]
        expected = {
            "memory_save", "memory_search",
            "image_save", "image_search", "image_send",
            "update_instructions", "file_write", "file_read",
            "reminder_create", "reminder_list", "reminder_cancel",
        }
        assert set(names) == expected
        assert len(names) == len(expected)


# ────────────────────────────────────────────────────────────────────
# Resilience and concurrency
# ────────────────────────────────────────────────────────────────────

class TestResilience:
    @pytest.mark.asyncio
    async def test_context_search_failure_does_not_fail_the_turn(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        fake_backend.turn_response = TurnResult(text="hi")
        agent = _make_agent(cfg, fake_backend)
        agent.consolidator.consolidate = AsyncMock(return_value=False)
        agent.search.search = AsyncMock(side_effect=RuntimeError("OpenAI down"))
        reply, _ = await agent.handle("hello")
        assert reply == "hi"
        assert fake_backend.turn_calls[0]["context"].endswith("</context>")
        agent.close()

    @pytest.mark.asyncio
    async def test_start_survives_index_sync_failure(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _make_agent(cfg, fake_backend)
        agent.index.sync = AsyncMock(side_effect=RuntimeError("OpenAI down"))
        await agent.start()
        await agent.aclose()

    @pytest.mark.asyncio
    async def test_memory_save_reports_success_when_indexing_fails(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _make_agent(cfg, fake_backend)
        agent.index.index_file = AsyncMock(side_effect=RuntimeError("OpenAI down"))
        out = await agent._tools.execute("memory_save", {"content": "Max is a beagle"})
        assert out.startswith("Memory saved to")
        assert "Max is a beagle" in cfg.daily_file().read_text()
        agent.close()


class TestConcurrentChats:
    @pytest.mark.asyncio
    async def test_tool_calls_see_their_own_chat(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """Two chats' turns overlapping must not mix up per-turn tool state."""
        seen: dict[str, str | None] = {}
        agent = _chat_agent(cfg, fake_backend, platform="slack")

        async def run_turn(request, tool_executor):
            key = request.session_key
            await asyncio.sleep(0.05)  # the other chat's message arrives meanwhile
            seen[key] = tool_executor.context.chat_id
            return TurnResult(text="ok", session_id=f"s-{key}")

        fake_backend.run_turn = run_turn
        await asyncio.gather(
            agent.handle("a", chat_id="A"),
            agent.handle("b", chat_id="B"),
        )
        assert seen == {"slack:A": "A", "slack:B": "B"}
        agent.close()


class TestInternalFilesGuardrail:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("target", [
        "memclaw.db", "memclaw.db-wal", ".env", "sessions.json", "meta.json",
        ".cursor/hooks/hook_policy.json", "whatsapp/session.db", "bot.log",
    ])
    async def test_internal_files_are_off_limits(self, cfg: MemclawConfig, fake_backend: FakeBackend, target):
        agent = _make_agent(cfg, fake_backend)
        for tool, args in (
            ("file_write", {"file_path": target, "content": "x"}),
            ("file_read", {"file_path": target}),
        ):
            out = await agent._tools.execute(tool, args)
            assert out.startswith("Blocked"), (tool, target, out)
        agent.close()

    @pytest.mark.asyncio
    async def test_written_markdown_is_indexed(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _make_agent(cfg, fake_backend)
        await agent._tools.execute("file_write", {"file_path": "todos.md", "content": "- renew passport"})
        results = await agent.search.search("renew passport", limit=3)
        assert results and "passport" in results[0].content
        agent.close()


class TestImageSearchPlatform:
    @pytest.mark.asyncio
    async def test_only_this_platforms_images_are_returned(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _make_agent(cfg, fake_backend, platform="telegram")
        await agent.index.store_platform_image("whatsapp", "wa-1", "sprint planning whiteboard")
        await agent.index.store_platform_image("telegram", "tg-1", "sprint planning whiteboard")
        out = await agent._tools.execute("image_search", {"query": "sprint planning whiteboard"})
        assert "id 2 " in out and "id 1 " not in out
        assert agent._found_images == []  # searching sends nothing
        out = await agent._tools.execute("image_send", {"ids": [1, 2]})
        assert [img["media_ref"] for img in agent._found_images] == ["tg-1"]
        assert "No stored image with id 1" in out
        agent.close()

    @pytest.mark.asyncio
    async def test_an_image_is_sent_once_per_reply(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _make_agent(cfg, fake_backend, platform="telegram")
        await agent.index.store_platform_image("telegram", "tg-passport", "passport")
        await agent.index.store_platform_image("telegram", "tg-id", "ID card printout")
        await agent._tools.execute("image_send", {"ids": [1, 1]})
        await agent._tools.execute("image_send", {"ids": [1, 2]})  # e.g. after a second search
        assert [img["media_ref"] for img in agent._found_images] == ["tg-passport", "tg-id"]
        agent.close()

    @pytest.mark.asyncio
    async def test_unknown_ids_send_nothing(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _make_agent(cfg, fake_backend, platform="telegram")
        out = await agent._tools.execute("image_send", {"ids": [99]})
        assert out == "Nothing to send. No stored image with id 99."
        assert agent._found_images == []
        agent.close()


class TestMemoryMdIsWrittenOnlyByConsolidation:
    def test_memory_save_has_no_permanent_option(self):
        from memclaw.tools import TOOL_DEFINITIONS

        schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "memory_save")["input_schema"]
        assert "permanent" not in schema["properties"]

    @pytest.mark.asyncio
    async def test_a_permanent_flag_from_an_old_session_is_ignored(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _make_agent(cfg, fake_backend)
        before = cfg.memory_file.read_text()
        out = await agent._tools.execute(
            "memory_save", {"content": "Daisy Disk serial ABC-123", "permanent": True},
        )
        assert out.startswith(f"Memory saved to {cfg.daily_file().name}")
        assert cfg.memory_file.read_text() == before
        assert "Daisy Disk serial ABC-123" in cfg.daily_file().read_text()
        agent.close()

class TestReviewFixes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("target", ["MEMORY.md", "memory/2026-10-15.md"])
    async def test_file_write_cannot_replace_memory_or_daily_notes(
        self, cfg: MemclawConfig, fake_backend: FakeBackend, target,
    ):
        agent = _make_agent(cfg, fake_backend)
        before = cfg.memory_file.read_text()
        out = await agent._tools.execute("file_write", {"file_path": target, "content": "x"})
        assert out.startswith("Blocked")
        assert cfg.memory_file.read_text() == before
        assert not (cfg.memory_dir / "memory" / "2026-10-15.md").exists()
        read = await agent._tools.execute("file_read", {"file_path": "MEMORY.md"})
        assert read == before
        agent.close()

    @pytest.mark.asyncio
    async def test_image_save_writes_nothing_when_the_embedding_fails(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _make_agent(cfg, fake_backend, platform="telegram")
        agent.index.get_embedding = AsyncMock(side_effect=RuntimeError("OpenAI down"))
        out = await agent._tools.execute(
            "image_save", {"description": "a whiteboard", "media_ref": "tg-1"},
        )
        assert out.startswith("Error")
        assert not cfg.daily_file().exists()
        agent.close()

    @pytest.mark.asyncio
    async def test_file_write_indexes_the_path_sync_uses(self, tmp_path: Path, fake_backend: FakeBackend):
        (tmp_path / "real").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "real")
        cfg = MemclawConfig(memory_dir=tmp_path / "link", openai=OpenAISettings(api_key="k"), claude=ClaudeSettings(api_key="k"))
        agent = _make_agent(cfg, fake_backend)
        await agent.index.sync()
        await agent._tools.execute("file_write", {"file_path": "todos.md", "content": "- milk"})
        assert not await agent.index.sync()  # nothing left to re-index or remove
        paths = {r[0] for r in agent.index.db.execute("SELECT DISTINCT file_path FROM chunks")}
        assert str(cfg.memory_dir / "todos.md") in paths

class TestNewDuringReset:
    @pytest.mark.asyncio
    async def test_message_right_after_new_does_not_resume_the_old_session(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        """A message that runs while /new is still disconnecting the backend
        must start fresh, not resume (and re-save) the old session."""
        agent = _chat_agent(cfg, fake_backend, platform="slack")
        await agent.handle("hi", chat_id="C1")
        seen: list[str | None] = []

        async def slow_reset(key):
            # The next message arrives while the reset is in progress.
            await agent.handle("next", chat_id="C1")
            seen.append(fake_backend.turn_calls[-1]["resume_session_id"])

        fake_backend.reset_session = slow_reset
        await agent.reset_conversation("C1")
        assert seen == [None]
        agent.close()
