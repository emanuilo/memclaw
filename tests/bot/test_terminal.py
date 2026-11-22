"""The terminal chat: typed lines go to the agent, its replies are printed,
`/new` resets, `/quit` (or Ctrl-D) ends it."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from memclaw.bot import DEFAULT_PLATFORM, PLATFORMS, chosen_platform
from memclaw.bot import terminal


@pytest.fixture
def screen(monkeypatch):
    """The terminal's console, recording what is printed."""
    console = Console(record=True, width=80, color_system=None)
    monkeypatch.setattr(terminal, "console", console)
    return console


def _typed(monkeypatch, screen: Console, *lines: str) -> None:
    """Make the prompt return *lines* in turn, then Ctrl-D."""
    queue = list(lines)

    def _input(_prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr(screen, "input", _input)


def _agent(reply: str = "Noted.") -> MagicMock:
    agent = MagicMock()
    agent.handle = AsyncMock(return_value=(reply, []))
    agent.reset_conversation = AsyncMock()
    return agent


@pytest.mark.asyncio
async def test_a_message_goes_to_the_agent_and_its_reply_is_printed(monkeypatch, screen):
    _typed(monkeypatch, screen, "  remember the milk  ", "", "/quit", "never read")
    agent = _agent("**Saved** the milk.")

    await terminal.chat(agent)

    agent.handle.assert_awaited_once_with("remember the milk")
    assert "Saved the milk." in screen.export_text()


@pytest.mark.asyncio
async def test_new_starts_a_fresh_conversation(monkeypatch, screen):
    _typed(monkeypatch, screen, "/NEW")
    agent = _agent()

    await terminal.chat(agent)

    agent.reset_conversation.assert_awaited_once_with()
    agent.handle.assert_not_awaited()
    assert "Started a new conversation." in screen.export_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("word", ["/quit", "/exit", "quit", "EXIT"])
async def test_quit_words_end_the_chat(monkeypatch, screen, word):
    _typed(monkeypatch, screen, word, "not sent")
    agent = _agent()

    await terminal.chat(agent)

    agent.handle.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_turn_is_shown_and_the_chat_goes_on(monkeypatch, screen):
    _typed(monkeypatch, screen, "first", "second")
    agent = _agent()
    agent.handle.side_effect = [RuntimeError("backend down"), ("ok", [])]

    await terminal.chat(agent)

    assert agent.handle.await_count == 2
    assert "Error: backend down" in screen.export_text()


def test_the_terminal_is_a_registered_platform():
    assert PLATFORMS["terminal"].label == "Terminal chat"
    assert DEFAULT_PLATFORM == "terminal"


@pytest.mark.parametrize("configured, started", [
    ("", "terminal"), ("terminal", "terminal"), ("slack", "slack"), ("nonsense", "terminal"),
])
def test_chosen_platform_falls_back_to_the_terminal(configured, started):
    config = MagicMock(platform=configured)
    assert chosen_platform(config).name == started
