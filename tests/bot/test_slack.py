"""Slack: the /new command. (What Slack shares with the other platforms is
tested in test_parity.py.)"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_new_message_resets_the_channel():
    from memclaw.bot.inbound import Inbound
    from memclaw.bot.slack import SlackBot

    handlers = SlackBot.__new__(SlackBot)
    handlers.config = MagicMock()
    handlers.config.slack.allowed_channels_list = ["C1"]
    handlers.config.slack.allowed_users_list = ["U1"]
    handlers.inbound = Inbound.__new__(Inbound)
    handlers.inbound.agent = agent = MagicMock()
    agent.reset_conversation = AsyncMock()
    agent.handle = AsyncMock()
    say = AsyncMock()

    await handlers._route_event(
        {"channel": "C1", "user": "U1", "text": "<@UBOT> /new", "ts": "1.0"}, say,
    )

    agent.reset_conversation.assert_awaited_once_with("C1")
    agent.handle.assert_not_awaited()
    say.assert_awaited_once_with(text="Started a new conversation.", thread_ts="1.0")


@pytest.mark.parametrize(("users", "channels", "warned", "shown"), [
    ("", "C1", True, "allowed users: nobody; allowed channels: C1"),
    ("U1", "", False, "allowed users: **; allowed channels: any"),
    ("", "", True, "allowed users: nobody; allowed channels: any"),
    ("U1", "C1", False, "allowed users: **; allowed channels: C1"),
])
def test_start_warns_only_about_an_empty_user_list(monkeypatch, tmp_path, users, channels, warned, shown):
    from memclaw.bot import console, slack
    from memclaw.config import MemclawConfig, SlackSettings

    monkeypatch.setattr(slack, "log_to", lambda *a: None)
    bot = MagicMock(start=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr(slack, "SlackBot", lambda config: bot)
    config = MemclawConfig(memory_dir=tmp_path, slack=SlackSettings(bot_token="xoxb-1", app_token="xapp-1"))
    # Set after construction: an empty value there would be read from the environment.
    config.slack.allowed_users, config.slack.allowed_channels = users, channels
    with console.capture() as captured:
        slack.run(config)
    out = " ".join(captured.get().split())

    assert ("SLACK_ALLOWED_USERS is empty, so nobody can use the bot" in out) == warned
    assert "SLACK_ALLOWED_CHANNELS" not in out
    assert shown in out
    bot.aclose.assert_awaited_once()  # the async shutdown ran
