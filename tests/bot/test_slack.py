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
    handlers.config.slack.allowed_channels_list = []
    handlers.config.slack.allowed_users_list = []
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
