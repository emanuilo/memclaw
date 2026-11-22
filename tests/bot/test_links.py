"""memclaw.bot.links: URLs in a message, and what a failed summary says."""
from __future__ import annotations

import pytest

from memclaw.bot.links import LinkProcessor


class _FailingOpenAI:
    async def chat(self, messages):
        raise RuntimeError("quota")


def test_extract_urls():
    links = LinkProcessor(_FailingOpenAI())
    text = "see https://a.com/x, and (https://b.org/y). again: https://a.com/x!"
    assert links.extract_urls(text) == ["https://a.com/x", "https://b.org/y"]


@pytest.mark.asyncio
async def test_failed_summary_is_reported_as_the_summary():
    """The error text becomes the summary the agent sees (as before)."""
    links = LinkProcessor(_FailingOpenAI())
    assert await links.summarize("page", "https://a.com") == "Could not summarize: quota"
