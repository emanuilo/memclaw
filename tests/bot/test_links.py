"""memclaw.bot.links: URLs in a message, and what a failed summary gives."""
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
async def test_failed_summary_is_no_summary():
    """None, as for a page that couldn't be fetched, so no link note is added."""
    links = LinkProcessor(_FailingOpenAI())
    assert await links.summarize("page", "https://a.com") is None
