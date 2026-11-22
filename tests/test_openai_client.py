"""memclaw.openai_client: the calls Memclaw makes, and the doctor's probe."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from memclaw import openai_client
from memclaw.openai_client import OpenAIClient, probe_openai


class _SDK:
    """Records calls; whisper fails with *whisper_status* if set."""

    def __init__(self, *, api_key, whisper_status=None):
        self.calls: list[tuple] = []
        self.whisper_status = whisper_status
        self.embeddings = SimpleNamespace(create=self._embed)
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._transcribe))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.closed = False

    async def _embed(self, **kw):
        self.calls.append(("embed", kw))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, 1.0]) for _ in kw["input"]])

    async def _transcribe(self, **kw):
        self.calls.append(("transcribe", kw))
        if self.whisper_status:
            raise type("APIError", (Exception,), {"status_code": self.whisper_status})("nope")
        return SimpleNamespace(text="hello")

    async def _chat(self, **kw):
        self.calls.append(("chat", kw))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    async def close(self):
        self.closed = True


@pytest.fixture
def sdk(monkeypatch):
    made: list[_SDK] = []

    def factory(**kw):
        made.append(_SDK(**kw, whisper_status=getattr(factory, "whisper_status", None)))
        return made[-1]

    monkeypatch.setattr(openai_client, "AsyncOpenAI", factory)
    factory.made = made
    return factory


@pytest.mark.asyncio
async def test_calls(sdk):
    client = OpenAIClient("sk-1")
    assert await client.embed(["a", "b"], model="m") == [[0.5, 1.0], [0.5, 1.0]]
    assert await client.transcribe(b"ogg", filename="voice.ogg", mime_type="audio/ogg") == "hello"
    assert await client.chat([{"role": "user", "content": "hi"}]) == ""
    assert sdk.made[0].calls == [
        ("embed", {"model": "m", "input": ["a", "b"]}),
        ("transcribe", {"model": "whisper-1", "file": ("voice.ogg", b"ogg", "audio/ogg")}),
        ("chat", {"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}],
                  "reasoning_effort": "low", "max_completion_tokens": 300}),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("status, whisper_ok", [(None, True), (400, True), (401, False)])
async def test_probe(sdk, status, whisper_ok):
    """A 400 from Whisper on the silent probe file still proves access."""
    sdk.whisper_status = status
    report = await probe_openai("sk-1")
    assert [o.ok for o in report] == [True, True, whisper_ok]
    assert report.all_ok is whisper_ok
    assert sdk.made[0].closed
    assert [name for name, _ in sdk.made[0].calls] == ["embed", "chat", "transcribe"]
