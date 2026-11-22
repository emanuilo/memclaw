"""One set of scenarios run through Telegram, Slack and WhatsApp.

Each platform is driven at its SDK boundary with fakes (the Telegram Update,
the Slack event + `say`, the WhatsApp MessageEv + client) and a fake agent,
OpenAI client and web page. The tests pin what the agent receives (prompt,
image, chat id) and what the user sees, so the three platforms provably run
the same message pipeline, and the platform quirks that differ are written
down here on purpose.
"""
from __future__ import annotations

import base64
import importlib
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from memclaw.config import MemclawConfig, OpenAISettings, SlackSettings, TelegramSettings
from memclaw.reminders import ReminderScheduler

URL = "https://example.com/post"
PAGE_TEXT = "The page text."
SUMMARY = "A post about testing."
LINK_TEXT = f"read {URL} later"
NOTE = (
    f"\n[Link summary] {URL}: {SUMMARY}"
    "\nThis summary has NOT been saved yet. Save it if the content is worth remembering."
)
VOICE_TEXT = f"call mom about {URL}"
VOICE_PROMPT = (
    f"[Voice message] {VOICE_TEXT}"
    "\nThis transcription has NOT been saved yet. Save it if the content is worth remembering."
    f"{NOTE}"
)
TRANSCRIPTION_FAILED = (
    "I couldn't transcribe that voice message. "
    "If you set up Memclaw recently, run `memclaw doctor` to check "
    "your OpenAI key has access to the whisper-1 model."
)
PHOTO = b"\x89PNG fake image bytes"
AUDIO = b"OggS fake audio bytes"
SUMMARIZE_MESSAGES = [
    {"role": "system", "content": (
        "Summarize this web content concisely. Focus on main points. Keep it under 50 tokens."
    )},
    {"role": "user", "content": f"Summarize content from {URL}:\n\n{PAGE_TEXT}"},
]

pytestmark = pytest.mark.asyncio

# ── Where the code lives: the only part that differs between layouts ──

TELEGRAM = "memclaw.bot.telegram"
SLACK = "memclaw.bot.slack"
WHATSAPP = "memclaw.bot.whatsapp"
LINKS = "memclaw.bot.links"
# The modules that construct MemclawAgent (replaced by FakeAgent).
AGENT_HOMES = ["memclaw.bot.inbound"]


def make_bot(module, config, openai):
    """Construct the platform's bot object the way its launcher does."""
    cls = {TELEGRAM: "TelegramBot", SLACK: "SlackBot", WHATSAPP: "WhatsAppBot"}
    return getattr(module, cls[module.__name__])(config)


def attach_telegram(bot, tg_bot):
    """What post_init does once python-telegram-bot has a Bot."""
    bot.attach_bot(tg_bot)


def install_openai(monkeypatch, fake):
    """Make *fake* the OpenAI SDK client the bots use."""
    monkeypatch.setattr("memclaw.openai_client.AsyncOpenAI", lambda **kwargs: fake)


# ── Fakes shared by every platform ───────────────────────────────────────


class FakeAgent:
    instances: list["FakeAgent"] = []

    def __init__(self, config, platform=None, *, scheduler=None, backend=None):
        self.platform = platform
        self.scheduler = scheduler
        self.calls: list[dict] = []
        self.fired: list[tuple] = []
        self.resets: list = []
        self.reply: tuple[str, list[dict]] = ("Got it.", [])
        FakeAgent.instances.append(self)

    async def handle(self, message, *, image_b64=None, image_media_type="image/jpeg",
                     chat_id=None, on_tool=None):
        self.calls.append({
            "prompt": message, "image_b64": image_b64,
            "image_media_type": image_media_type, "chat_id": chat_id,
        })
        return self.reply

    def record_reminder_fired(self, text, chat_id=None):
        self.fired.append((text, chat_id))

    async def reset_conversation(self, chat_id=None):
        self.resets.append(chat_id)

    async def start(self, **kw):
        pass

    async def start_background_sync(self, interval=60):
        pass

    async def aclose(self):
        pass

    def close(self):
        pass


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.transcriptions: list[tuple] = []
        self.chats: list[dict] = []
        self.fail_transcription = False
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._transcribe))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    async def _transcribe(self, *, model, file):
        self.transcriptions.append((model, file))
        if self.fail_transcription:
            raise RuntimeError("model not available")
        return SimpleNamespace(text=VOICE_TEXT)

    async def _chat(self, **kwargs):
        self.chats.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=SUMMARY))])

    async def close(self):
        pass


async def fake_fetch(self, url, timeout=10.0):
    return PAGE_TEXT if url == URL else None


@pytest.fixture
def env(tmp_path, monkeypatch):
    FakeAgent.instances.clear()
    for module in AGENT_HOMES:
        monkeypatch.setattr(f"{module}.MemclawAgent", FakeAgent)
    monkeypatch.setattr(f"{LINKS}.LinkProcessor.fetch_content", fake_fetch)
    openai = FakeOpenAI()
    install_openai(monkeypatch, openai)
    config = MemclawConfig(
        memory_dir=tmp_path / "m",
        openai=OpenAISettings(api_key="sk-test"),
        telegram=TelegramSettings(bot_token="123:abc", allowed_user_ids="1"),
        slack=SlackSettings(bot_token="xoxb-1", app_token="xapp-1", allowed_users="U1"),
    )
    return SimpleNamespace(config=config, openai=openai, monkeypatch=monkeypatch)


# ── Telegram driver ──────────────────────────────────────────────────────


class FakeTelegramBot:
    def __init__(self):
        self.sent: list[dict] = []
        self.photos: list[dict] = []
        self.files: dict[str, bytes] = {}

    async def send_chat_action(self, **kw):
        pass

    async def send_message(self, *args, **kw):
        self.sent.append({"args": args, **kw})
        return SimpleNamespace(edit_text=_anoop, delete=_anoop)

    async def send_photo(self, **kw):
        self.photos.append(kw)

    async def get_file(self, file_id):
        data = self.files[file_id]

        async def download_as_bytearray():
            return bytearray(data)
        return SimpleNamespace(download_as_bytearray=download_as_bytearray)


async def _anoop(*a, **kw):
    pass


class TelegramDriver:
    name = "telegram"
    chat_id = "42"

    def __init__(self, env):
        self.env = env
        self.bot = make_bot(importlib.import_module(TELEGRAM), env.config, env.openai)
        self.tg = FakeTelegramBot()
        attach_telegram(self.bot, self.tg)
        self.replies: list[str] = []
        self.reply_kwargs: list[dict] = []
        self.markdown_fails = False

    @property
    def agent(self) -> FakeAgent:
        return FakeAgent.instances[-1]

    def _update(self, *, user=1, text=None, caption=None, photo=None, voice=None, reply_to=None):
        async def reply_text(text, **kw):
            if self.markdown_fails and kw.get("parse_mode"):
                raise ValueError("Can't parse entities")
            self.replies.append(text)
            self.reply_kwargs.append(kw)

        message = SimpleNamespace(
            text=text, caption=caption, photo=photo, voice=voice, chat_id=int(self.chat_id),
            reply_to_message=reply_to, reply_text=reply_text,
        )
        return SimpleNamespace(
            effective_user=SimpleNamespace(id=user),
            effective_chat=SimpleNamespace(id=int(self.chat_id)),
            message=message,
        )

    def _context(self):
        return SimpleNamespace(bot=self.tg, args=[])

    async def text(self, text, *, stranger=False):
        await self.bot.handle_text(self._update(user=7 if stranger else 1, text=text), self._context())

    async def photo(self, caption):
        self.tg.files["PHOTO1"] = PHOTO
        photo = [SimpleNamespace(file_id="small"), SimpleNamespace(file_id="PHOTO1")]
        await self.bot.handle_photo(self._update(photo=photo, caption=caption), self._context())

    async def voice(self):
        self.tg.files["VOICE1"] = AUDIO
        await self.bot.handle_voice(self._update(voice=SimpleNamespace(file_id="VOICE1")), self._context())

    async def new(self):
        await self.bot.new_command(self._update(text="/new"), self._context())

    def delivered(self):
        return [(m["chat_id"], m["text"]) for m in self.tg.sent]


# ── Slack driver ─────────────────────────────────────────────────────────


class FakeSlackApp:
    def __init__(self, token=None, **kw):
        self.token = token
        self.events: dict = {}
        self.posted: list[dict] = []
        self.uploads: list[dict] = []
        self.client = SimpleNamespace(chat_postMessage=self._post, files_upload_v2=self._upload)

    def event(self, name):
        def register(fn):
            self.events[name] = fn
            return fn
        return register

    async def _post(self, **kw):
        self.posted.append(kw)

    async def _upload(self, **kw):
        self.uploads.append(kw)
        return {"ok": True}


class FakeHttpx:
    """Stands in for httpx.AsyncClient (the Slack file download)."""

    content = PHOTO
    fail = False
    calls: list = []

    def __init__(self, *, headers=None, timeout=None):
        FakeHttpx.calls.append(headers)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if FakeHttpx.fail:
            raise RuntimeError("404")
        return SimpleNamespace(content=FakeHttpx.content, raise_for_status=lambda: None)


class SlackDriver:
    name = "slack"
    chat_id = "C1"

    def __init__(self, env):
        self.env = env
        env.monkeypatch.setattr(f"{SLACK}.AsyncApp", FakeSlackApp)
        env.monkeypatch.setattr("httpx.AsyncClient", FakeHttpx)
        FakeHttpx.fail = False
        FakeHttpx.calls = []
        self.bot = make_bot(importlib.import_module(SLACK), env.config, env.openai)
        self.app = _only_instance_of(self.bot, FakeSlackApp)
        self.said: list[dict] = []

    @property
    def replies(self) -> list[str]:
        return [s["text"] for s in self.said]

    @property
    def agent(self) -> FakeAgent:
        return FakeAgent.instances[-1]

    async def _event(self, event, kind="message"):
        async def say(**kw):
            self.said.append(kw)
        await self.app.events[kind](event=event, say=say, client=None)

    async def text(self, text, *, stranger=False, user=None):
        user = user or ("U9" if stranger else "U1")
        await self._event({"channel": "C1", "user": user, "text": text, "ts": "1.0"})

    async def photo(self, caption):
        FakeHttpx.content = PHOTO
        await self._event({
            "channel": "C1", "user": "U1", "text": f"<@UBOT> {caption}", "ts": "1.0",
            "files": [{"mimetype": "image/png", "name": "p.png", "url_private_download": "https://f/p"}],
        }, kind="app_mention")

    async def voice(self):
        FakeHttpx.content = AUDIO
        await self._event({
            "channel": "C1", "user": "U1", "text": "", "ts": "1.0",
            "files": [{"mimetype": "audio/mp4", "name": "a.m4a", "url_private": "https://f/a"}],
        })

    async def new(self):
        await self._event({"channel": "C1", "user": "U1", "text": "<@UBOT> /new", "ts": "1.0"})

    def delivered(self):
        return [(m["channel"], m["text"]) for m in self.app.posted]


def _only_instance_of(obj, cls):
    """The one attribute of *obj* (one level deep) that is a *cls*."""
    seen = [v for v in vars(obj).values() if isinstance(v, cls)]
    for v in vars(obj).values():
        if hasattr(v, "__dict__") and not isinstance(v, type):
            seen += [w for w in vars(v).values() if isinstance(w, cls)]
    assert len(seen) == 1, seen
    return seen[0]


# ── WhatsApp driver (neonize is replaced by fakes: it needs libmagic) ────


class FakeNeonizeClient:
    def __init__(self, path):
        self.path = path
        self.events: dict = {}
        self.sent: list[tuple] = []
        self.replied: list[str] = []
        self.images: list[tuple] = []
        self.download: bytes | Exception = PHOTO

    def event(self, ev_type):
        def register(fn):
            self.events[ev_type] = fn
            return fn
        return register

    async def send_message(self, jid, text):
        self.sent.append((jid, text))

    async def reply_message(self, text, ev):
        self.replied.append(text)

    async def download_any(self, message):
        if isinstance(self.download, Exception):
            raise self.download
        return self.download

    async def send_image(self, chat, path, caption=None):
        self.images.append((chat, path, caption))


def _ensure_neonize_importable():
    """neonize needs the libmagic system library; without it, stand in
    placeholder modules so the WhatsApp module imports (the tests replace
    the client anyway)."""
    try:
        import neonize.aioze.client  # noqa: F401
        import neonize.aioze.events  # noqa: F401
        import neonize.utils.jid  # noqa: F401
    except ImportError:
        names = ["neonize", "neonize.aioze", "neonize.aioze.client", "neonize.aioze.events",
                 "neonize.utils", "neonize.utils.jid"]
        stubs = {name: types.ModuleType(name) for name in names}
        stubs["neonize.aioze.client"].NewAClient = FakeNeonizeClient
        for event in ("ConnectedEv", "MessageEv", "PairStatusEv"):
            setattr(stubs["neonize.aioze.events"], event, type(event, (), {}))
        stubs["neonize.utils.jid"].build_jid = None
        sys.modules.update(stubs)


_ensure_neonize_importable()


def _fields(present: bool):
    return lambda: [1] if present else []


def _wa_event(*, text="", image=None, audio=None, from_me=True, chat_user="111", group=False):
    message = SimpleNamespace(
        conversation=text,
        extendedTextMessage=SimpleNamespace(text=""),
        imageMessage=SimpleNamespace(
            ListFields=_fields(image is not None),
            caption=(image or {}).get("caption", ""), mimetype=(image or {}).get("mimetype", ""),
        ),
        audioMessage=SimpleNamespace(
            ListFields=_fields(audio is not None), mimetype=(audio or {}).get("mimetype", ""),
        ),
    )
    chat = SimpleNamespace(User=chat_user, Server="s.whatsapp.net")
    source = SimpleNamespace(
        IsGroup=group, IsFromMe=from_me, Chat=chat,
        Sender=SimpleNamespace(User="111" if from_me else chat_user),
    )
    return SimpleNamespace(Info=SimpleNamespace(MessageSource=source), Message=message)


class WhatsAppDriver:
    name = "whatsapp"
    chat_id = "111@s.whatsapp.net"

    def __init__(self, env):
        self.env = env
        self.module = importlib.import_module(WHATSAPP)
        env.monkeypatch.setattr(self.module, "NewAClient", FakeNeonizeClient)
        env.monkeypatch.setattr(self.module, "build_jid", lambda user, server: f"jid:{user}@{server}")
        self.bot = make_bot(self.module, env.config, env.openai)
        self.client = _only_instance_of(self.bot, FakeNeonizeClient)

    @property
    def agent(self) -> FakeAgent:
        return FakeAgent.instances[-1]

    @property
    def replies(self) -> list[str]:
        return self.client.replied + [text for _, text in self.client.sent]

    async def _message(self, ev):
        await self.client.events[self.module.MessageEv](self.client, ev)

    async def text(self, text, *, stranger=False):
        await self._message(_wa_event(text=text, from_me=not stranger, chat_user="222" if stranger else "111"))

    async def photo(self, caption):
        self.client.download = PHOTO
        await self._message(_wa_event(image={"caption": caption, "mimetype": "image/jpeg"}))

    async def voice(self):
        self.client.download = AUDIO
        await self._message(_wa_event(audio={"mimetype": "audio/ogg; codecs=opus"}))

    async def new(self):
        await self._message(_wa_event(text=" /NEW "))

    def delivered(self):
        return [(jid, text) for jid, text in self.client.sent]


DRIVERS = [TelegramDriver, SlackDriver, WhatsAppDriver]


@pytest.fixture(params=DRIVERS, ids=lambda d: d.name)
def chat(request, env):
    return request.param(env)


# ── The same scenarios on every platform ─────────────────────────────────


async def test_plain_text(chat):
    await chat.text("remember the milk")
    assert chat.agent.calls == [{
        "prompt": "remember the milk", "image_b64": None,
        "image_media_type": "image/jpeg", "chat_id": chat.chat_id,
    }]
    assert chat.replies == ["Got it."]


async def test_text_with_a_link(chat):
    await chat.text(LINK_TEXT)
    assert [c["prompt"] for c in chat.agent.calls] == [f"{LINK_TEXT}\n{NOTE}"]
    assert chat.env.openai.chats == [{
        "model": "gpt-5-mini", "messages": SUMMARIZE_MESSAGES,
        "reasoning_effort": "low", "max_completion_tokens": 300,
    }]
    assert chat.replies == ["Got it."]


async def test_link_that_cannot_be_fetched_adds_nothing(chat):
    await chat.text("see https://example.com/elsewhere")
    assert [c["prompt"] for c in chat.agent.calls] == ["see https://example.com/elsewhere"]
    assert chat.env.openai.chats == []


async def test_photo_with_a_link_in_the_caption(chat):
    await chat.photo(LINK_TEXT)
    [call] = chat.agent.calls
    media_ref = {
        "telegram": "PHOTO1",
        "slack": str(chat.env.config.slack_media_dir),
        "whatsapp": str(chat.env.config.whatsapp_media_dir),
    }[chat.name]
    prefix = f"User sent a photo. media_ref={media_ref}"
    assert call["prompt"].startswith(prefix)
    ref = call["prompt"].split("media_ref=", 1)[1].split("\n", 1)[0]
    assert call["prompt"] == f"User sent a photo. media_ref={ref}\nCaption: {LINK_TEXT}{NOTE}"
    assert call["image_b64"] == base64.b64encode(PHOTO).decode()
    assert call["chat_id"] == chat.chat_id
    if chat.name == "telegram":
        assert call["image_media_type"] == "image/jpeg"
    else:
        # Slack and WhatsApp keep a copy under their media dir.
        assert Path(ref).read_bytes() == PHOTO
        assert Path(ref).suffix == {"slack": ".png", "whatsapp": ".jpg"}[chat.name]
        assert call["image_media_type"] == {"slack": "image/png", "whatsapp": "image/jpeg"}[chat.name]
    assert chat.replies == ["Got it."]


async def test_photo_without_caption(chat):
    await chat.photo("")
    [call] = chat.agent.calls
    assert "\n" not in call["prompt"]
    assert call["prompt"].startswith("User sent a photo. media_ref=")
    assert chat.env.openai.chats == []


async def test_failed_voice_transcription(chat):
    chat.env.openai.fail_transcription = True
    await chat.voice()
    assert chat.agent.calls == []
    assert chat.replies == [TRANSCRIPTION_FAILED]
    assert chat.env.openai.transcriptions == [("whisper-1", {
        "telegram": ("voice.ogg", AUDIO, "audio/ogg"),
        "slack": ("audio.m4a", AUDIO, "audio/mp4"),
        "whatsapp": ("voice.ogg", AUDIO, "audio/ogg; codecs=opus"),
    }[chat.name])]


async def test_voice_note(chat):
    await chat.voice()
    assert chat.agent.calls == [{
        "prompt": VOICE_PROMPT, "image_b64": None,
        "image_media_type": "image/jpeg", "chat_id": chat.chat_id,
    }]
    assert chat.replies == ["Got it."]


async def test_stranger_is_ignored(chat):
    await chat.text("hello")
    await chat.text("let me in", stranger=True)
    assert [c["prompt"] for c in chat.agent.calls] == ["hello"]
    assert chat.replies == ["Got it."]


async def test_new_conversation(chat):
    await chat.new()
    assert chat.agent.resets == [chat.chat_id]
    assert chat.agent.calls == []
    assert chat.replies == ["Started a new conversation."]


async def test_reminder_delivery(chat):
    scheduler: ReminderScheduler = chat.agent.scheduler
    assert chat.agent.platform == chat.name
    scheduler.create(
        platform=chat.name, chat_id=chat.chat_id, text="stretch",
        fire_at=datetime.now() - timedelta(seconds=5),
    )
    await scheduler._tick()
    target = int(chat.chat_id) if chat.name == "telegram" else chat.chat_id
    if chat.name == "whatsapp":
        target = "jid:111@s.whatsapp.net"
    assert chat.delivered() == [(target, "⏰ Reminder: stretch")]
    assert chat.agent.fired == [("⏰ Reminder: stretch", chat.chat_id)]
    assert scheduler.list_for(chat.name, chat.chat_id) == []


async def test_long_reply(chat):
    chat.agent.reply = ("x" * 5000, [])
    await chat.text("hi")
    limit = {"telegram": 4096, "slack": 4000, "whatsapp": 5000}[chat.name]
    assert chat.replies == ["x" * limit]


# ── Where the platforms differ (kept as they are; see the docstrings) ────


async def test_found_images(chat, tmp_path):
    own = tmp_path / "own.jpg"
    own.write_bytes(b"img")
    chat.agent.reply = ("Here.", [
        {"platform": chat.name, "media_ref": str(own), "file_id": str(own), "caption": "mine"},
        {"platform": "elsewhere", "media_ref": "X", "file_id": "X", "description": "a cat"},
    ])
    await chat.text("show me")
    if chat.name == "telegram":
        # Telegram sends every image by file_id, whatever platform saved it.
        assert [p["photo"] for p in chat.tg.photos] == [str(own), "X"]
        assert [p["caption"] for p in chat.tg.photos] == ["mine", None]
        assert chat.replies == ["Here."]
    elif chat.name == "slack":
        assert [(u["file"], u["initial_comment"]) for u in chat.app.uploads] == [(str(own), "mine")]
        assert chat.replies == ["Here.\n\n_(Found image: a cat -- originally saved via elsewhere)_"]
    else:
        assert [(path, caption) for _, path, caption in chat.client.images] == [(str(own), "mine")]
        assert chat.replies == ["Here.\n\n(Found image: a cat — originally saved via elsewhere)"]


async def test_reply_formatting(chat):
    await chat.text("hi")
    if chat.name == "telegram":
        assert chat.reply_kwargs == [{"parse_mode": "Markdown"}]
    elif chat.name == "slack":
        assert chat.said == [{"text": "Got it.", "thread_ts": "1.0"}]


async def test_telegram_falls_back_to_plain_text(env):
    chat = TelegramDriver(env)
    chat.markdown_fails = True
    await chat.text("hi")
    assert chat.replies == ["Got it."] and chat.reply_kwargs == [{}]


async def test_telegram_quotes_the_replied_message(env):
    chat = TelegramDriver(env)
    update = chat._update(text=LINK_TEXT, reply_to=SimpleNamespace(text=None, caption="the photo caption"))
    await chat.bot.handle_text(update, chat._context())
    assert [c["prompt"] for c in chat.agent.calls] == [
        f"[Replying to message] the photo caption\n{LINK_TEXT}\n{NOTE}"
    ]


async def test_telegram_empty_allowlist_blocks_everyone(env):
    """Quirk: on Telegram an empty ALLOWED_USER_IDS lets nobody in..."""
    env.config.telegram.allowed_user_ids = ""
    chat = TelegramDriver(env)
    await chat.text("hello")
    assert chat.agent.calls == [] and chat.replies == []


async def test_slack_empty_allowlist_lets_everyone_in(env):
    """...while on Slack an empty SLACK_ALLOWED_USERS lets everyone in."""
    env.config.slack.allowed_users = ""
    chat = SlackDriver(env)
    await chat.text("hello", user="U-anyone")
    assert [c["prompt"] for c in chat.agent.calls] == ["hello"]


async def test_slack_allowlists(env):
    env.config.slack.allowed_users = "U1"
    env.config.slack.allowed_channels = "C1"
    chat = SlackDriver(env)
    await chat.text("from a stranger", stranger=True)
    await chat._event({"channel": "C2", "user": "U1", "text": "other channel", "ts": "1.0"})
    await chat._event({"channel": "C1", "user": "U1", "text": "a bot", "ts": "1.0", "bot_id": "B1"})
    await chat.text("ok")
    assert [c["prompt"] for c in chat.agent.calls] == ["ok"]


async def test_slack_download_failure(env):
    chat = SlackDriver(env)
    FakeHttpx.fail = True
    await chat.photo("x")
    await chat.voice()
    assert chat.agent.calls == []
    assert chat.replies == [
        "Sorry, I couldn't download that image.", "Sorry, I couldn't download that audio file.",
    ]
    assert FakeHttpx.calls[0] == {"Authorization": "Bearer xoxb-1"}


async def test_whatsapp_download_failure(env):
    chat = WhatsAppDriver(env)
    chat.client.download = RuntimeError("gone")
    await chat._message(_wa_event(image={"caption": "", "mimetype": "image/jpeg"}))
    await chat._message(_wa_event(audio={"mimetype": "audio/ogg"}))
    assert chat.agent.calls == []
    assert chat.replies == [
        "Sorry, I couldn't download that image.", "Sorry, I couldn't download that audio message.",
    ]


async def test_whatsapp_only_self_chat(env):
    chat = WhatsAppDriver(env)
    await chat._message(_wa_event(text="to a friend", chat_user="222"))  # my DM to someone else
    await chat._message(_wa_event(text="group", group=True))
    await chat.text("note to self")
    assert [c["prompt"] for c in chat.agent.calls] == ["note to self"]
