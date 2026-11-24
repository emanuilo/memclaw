"""Memclaw's settings, and the only module that reads the environment.

Each setting is declared once, as a dataclass field made with `setting()`:
its environment variable, its default, and the setup wizard's question if it
asks one. `Settings.__post_init__` is the one loader: every field still at its
default is filled from its environment variable, so an explicit constructor
value wins. Plain fields (the memory tuning) aren't read from the environment.

The settings are split into small groups hung off `MemclawConfig`
(`config.openai`, `config.claude`, `config.slack`, ...).

Adding a setting: add one `setting(...)` line to its group, a line to
`.env.example` and a row to docs/configuration.md (tests/test_config.py fails
until both are there). Add `prompt=` if the setup wizard should ask for it.

Where a value comes from, highest first: the constructor, the process
environment, a `.env` found by python-dotenv (searched upward from this file's
folder; from the current folder in a REPL), then ~/.memclaw/.env.
"""

from __future__ import annotations

import importlib.resources
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields, replace
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ENV_FILE = Path.home() / ".memclaw" / ".env"

# Neither file overrides a variable already set in the environment.
load_dotenv()
load_dotenv(ENV_FILE)


# ── Declaring settings ──────────────────────────────────────────────


@dataclass(frozen=True)
class Setting:
    """What Memclaw knows about one setting, besides its default."""

    env: str  # the environment variable (and ~/.memclaw/.env key)
    prompt: str | None = None  # the setup wizard's question; None = not asked by it
    required: bool = False  # the wizard refuses an empty answer
    secret: bool = False  # kept out of repr()
    parse: Callable[[str], Any] = str  # environment text -> value
    group: str = ""  # filled in by all_settings(): "general", "openai", "slack", ...
    name: str = ""  # filled in by all_settings(): the field name


def setting(env: str, default: Any = "", **meta: Any) -> Any:
    """A settings field read from *env*, defaulting to *default*. *meta* is the
    rest of `Setting` (`prompt=`, `required=`, `secret=`, `parse=`)."""
    info = Setting(env, **meta)
    return field(default=default, repr=not info.secret, metadata={"setting": info})


def _port(raw: str) -> int:
    try:
        port = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"must be an integer, got {raw.strip()!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"must be between 1 and 65535, got {port}")
    return port


def _split(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass
class Settings:
    """Base of every settings group: the generic environment loader."""

    def __post_init__(self) -> None:
        for f in fields(self):
            meta: Setting | None = f.metadata.get("setting")
            if meta is None or getattr(self, f.name) != f.default:
                continue
            raw = os.environ.get(meta.env, "")
            if not raw.strip():
                continue
            try:
                setattr(self, f.name, meta.parse(raw))
            except ValueError as exc:
                raise ValueError(f"{meta.env} {exc}") from exc


# ── The groups ──────────────────────────────────────────────────────


@dataclass
class OpenAISettings(Settings):
    # Embeddings, link summaries, image descriptions and voice transcription.
    api_key: str = setting(
        "OPENAI_API_KEY", secret=True, required=True,
        prompt="OpenAI API key (for embeddings + voice transcription)",
    )


@dataclass
class MemorySettings(Settings):
    """Indexing, search and consolidation tuning; not read from the
    environment. Described in docs/memory-architecture.md."""

    embedding_model: str = "text-embedding-3-small"
    chunk_target_words: int = 300
    chunk_overlap_words: int = 60
    vector_weight: float = 0.7
    text_weight: float = 0.3
    decay_half_life_days: int = 30
    mmr_lambda: float = 0.7
    # Consolidation runs (after a reply) once this many past days have notes
    # that aren't in MEMORY.md yet. 1 = after each day.
    consolidation_threshold: int = 1


@dataclass
class ClaudeSettings(Settings):
    """The Claude Agent SDK backend. Empty means the built-in default (see
    memclaw.backends.claude), which also asks for these in the wizard."""

    api_key: str = setting("ANTHROPIC_API_KEY", secret=True)
    oauth_token: str = setting("CLAUDE_CODE_OAUTH_TOKEN", secret=True)
    model: str = setting("CLAUDE_MODEL")
    effort: str = setting("CLAUDE_EFFORT")
    # Memory consolidation is a rare one-shot call where a stronger model pays off.
    consolidation_model: str = setting("CLAUDE_CONSOLIDATION_MODEL")
    consolidation_effort: str = setting("CLAUDE_CONSOLIDATION_EFFORT")


@dataclass
class CursorSettings(Settings):
    """The Cursor SDK backend (memclaw.backends.cursor, which also asks for
    these in the wizard)."""

    api_key: str = setting("CURSOR_API_KEY", secret=True)
    model: str = setting("CURSOR_MODEL")
    effort: str = setting("CURSOR_EFFORT")
    # The local MCP HTTP server that serves Memclaw's tools to the Cursor agent.
    mcp_http_port: int = setting("MEMCLAW_MCP_PORT", 17373, parse=_port)


@dataclass
class TelegramSettings(Settings):
    bot_token: str = setting(
        "TELEGRAM_BOT_TOKEN", secret=True, required=True, prompt="Telegram bot token")
    allowed_user_ids: str = setting(
        "ALLOWED_USER_IDS", required=True, prompt="Allowed Telegram user IDs (comma-separated)")

    @property
    def allowed_user_ids_list(self) -> list[int]:
        return [int(uid) for uid in _split(self.allowed_user_ids)]


@dataclass
class SlackSettings(Settings):
    bot_token: str = setting(
        "SLACK_BOT_TOKEN", secret=True, required=True, prompt="Slack bot token (xoxb-...)")
    app_token: str = setting(
        "SLACK_APP_TOKEN", secret=True, required=True,
        prompt="Slack app-level token for Socket Mode (xapp-...)")
    # Empty = any channel (DMs included) the allowed users write in.
    allowed_channels: str = setting(
        "SLACK_ALLOWED_CHANNELS", prompt="Allowed Slack channel IDs (comma-separated)")
    # Like ALLOWED_USER_IDS, an empty list lets nobody in.
    allowed_users: str = setting(
        "SLACK_ALLOWED_USERS", required=True, prompt="Allowed Slack user IDs (comma-separated)")

    @property
    def allowed_channels_list(self) -> list[str]:
        return _split(self.allowed_channels)

    @property
    def allowed_users_list(self) -> list[str]:
        return _split(self.allowed_users)


@dataclass
class MemclawConfig(Settings):
    """Configuration for Memclaw memory assistant."""

    memory_dir: Path = field(default_factory=lambda: Path.home() / ".memclaw")
    # Which agent backend to use (see memclaw.backends.REGISTRY). Empty
    # resolves to the default backend at runtime.
    agent_backend: str = setting("AGENT_BACKEND")
    # Which front-end the bare `memclaw` launches: "terminal" (also when
    # empty), "telegram", "slack" or "whatsapp".
    platform: str = setting("MEMCLAW_PLATFORM")

    openai: OpenAISettings = field(default_factory=OpenAISettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    claude: ClaudeSettings = field(default_factory=ClaudeSettings)
    cursor: CursorSettings = field(default_factory=CursorSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    slack: SlackSettings = field(default_factory=SlackSettings)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.memory_dir = Path(self.memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_subdir.mkdir(exist_ok=True)
        self._init_default_files()

    def _init_default_files(self):
        """Copy default files into memory_dir if they don't exist yet."""
        defaults = {"AGENTS.md": self.agent_file}
        for filename, dest in defaults.items():
            if not dest.exists():
                src = importlib.resources.files("memclaw.defaults").joinpath(filename)
                dest.write_text(src.read_text(encoding="utf-8"))

        if not self.memory_file.exists():
            self.memory_file.write_text("# Personal Memory\n")

    # Paths derived from memory_dir.

    @property
    def db_path(self) -> Path:
        return self.memory_dir / "memclaw.db"

    @property
    def memory_subdir(self) -> Path:
        return self.memory_dir / "memory"

    @property
    def memory_file(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    @property
    def agent_file(self) -> Path:
        return self.memory_dir / "AGENTS.md"

    def daily_file(self, dt: date | None = None) -> Path:
        dt = dt or date.today()
        return self.memory_subdir / f"{dt.isoformat()}.md"

    @property
    def whatsapp_dir(self) -> Path:
        d = self.memory_dir / "whatsapp"
        d.mkdir(exist_ok=True)
        return d

    @property
    def whatsapp_session_db(self) -> Path:
        return self.whatsapp_dir / "session.db"

    @property
    def whatsapp_media_dir(self) -> Path:
        d = self.whatsapp_dir / "media"
        d.mkdir(exist_ok=True)
        return d

    @property
    def slack_dir(self) -> Path:
        d = self.memory_dir / "slack"
        d.mkdir(exist_ok=True)
        return d

    @property
    def slack_media_dir(self) -> Path:
        d = self.slack_dir / "media"
        d.mkdir(exist_ok=True)
        return d


def all_settings() -> list[Setting]:
    """Every setting read from the environment, in declaration order, with
    its group and name filled in.

    Top-level settings are in the group "general"; the others are in the
    group named after their attribute on MemclawConfig ("openai", "slack", ...).
    """
    found = _settings_of(MemclawConfig, "general")
    for f in fields(MemclawConfig):
        if isinstance(f.default_factory, type) and issubclass(f.default_factory, Settings):
            found += _settings_of(f.default_factory, f.name)
    return found


def _settings_of(cls: type, group: str) -> list[Setting]:
    return [
        replace(f.metadata["setting"], group=group, name=f.name)
        for f in fields(cls) if "setting" in f.metadata
    ]


def env_name(settings: Settings, name: str) -> str:
    """The environment variable behind *settings*' field *name*."""
    return next(f.metadata["setting"].env for f in fields(settings) if f.name == name)


# ── The process environment and ~/.memclaw/.env ─────────────────────


def process_environment() -> dict[str, str]:
    """A copy of this process's environment, to pass on to a child process."""
    return dict(os.environ)


def set_environment(values: dict[str, str], drop_keys: Iterable[str] = ()) -> None:
    """Set *values* (empty ones skipped) and remove *drop_keys* in this
    process's environment, so a MemclawConfig made afterwards sees them and
    child processes inherit them."""
    for key in drop_keys:
        os.environ.pop(key, None)
    for key, value in values.items():
        if value:
            os.environ[key] = value


def reload_env_file() -> None:
    """Load ~/.memclaw/.env again, its values winning over the environment
    (after the setup wizard has just written it)."""
    load_dotenv(ENV_FILE, override=True)


def read_env_file() -> dict[str, str]:
    """Load existing values from ~/.memclaw/.env."""
    values: dict[str, str] = {}
    if not ENV_FILE.exists():
        return values
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip()
    return values


def update_env_file(values: dict[str, str], drop_keys: list[str] | tuple[str, ...] = ()) -> None:
    """Merge *values* into ~/.memclaw/.env and remove *drop_keys*.

    Keys are written in their existing order, new ones appended; a key with
    an empty value is removed, matching how the wizard writes the file.
    """
    merged = read_env_file()
    merged.update(values)
    for key in drop_keys:
        merged.pop(key, None)
    write_env_file(merged)


def write_env_file(values: dict[str, str]) -> None:
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in values.items() if v]
    ENV_FILE.write_text("\n".join(lines) + "\n")
