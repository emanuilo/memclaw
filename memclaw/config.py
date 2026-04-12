from __future__ import annotations

import importlib.resources
import os
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Load .env from current directory and ~/.memclaw/.env
load_dotenv()
load_dotenv(Path.home() / ".memclaw" / ".env")


@dataclass
class MemclawConfig:
    """Configuration for Memclaw memory assistant."""

    memory_dir: Path = field(default_factory=lambda: Path.home() / ".memclaw")
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    chunk_target_words: int = 300
    chunk_overlap_words: int = 60
    vector_weight: float = 0.7
    text_weight: float = 0.3
    decay_half_life_days: int = 30
    mmr_lambda: float = 0.7
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Conversation continuity
    conversation_history_limit: int = 10

    # Memory consolidation
    consolidation_threshold: int = 7

    # Telegram bot settings
    telegram_bot_token: str = ""
    allowed_user_ids: str = ""

    def __post_init__(self):
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.telegram_bot_token:
            self.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not self.allowed_user_ids:
            self.allowed_user_ids = os.environ.get("ALLOWED_USER_IDS", "")
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
    def images_dir(self) -> Path:
        d = self.memory_dir / "images"
        d.mkdir(exist_ok=True)
        return d

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
    def allowed_user_ids_list(self) -> list[int]:
        if not self.allowed_user_ids:
            return []
        return [
            int(uid.strip())
            for uid in self.allowed_user_ids.split(",")
            if uid.strip()
        ]

