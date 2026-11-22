"""Per-chat conversation sessions.

`ChatSessions` is the agent's bookkeeping for each chat: which conversation
it is in, the reminders it hasn't seen yet, and `/new`. `SessionStore` is
the saved part (sessions.json): it maps a chat's session key to the
backend session id the conversation lives in, so a restarted Memclaw
resumes it. Each entry also records:

- ``fingerprint`` — hash of the code-level instructions (system-prompt
  template + backend) the session started with. Claude Code keeps a
  session's original system prompt on resume, so a session started under
  different instructions is not resumed.
- ``agents_hash`` / ``memory_hash`` — hashes of the AGENTS.md and MEMORY.md
  content the session has seen, so a later change is sent to it once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# Session key for the interactive terminal (no chat id).
_CLI_SESSION_KEY = "cli"


class SessionStore:
    """``{session_key: {session_id, fingerprint, agents_hash, memory_hash}}``
    in a JSON file, written atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict[str, str]] = {}
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError:
            raw = {}
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable session store {p}: {exc}", p=path, exc=exc)
            raw = {}
        if isinstance(raw, dict):
            self._data = {k: v for k, v in raw.items() if isinstance(v, dict)}

    def get(self, key: str, fingerprint: str) -> dict[str, str] | None:
        """The entry to resume, or None. An entry from other instructions is
        dropped."""
        entry = self._data.get(key)
        if entry is None or not entry.get("session_id"):
            return None
        if entry.get("fingerprint") != fingerprint:
            logger.info("Instructions changed; not resuming the old session for {k}", k=key)
            self.drop(key)
            return None
        return dict(entry)

    def set(self, key: str, entry: dict[str, str]) -> None:
        if self._data.get(key) == entry:
            return
        self._data[key] = dict(entry)
        self._save()

    def drop(self, key: str) -> None:
        if self._data.pop(key, None) is not None:
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)


@dataclass
class TurnStart:
    """A chat's session as a turn begins (`ChatSessions.begin`)."""

    key: str
    # /new count when the turn began; a turn that a /new overtook doesn't
    # save its session.
    generation: int
    # The saved session to resume, or None.
    entry: dict[str, str] | None
    # Reminders delivered since the chat's last message.
    notes: list[str]

    @property
    def resume_session_id(self) -> str | None:
        return self.entry["session_id"] if self.entry else None

    def saw_other(self, field: str, content_hash: str) -> bool:
        """Whether the resumed session saw another version of a file than
        *content_hash* (*field*: "agents_hash" or "memory_hash")."""
        return self.entry is not None and self.entry.get(field) != content_hash


class ChatSessions:
    """Each chat's conversation: its saved backend session (resumed unless
    the instructions changed, see *fingerprint*), the reminders delivered
    since its last message, and `/new`."""

    def __init__(self, path: Path, fingerprint: str, platform: str | None = None) -> None:
        self._store = SessionStore(path)
        self.fingerprint = fingerprint
        self._platform = platform
        self._pending_notes: dict[str, list[str]] = {}
        # Bumped by /new so a turn that was running doesn't re-save the
        # session it just reset.
        self._generations: dict[str, int] = {}

    def key(self, chat_id: str | None) -> str:
        """The session key of *chat_id*: `<platform>:<chat id>`, or "cli"
        for the terminal (no chat id)."""
        if chat_id is None:
            return _CLI_SESSION_KEY
        return f"{self._platform}:{chat_id}" if self._platform else str(chat_id)

    def saved(self, key: str) -> dict[str, str] | None:
        """The saved session of *key* that would be resumed, or None."""
        return self._store.get(key, self.fingerprint)

    def add_note(self, chat_id: str | None, text: str) -> None:
        """Queue a delivered reminder for the chat's next message."""
        self._pending_notes.setdefault(self.key(chat_id), []).append(text)

    def reset(self, chat_id: str | None) -> str:
        """Forget the chat's session and queued reminders (`/new`). Returns
        its session key."""
        key = self.key(chat_id)
        self._generations[key] = self._generations.get(key, 0) + 1
        self._store.drop(key)
        self._pending_notes.pop(key, None)
        return key

    def begin(self, chat_id: str | None) -> TurnStart:
        """What the chat's session has seen, as a turn begins."""
        key = self.key(chat_id)
        return TurnStart(
            key=key,
            generation=self._generations.get(key, 0),
            entry=self.saved(key),
            notes=list(self._pending_notes.get(key, ())),
        )

    def finish(
        self, turn: TurnStart, session_id: str | None, *, agents_hash: str, memory_hash: str,
    ) -> None:
        """After a successful turn: the session has seen *turn*'s notes and
        these file versions. Only those notes are dropped (a reminder
        delivered during the turn stays queued), and the session is saved
        unless a /new came in meanwhile."""
        pending = self._pending_notes.get(turn.key)
        if pending:
            del pending[:len(turn.notes)]
        if session_id and self._generations.get(turn.key, 0) == turn.generation:
            self._store.set(turn.key, {
                "session_id": session_id,
                "fingerprint": self.fingerprint,
                "agents_hash": agents_hash,
                "memory_hash": memory_hash,
            })
