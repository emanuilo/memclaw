"""Persisted per-chat conversation sessions.

Maps a chat's session key to the backend session id the conversation lives
in, so a restarted Memclaw resumes it. Each entry also records:

- ``fingerprint`` — hash of the code-level instructions (system-prompt
  template + backend) the session started with. Claude Code keeps a
  session's original system prompt on resume, so a session started under
  different instructions is not resumed.
- ``agents_hash`` / ``memory_hash`` — hashes of the AGENTS.md and MEMORY.md
  content the session has seen, so a later change is sent to it once.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


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
