"""Tests for ChatSessions — each chat's session, queued reminders and /new
(the agent's turn-level behaviour is in test_agent.py)."""
from __future__ import annotations

import json
from pathlib import Path

from memclaw.sessions import ChatSessions


def _chats(tmp_path: Path, *, platform: str | None = "telegram", fingerprint: str = "fp1") -> ChatSessions:
    return ChatSessions(tmp_path / "sessions.json", fingerprint, platform)


def _saved(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "sessions.json").read_text())


def test_session_keys():
    chats = ChatSessions(Path("unused.json"), "fp", "slack")
    assert chats.key("C1") == "slack:C1"
    assert chats.key(None) == "cli"
    assert ChatSessions(Path("unused.json"), "fp").key("42") == "42"


def test_a_finished_turn_is_saved_and_resumed(tmp_path: Path):
    chats = _chats(tmp_path)
    turn = chats.begin("42")
    assert turn.entry is None and turn.resume_session_id is None

    chats.finish(turn, "sess-1", agents_hash="a1", memory_hash="m1")

    assert _saved(tmp_path) == {"telegram:42": {
        "session_id": "sess-1", "fingerprint": "fp1", "agents_hash": "a1", "memory_hash": "m1",
    }}
    again = _chats(tmp_path).begin("42")
    assert again.resume_session_id == "sess-1"
    assert again.saw_other("agents_hash", "a2") and not again.saw_other("memory_hash", "m1")


def test_a_session_from_other_instructions_is_not_resumed(tmp_path: Path):
    chats = _chats(tmp_path)
    chats.finish(chats.begin("42"), "sess-1", agents_hash="a", memory_hash="m")

    assert _chats(tmp_path, fingerprint="fp2").begin("42").entry is None


def test_a_turn_without_a_session_id_saves_nothing(tmp_path: Path):
    chats = _chats(tmp_path)
    chats.finish(chats.begin("42"), None, agents_hash="a", memory_hash="m")
    assert not (tmp_path / "sessions.json").exists()


def test_only_the_notes_a_turn_carried_are_dropped(tmp_path: Path):
    chats = _chats(tmp_path)
    chats.add_note("42", "first")
    turn = chats.begin("42")
    chats.add_note("42", "during the turn")
    chats.add_note("7", "other chat")

    chats.finish(turn, "s", agents_hash="a", memory_hash="m")

    assert turn.notes == ["first"]
    assert chats.begin("42").notes == ["during the turn"]
    assert chats.begin("7").notes == ["other chat"]


def test_new_forgets_the_session_and_notes(tmp_path: Path):
    chats = _chats(tmp_path)
    chats.finish(chats.begin("42"), "sess-1", agents_hash="a", memory_hash="m")
    chats.add_note("42", "a reminder")

    assert chats.reset("42") == "telegram:42"

    turn = chats.begin("42")
    assert turn.entry is None and turn.notes == []
    assert _saved(tmp_path) == {}


def test_a_turn_overtaken_by_new_does_not_save_its_session(tmp_path: Path):
    chats = _chats(tmp_path)
    turn = chats.begin("42")
    chats.reset("42")

    chats.finish(turn, "old-session", agents_hash="a", memory_hash="m")

    assert chats.begin("42").entry is None
