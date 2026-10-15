"""Tests for the MEMORY.md → note reference check."""
from __future__ import annotations

from pathlib import Path

from memclaw.memory_refs import check_references


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "2026-05-12.md").write_text("Lion8 prod chat login: admin / secret")
    (tmp_path / "todos.md").write_text("- renew passport")
    return tmp_path


def test_valid_references_are_kept_unchanged(tmp_path: Path):
    root = _vault(tmp_path)
    text = "## Index\n- Lion8 prod login → memory/2026-05-12.md\n- Todos → todos.md"
    result = check_references(text, root)
    assert result.text == text
    assert result.refs == 2 and not result.dropped and not result.unconfirmed


def test_missing_note_is_dropped(tmp_path: Path):
    root = _vault(tmp_path)
    result = check_references(
        "- Lion8 prod login → memory/2026-05-12.md, memory/2026-05-13.md", root,
    )
    assert result.text == "- Lion8 prod login → memory/2026-05-12.md"
    assert len(result.dropped) == 1 and "2026-05-13.md" in result.dropped[0]


def test_line_with_only_missing_notes_keeps_its_summary(tmp_path: Path):
    root = _vault(tmp_path)
    result = check_references("- Slack tokens → memory/2026-04-12.md", root)
    assert result.text == "- Slack tokens"


def test_note_that_shares_no_words_is_reported(tmp_path: Path):
    root = _vault(tmp_path)
    result = check_references("- Daisy Disk serial → memory/2026-05-12.md", root)
    assert result.text == "- Daisy Disk serial → memory/2026-05-12.md"
    assert len(result.unconfirmed) == 1


def test_bare_date_file_resolves_to_daily_dir(tmp_path: Path):
    root = _vault(tmp_path)
    result = check_references("- Lion8 login → 2026-05-12.md", root)
    assert not result.dropped


def test_prose_arrows_and_self_references_are_ignored(tmp_path: Path):
    root = _vault(tmp_path)
    text = "- RSD 447,715 → 274,077 over the trip\n- see → MEMORY.md"
    result = check_references(text, root)
    assert result.text == text and result.refs == 0
