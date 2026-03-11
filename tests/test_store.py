"""Tests for MemoryStore — particularly list_unconsolidated_files (spec #2)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from memclaw.config import MemclawConfig
from memclaw.store import MemoryStore


def _make_store(tmp_path: Path) -> tuple[MemoryStore, MemclawConfig]:
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    return MemoryStore(cfg), cfg


def test_list_unconsolidated_files_all(tmp_path: Path):
    """With no consolidated_through, all daily files are returned."""
    store, cfg = _make_store(tmp_path)

    # Create some daily files
    for name in ["2025-03-01.md", "2025-03-05.md", "2025-03-10.md"]:
        (cfg.memory_subdir / name).write_text(f"# {name}\ncontent")

    files = store.list_unconsolidated_files(consolidated_through=None)
    stems = [f.stem for f in files]
    assert stems == ["2025-03-01", "2025-03-05", "2025-03-10"]


def test_list_unconsolidated_files_with_cutoff(tmp_path: Path):
    """Files at or before consolidated_through are excluded."""
    store, cfg = _make_store(tmp_path)

    for name in ["2025-03-01.md", "2025-03-05.md", "2025-03-10.md"]:
        (cfg.memory_subdir / name).write_text("content")

    files = store.list_unconsolidated_files(consolidated_through=date(2025, 3, 5))
    stems = [f.stem for f in files]
    assert stems == ["2025-03-10"]


def test_list_unconsolidated_files_sorted(tmp_path: Path):
    """Results are sorted by date ascending."""
    store, cfg = _make_store(tmp_path)

    # Create in reverse order
    for name in ["2025-03-10.md", "2025-03-01.md", "2025-03-05.md"]:
        (cfg.memory_subdir / name).write_text("content")

    files = store.list_unconsolidated_files()
    stems = [f.stem for f in files]
    assert stems == ["2025-03-01", "2025-03-05", "2025-03-10"]


def test_list_unconsolidated_files_ignores_non_daily(tmp_path: Path):
    """Non-date filenames are ignored."""
    store, cfg = _make_store(tmp_path)

    (cfg.memory_subdir / "2025-03-01.md").write_text("content")
    (cfg.memory_subdir / "random-notes.md").write_text("content")
    (cfg.memory_subdir / "not-a-date.md").write_text("content")

    files = store.list_unconsolidated_files()
    assert len(files) == 1
    assert files[0].stem == "2025-03-01"


def test_list_unconsolidated_files_empty(tmp_path: Path):
    """Returns empty list when no daily files exist."""
    store, _ = _make_store(tmp_path)
    assert store.list_unconsolidated_files() == []


def test_list_unconsolidated_files_all_consolidated(tmp_path: Path):
    """Returns empty list when all files are before the cutoff."""
    store, cfg = _make_store(tmp_path)

    (cfg.memory_subdir / "2025-03-01.md").write_text("content")
    (cfg.memory_subdir / "2025-03-05.md").write_text("content")

    files = store.list_unconsolidated_files(consolidated_through=date(2025, 3, 10))
    assert files == []


def test_save_creates_daily_file(tmp_path: Path):
    store, cfg = _make_store(tmp_path)
    path = store.save("Hello world")
    assert path.exists()
    content = path.read_text()
    assert "Hello world" in content


def test_save_permanent(tmp_path: Path):
    store, cfg = _make_store(tmp_path)
    path = store.save("Permanent fact", permanent=True)
    assert path == cfg.memory_file
    assert "Permanent fact" in path.read_text()
