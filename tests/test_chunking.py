"""Tests for chunk_text: how a note is split into the chunks the index stores."""
from __future__ import annotations

from pathlib import Path

from memclaw.chunking import chunk_text
from memclaw.config import MemclawConfig, MemorySettings
from memclaw.memory import MemoryStore

_DEFAULTS = MemorySettings()


def _chunks(text: str, file_path: str):
    return chunk_text(
        text, file_path,
        target_words=_DEFAULTS.chunk_target_words,
        overlap_words=_DEFAULTS.chunk_overlap_words,
    )


class TestChunking:
    def test_basic_chunking(self):
        text = "Hello world.\n\n---\n\nSecond paragraph."
        chunks = _chunks(text, "test.md")
        assert len(chunks) >= 1
        assert all(c.file_path == "test.md" for c in chunks)

    def test_empty_text(self):
        chunks = _chunks("", "test.md")
        assert chunks == []


class TestEntryChunking:
    def test_each_entry_is_its_own_chunk(self, tmp_path: Path):
        store = MemoryStore(MemclawConfig(memory_dir=tmp_path / "m"))
        store.save("First entry about apples.")
        path = store.save("Second entry about bananas.")
        chunks = _chunks(path.read_text(), str(path))
        bananas = [c for c in chunks if "bananas" in c.content]
        assert len(bananas) == 1
        assert "apples" not in bananas[0].content
        assert bananas[0].content.count("## ") == 1

    def test_long_entry_is_split_with_overlap_under_its_heading(self):
        body = "\n".join(f"line {i} " + "word " * 20 for i in range(40))
        chunks = _chunks(f"## 10:00 - Note\n\n{body}\n\n---\n", "f.md")
        assert len(chunks) > 1
        assert all(c.content.startswith("## 10:00 - Note") for c in chunks)
        # Consecutive chunks share some text.
        assert chunks[0].content.splitlines()[-1] in chunks[1].content


class TestSizeSplits:
    def test_a_heading_starts_a_new_chunk_once_half_full(self):
        text = "# A\n" + "word " * 12 + "\n# B\nmore words here"
        chunks = chunk_text(text, "f.md", target_words=20, overlap_words=5)
        assert [c.content.splitlines()[0] for c in chunks] == ["# A", "# B"]
        assert [(c.line_start, c.line_end) for c in chunks] == [(0, 1), (2, 3)]

    def test_an_entry_overlapped_whole_is_not_repeated(self):
        # The overlap reaches back to the entry's first line: nothing is kept.
        chunks = chunk_text("one two three four", "f.md", target_words=4, overlap_words=50)
        assert [c.content for c in chunks] == ["one two three four"]
