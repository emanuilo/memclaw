"""Splitting a note into the chunks the index embeds and searches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    file_path: str
    line_start: int
    line_end: int
    content: str
    embedding: np.ndarray | None = None


def chunk_text(text: str, file_path: str, *, target_words: int, overlap_words: int) -> list[Chunk]:
    """Split text into chunks by entries, headings and size.

    A ``---`` line ends an entry (``MemoryStore.save`` writes one after
    each), so every entry starts a fresh chunk. A heading starts a new
    chunk once the current one is half full. Only an entry too long for
    one chunk is split by size, with overlap and its heading repeated.
    """
    chunker = _Chunker(file_path, target_words, overlap_words)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        chunker.add(i, line)
    if chunker.current:
        chunker.flush(len(lines) - 1)
    return chunker.chunks


class _Chunker:
    """The state of one `chunk_text` pass: the lines of the chunk being
    built, where it starts, its word count and the heading it is under."""

    def __init__(self, file_path: str, target_words: int, overlap_words: int):
        self.file_path = file_path
        self.target = target_words
        self.overlap = overlap_words
        self.chunks: list[Chunk] = []
        self.current: list[str] = []
        self.start = 0
        self.words = 0
        self.heading = ""

    def add(self, i: int, line: str) -> None:
        is_heading = line.startswith("#")
        if is_heading and self.words > self.target // 2:
            self.flush(i - 1)
            self.current, self.start, self.words = [], i, 0
        if is_heading:
            self.heading = line

        self.current.append(line)
        self.words += len(line.split())

        if line.strip() == "---":
            self.flush(i)
            self.current, self.start, self.words, self.heading = [], i + 1, 0, ""
        elif self.words >= self.target:
            self.flush(i)
            self._keep_overlap(i)

    def flush(self, end: int) -> None:
        content = "\n".join(self.current).strip()
        if content and content != "---":
            self.chunks.append(Chunk(file_path=self.file_path, line_start=self.start,
                                     line_end=end, content=content))

    def _keep_overlap(self, i: int) -> None:
        """After a size split at line *i*: keep a tail of the entry for
        continuity, under its heading."""
        overlap_words = 0
        keep = len(self.current)
        for j in range(len(self.current) - 1, -1, -1):
            overlap_words += len(self.current[j].split())
            keep = j
            if overlap_words >= self.overlap:
                break
        self.current = self.current[keep:] if keep > 0 else []
        self.start = i - len(self.current) + 1
        if self.heading and self.current and not any(ln.startswith("#") for ln in self.current):
            self.current.insert(0, self.heading)
        self.words = sum(len(ln.split()) for ln in self.current)
