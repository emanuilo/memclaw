"""References from MEMORY.md into the detailed notes.

MEMORY.md is a profile plus an index: details stay in the daily files (and
other notes) and MEMORY.md points at them with lines such as

    - Lion8 credentials and SSH commands → memory/2026-05-12.md, memory/2026-06-13.md

Consolidation writes these references with an LLM, so they are checked here
before MEMORY.md is saved: a reference to a file that doesn't exist is
dropped; one whose file shares no words with the line is kept but reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ARROW = "→"

# A note path: memory/2026-05-12.md, a bare 2026-05-12.md, or a top-level
# file such as todos.md.
_REF_RE = re.compile(r"(?<![\w/.~-])((?:memory/)?[\w-]+\.md)\b")
_DATE_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")
_WORD_RE = re.compile(r"\w{4,}", re.UNICODE)
_NOT_REFS = {"MEMORY.md", "AGENTS.md"}


@dataclass
class RefCheck:
    text: str
    refs: int = 0
    dropped: list[str] = field(default_factory=list)
    unconfirmed: list[str] = field(default_factory=list)


def _resolve(ref: str, memory_dir: Path) -> Path:
    if _DATE_FILE_RE.match(ref):
        ref = f"memory/{ref}"
    return memory_dir / ref


def _shares_a_word(line: str, target: str) -> bool:
    words = {w.lower() for w in _WORD_RE.findall(line.split(ARROW)[0])}
    text = target.lower()
    return any(w in text for w in words)


def check_references(text: str, memory_dir: Path) -> RefCheck:
    """Validate the ``→ file`` references in *text*; return the text with
    references to missing files removed."""
    result = RefCheck(text=text)
    out: list[str] = []
    for line in text.split("\n"):
        if ARROW not in line:
            out.append(line)
            continue
        head, _, tail = line.partition(ARROW)
        refs = [r for r in _REF_RE.findall(tail) if r not in _NOT_REFS]
        kept: list[str] = []
        for ref in refs:
            result.refs += 1
            path = _resolve(ref, memory_dir)
            if not path.is_file():
                result.dropped.append(f"{ref} ({head.strip()[:60]})")
                continue
            if not _shares_a_word(line, path.name + "\n" + path.read_text(errors="replace")):
                result.unconfirmed.append(f"{ref} ({head.strip()[:60]})")
            kept.append(ref)
        if len(kept) == len(refs):
            out.append(line)  # all fine (or an arrow used in prose)
        elif kept:
            out.append(f"{head.rstrip()} {ARROW} {', '.join(kept)}")
        else:
            out.append(head.rstrip())  # every reference was wrong
    result.text = "\n".join(out)
    return result
