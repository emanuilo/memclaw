from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from .config import MemclawConfig


class MemoryStore:
    """Manages reading and writing memory markdown files.

    Memories are stored as plain Markdown — daily logs in memory/YYYY-MM-DD.md
    and curated long-term facts in MEMORY.md.
    """

    def __init__(self, config: MemclawConfig):
        self.config = config

    def save(
        self,
        content: str,
        *,
        permanent: bool = False,
        entry_type: str = "note",
        tags: list[str] | None = None,
    ) -> Path:
        """Append a new entry to a memory file.

        Args:
            content: The text content to save.
            permanent: If True, append to MEMORY.md. Otherwise, use today's daily file.
            entry_type: Type label (note, image, link, voice).
            tags: Optional tags for categorization.

        Returns:
            Path to the file where content was saved.
        """
        target = self.config.memory_file if permanent else self.config.daily_file()

        now = datetime.now()
        timestamp = now.strftime("%H:%M")

        entry = f"\n## {timestamp} - {entry_type.title()}\n\n"
        entry += content.strip() + "\n"
        if tags:
            entry += f"\nTags: {', '.join(tags)}\n"
        entry += "\n---\n"

        if not target.exists():
            header = (
                "# Personal Memory\n\n"
                if permanent
                else f"# {date.today().strftime('%A, %B %d, %Y')}\n\n"
            )
            target.write_text(header)

        with open(target, "a") as f:
            f.write(entry)

        return target

    def read_file(self, path: Path) -> str:
        if path.exists():
            return path.read_text()
        return ""

    def list_files(self) -> list[Path]:
        """List all memory markdown files, MEMORY.md first then daily files sorted."""
        files = []
        if self.config.memory_file.exists():
            files.append(self.config.memory_file)
        files.extend(sorted(self.config.memory_subdir.glob("*.md")))
        return files

    def get_all_content(self) -> list[tuple[Path, str]]:
        """Read all memory files and return (path, content) pairs."""
        result = []
        for path in self.list_files():
            content = self.read_file(path)
            if content.strip():
                result.append((path, content))
        return result

    _DAILY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")

    def list_unconsolidated_files(self, consolidated_through: date | None = None) -> list[Path]:
        """List daily files whose date is after *consolidated_through*.

        If *consolidated_through* is None, returns all daily files.
        Results are sorted by date ascending.
        """
        daily_files: list[tuple[date, Path]] = []
        for path in self.config.memory_subdir.glob("*.md"):
            m = self._DAILY_RE.match(path.name)
            if m is None:
                continue
            file_date = date.fromisoformat(m.group(1))
            if consolidated_through is not None and file_date <= consolidated_through:
                continue
            daily_files.append((file_date, path))

        daily_files.sort(key=lambda t: t[0])
        return [path for _, path in daily_files]
