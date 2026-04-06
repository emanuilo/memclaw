from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from .config import MemclawConfig

# Obsidian callout types for each entry type
_OBSIDIAN_CALLOUT_MAP = {
    "note": "note",
    "image": "example",
    "link": "info",
    "voice": "quote",
}


class MemoryStore:
    """Manages reading and writing memory markdown files.

    Memories are stored as plain Markdown — daily logs in memory/YYYY-MM-DD.md
    and curated long-term facts in MEMORY.md.

    When obsidian_mode is enabled, files include YAML frontmatter, #tag syntax,
    callout blocks, and wikilinks for compatibility with Obsidian.
    """

    def __init__(self, config: MemclawConfig):
        self.config = config

    @property
    def _obsidian(self) -> bool:
        return self.config.obsidian_mode

    # ------------------------------------------------------------------
    # Frontmatter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _daily_frontmatter(dt: date) -> str:
        return (
            "---\n"
            "type: daily-note\n"
            f"date: {dt.isoformat()}\n"
            "tags:\n"
            "  - memclaw\n"
            "  - daily\n"
            "source: memclaw\n"
            "---\n\n"
        )

    @staticmethod
    def _permanent_frontmatter() -> str:
        return (
            "---\n"
            "type: permanent-memory\n"
            "tags:\n"
            "  - memclaw\n"
            "  - memory\n"
            "source: memclaw\n"
            "---\n\n"
        )

    # ------------------------------------------------------------------
    # Entry formatting
    # ------------------------------------------------------------------

    def _format_tags(self, tags: list[str] | None) -> str:
        if not tags:
            return ""
        if self._obsidian:
            return "\n" + " ".join(f"#{t}" for t in tags) + "\n"
        return f"\nTags: {', '.join(tags)}\n"

    def _format_entry(
        self,
        content: str,
        entry_type: str,
        tags: list[str] | None,
    ) -> str:
        now = datetime.now()
        timestamp = now.strftime("%H:%M")
        tag_text = self._format_tags(tags)

        if self._obsidian:
            callout = _OBSIDIAN_CALLOUT_MAP.get(entry_type, "note")
            lines = content.strip().split("\n")
            body = "\n".join(f"> {line}" for line in lines)
            if tag_text.strip():
                body += "\n> " + tag_text.strip()
            entry = f"\n> [!{callout}] {timestamp} - {entry_type.title()}\n{body}\n\n"
        else:
            entry = f"\n## {timestamp} - {entry_type.title()}\n\n"
            entry += content.strip() + "\n"
            entry += tag_text
            entry += "\n---\n"

        return entry

    # ------------------------------------------------------------------
    # File creation
    # ------------------------------------------------------------------

    def _create_daily_header(self, dt: date) -> str:
        header = ""
        if self._obsidian:
            header += self._daily_frontmatter(dt)
        header += f"# {dt.strftime('%A, %B %d, %Y')}\n\n"
        return header

    def _create_permanent_header(self) -> str:
        header = ""
        if self._obsidian:
            header += self._permanent_frontmatter()
        header += "# Personal Memory\n\n"
        return header

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        entry = self._format_entry(content, entry_type, tags)

        if not target.exists():
            if permanent:
                target.write_text(self._create_permanent_header())
            else:
                target.write_text(self._create_daily_header(date.today()))

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
