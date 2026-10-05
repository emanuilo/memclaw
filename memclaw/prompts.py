"""Shared interactive prompts for the setup wizard.

Lives outside setup.py so backends can import it at module level: setup.py
imports the backend registry, so backends importing setup.py would be
circular.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.prompt import Prompt

if TYPE_CHECKING:
    from rich.console import Console


def choose(
    console: "Console",
    *,
    title: str,
    rows: list[str],
    default: int = 0,
    footer: str = "",
    separator: str = "\n",
) -> int:
    """Show *rows* as a numbered panel and return the 0-based chosen index.

    *default* is the 0-based index preselected when the user presses Enter.
    *separator* joins the numbered rows — "\\n\\n" spaces them out, a run of
    spaces puts them on one line. *footer* is printed under the rows.
    """
    body = separator.join(
        f"[bold]{i})[/bold] {row}" for i, row in enumerate(rows, 1)
    )
    if footer:
        body = f"{body}\n\n{footer}"

    console.print()
    console.print(Panel(body, title=title, border_style="bright_cyan"))
    choice = Prompt.ask(
        "Choose",
        choices=[str(i) for i in range(1, len(rows) + 1)],
        default=str(default + 1),
    )
    return int(choice) - 1
