"""Asking the user something in the terminal: a numbered choice, masked input.

Used by the setup wizard (memclaw.setup) and by the backends, which ask
their own questions there (`wizard_setup`). It sits in a layer below the
backends, not inside memclaw.setup: setup imports the backend registry, so
backends importing setup would import upward.
"""

from __future__ import annotations

import sys
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


def masked_input(console: "Console", prompt_text: str, *, visible: int = 4) -> str:
    """Read a line from stdin, echoing only the first `visible` chars verbatim
    and '*' for everything after. Used for API keys and other config values
    during the wizard so pasted secrets don't sit in the scrollback in clear.
    """
    console.print(prompt_text, end=": ")
    sys.stdout.flush()

    if not sys.stdin.isatty():
        line = sys.stdin.readline().rstrip("\n")
        sys.stdout.write("\n")
        sys.stdout.flush()
        return line

    try:
        import termios  # POSIX-only
        import tty  # POSIX-only
    except ImportError:
        # Non-POSIX (e.g. Windows): fall back to unmasked input.
        return input()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return _read_masked(visible)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\n")
        sys.stdout.flush()


def _read_masked(visible: int) -> str:
    """Read keys up to Enter from a terminal in cbreak mode, echoing the
    first *visible* characters and '*' for the rest. Backspace deletes."""
    buf: list[str] = []
    while True:
        ch = sys.stdin.read(1)
        if ch in ("\r", "\n"):
            return "".join(buf)
        if ch == "\x03":  # Ctrl-C
            raise KeyboardInterrupt
        if ch in ("\x7f", "\b"):
            if buf:
                buf.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        if ord(ch) < 32:
            continue
        buf.append(ch)
        sys.stdout.write(ch if len(buf) <= visible else "*")
        sys.stdout.flush()
