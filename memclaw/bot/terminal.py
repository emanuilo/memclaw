"""The terminal: chat with Memclaw in the shell that runs `memclaw`.

Unlike the chat apps there's nothing to download or transcribe: a line
typed at the prompt goes straight to the agent, and its reply is printed
as Markdown.
"""

from __future__ import annotations

import asyncio

from rich.markdown import Markdown
from rich.panel import Panel

from ..agent import MemclawAgent
from ..config import MemclawConfig
from . import console, log_to

_QUIT = ("/quit", "/exit", "quit", "exit")


async def chat(agent: MemclawAgent) -> None:
    """Read messages from the terminal and print the agent's replies, until
    the user quits (`/quit`, Ctrl-D or Ctrl-C). `/new` starts a fresh
    conversation."""
    while True:
        try:
            user_input = console.input("[bold green]> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            return

        stripped = user_input.strip()
        if stripped.lower() in _QUIT:
            return
        if not stripped:
            continue
        if stripped.lower() == "/new":
            await agent.reset_conversation()
            console.print("[cyan]Started a new conversation.[/cyan]\n")
            continue

        try:
            with console.status("[cyan]Thinking...[/cyan]"):
                response, _images = await agent.handle(stripped)
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}\n")
            continue

        if response:
            console.print()
            console.print(Markdown(response))
        console.print()


async def _run(config: MemclawConfig) -> None:
    console.print(
        Panel(
            "[bold]Memclaw[/bold] — Your Personal Memory Vault\n\n"
            "Type your thoughts, questions, or commands.\n"
            "Type [bold]/new[/bold] to start a fresh conversation, "
            "[bold]/quit[/bold] to exit.",
            title="memclaw",
            border_style="bright_cyan",
        )
    )

    agent = MemclawAgent(config)

    # Spec #9: run a full index sync once at startup
    with console.status("[cyan]Syncing index...[/cyan]"):
        await agent.start()

    try:
        await chat(agent)
    finally:
        await agent.aclose()
        console.print("\nGoodbye! Your memories are safe.")


def run(config: MemclawConfig) -> None:
    """Chat in the terminal and block until the user quits."""
    log_to()
    asyncio.run(_run(config))
