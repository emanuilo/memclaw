from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import MemclawConfig
from .index import MemoryIndex
from .search import HybridSearch
from .store import MemoryStore

console = Console()


@click.group(invoke_without_command=True)
@click.option(
    "--memory-dir",
    type=click.Path(),
    default=None,
    help="Path to memory directory (default: ~/.memclaw)",
)
@click.pass_context
def cli(ctx, memory_dir):
    """Memclaw -- your personal memory vault, powered by AI."""
    ctx.ensure_object(dict)
    config = MemclawConfig(memory_dir=memory_dir) if memory_dir else MemclawConfig()
    ctx.obj["config"] = config

    if ctx.invoked_subcommand is None:
        asyncio.run(_interactive(config))


# ------------------------------------------------------------------
# Interactive mode
# ------------------------------------------------------------------

async def _interactive(config: MemclawConfig):
    from .agent import MemclawAgent

    console.print(
        Panel(
            "[bold]Memclaw[/bold] — Your Personal Memory Vault\n\n"
            "Type your thoughts, questions, or commands.\n"
            "Type [bold]/quit[/bold] to exit.",
            title="memclaw",
            border_style="bright_cyan",
        )
    )

    agent = MemclawAgent(config)

    try:
        while True:
            try:
                user_input = console.input("[bold green]> [/bold green]")
            except (EOFError, KeyboardInterrupt):
                break

            stripped = user_input.strip()
            if stripped.lower() in ("/quit", "/exit", "quit", "exit"):
                break
            if not stripped:
                continue

            with console.status("[cyan]Thinking...[/cyan]"):
                response = await agent.chat(stripped)

            if response:
                console.print()
                console.print(Markdown(response))
            console.print()
    finally:
        agent.close()
        console.print("\nGoodbye! Your memories are safe.")


# ------------------------------------------------------------------
# Direct commands (work without the Claude agent / Anthropic key)
# ------------------------------------------------------------------

@cli.command()
@click.argument("content")
@click.option("--permanent", is_flag=True, help="Save to MEMORY.md instead of today's daily file")
@click.pass_context
def save(ctx, content, permanent):
    """Save a memory directly (no agent needed)."""
    config: MemclawConfig = ctx.obj["config"]
    store = MemoryStore(config)
    index = MemoryIndex(config)

    file_path = store.save(content, permanent=permanent)
    asyncio.run(index.index_file(file_path))
    index.close()

    console.print(f"[green]✓[/green] Memory saved to [bold]{file_path.name}[/bold]")


@cli.command()
@click.argument("query_text")
@click.option("--limit", default=5, help="Number of results to return")
@click.pass_context
def search(ctx, query_text, limit):
    """Search your memories (no agent needed)."""
    config: MemclawConfig = ctx.obj["config"]
    index = MemoryIndex(config)
    engine = HybridSearch(config, index)

    results = asyncio.run(engine.search(query_text, limit=limit))
    index.close()

    if not results:
        console.print("[yellow]No matching memories found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        source = Path(r.file_path).stem
        console.print(
            Panel(
                r.content.strip(),
                title=f"[{i}] {source} (score: {r.score:.2f}, {r.match_type})",
                border_style="blue",
            )
        )


@cli.command(name="index")
@click.pass_context
def rebuild_index(ctx):
    """Rebuild the search index from all memory files."""
    config: MemclawConfig = ctx.obj["config"]
    index = MemoryIndex(config)

    changed = asyncio.run(index.sync())
    stats = index.get_stats()
    index.close()

    label = "updated" if changed else "already up to date"
    console.print(f"[green]✓[/green] Index {label}")
    console.print(f"  Chunks: {stats['chunks']}  Files: {stats['files']}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show memory vault status."""
    config: MemclawConfig = ctx.obj["config"]
    store = MemoryStore(config)
    index = MemoryIndex(config)

    files = store.list_files()
    stats = index.get_stats()
    index.close()

    console.print(
        Panel(
            f"Memory directory : {config.memory_dir}\n"
            f"Memory files     : {len(files)}\n"
            f"Indexed chunks   : {stats['chunks']}\n"
            f"Database         : {config.db_path}",
            title="Memclaw Status",
            border_style="bright_cyan",
        )
    )
