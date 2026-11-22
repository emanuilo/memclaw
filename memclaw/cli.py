from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .bot import PLATFORMS
from .config import MemclawConfig, reload_env_file
from .memory import HybridSearch, MemoryIndex, MemoryStore
from .setup import needs_setup, print_logo, print_probe_report, run_setup

console = Console()


def _require_backend_auth(config: MemclawConfig) -> None:
    """Exit with a helpful message unless the chosen backend is configured."""
    from .backends import REGISTRY, get_backend_class, resolve_backend_name

    name = resolve_backend_name(config)
    try:
        backend_cls = get_backend_class(name)
    except ValueError:
        known = ", ".join(REGISTRY) or "(none)"
        console.print(
            f"[red]Error:[/red] unknown agent backend [bold]{name}[/bold] "
            f"(set via AGENT_BACKEND). Available: {known}.\n"
            "Run [bold]memclaw configure[/bold] to pick a valid backend."
        )
        raise SystemExit(1)
    if backend_cls.is_configured(config):
        return
    console.print(
        f"[red]Error:[/red] {backend_cls.configuration_help()}\n"
        "Run [bold]memclaw configure[/bold] to set it."
    )
    raise SystemExit(1)


def _require_openai(config: MemclawConfig) -> None:
    if not config.openai.api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY is not set.")
        console.print("Run [bold]memclaw configure[/bold] to set it.")
        raise SystemExit(1)


def _ensure_setup(ctx):
    """Run first-time setup if ~/.memclaw/.env doesn't exist, then reload config."""
    if needs_setup():
        run_setup(ctx.obj["config"])
        # Reload .env so newly saved keys are picked up
        reload_env_file()
        memory_dir = ctx.obj.get("memory_dir")
        config = MemclawConfig(memory_dir=memory_dir) if memory_dir else MemclawConfig()
        ctx.obj["config"] = config


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
    ctx.obj["memory_dir"] = memory_dir
    config = MemclawConfig(memory_dir=memory_dir) if memory_dir else MemclawConfig()
    ctx.obj["config"] = config

    if ctx.invoked_subcommand is None:
        print_logo()
        _ensure_setup(ctx)
        config = ctx.obj["config"]
        _require_backend_auth(config)
        _require_openai(config)

        platform = PLATFORMS.get(config.platform)
        if platform is not None:
            platform.run(config)
        else:
            asyncio.run(_interactive(config))


# ------------------------------------------------------------------
# Interactive (terminal) mode
# ------------------------------------------------------------------

async def _interactive(config: MemclawConfig):
    import sys

    from loguru import logger

    from .agent import MemclawAgent

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
    )

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
    finally:
        await agent.aclose()
        console.print("\nGoodbye! Your memories are safe.")


# ------------------------------------------------------------------
# Direct commands (work without the Claude agent / Anthropic key)
# ------------------------------------------------------------------

@cli.command()
@click.argument("content")
@click.pass_context
def save(ctx, content):
    """Save a memory directly (no agent needed)."""
    config: MemclawConfig = ctx.obj["config"]
    store = MemoryStore(config)
    index = MemoryIndex(config)

    file_path = store.save(content)
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


@cli.command()
@click.option("--since", "since_date", default=None, help="Consolidate daily files after this date (YYYY-MM-DD)")
@click.pass_context
def consolidate(ctx, since_date):
    """Consolidate daily memory files into MEMORY.md."""
    from datetime import date as date_type

    from .agent import MemclawAgent

    config: MemclawConfig = ctx.obj["config"]

    _require_backend_auth(config)
    _require_openai(config)

    override = None
    if since_date:
        try:
            override = date_type.fromisoformat(since_date)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid date format: {since_date}. Use YYYY-MM-DD.")
            raise SystemExit(1)

    async def _run():
        agent = MemclawAgent(config)
        try:
            with console.status("[cyan]Syncing index...[/cyan]"):
                await agent.start(include_backend=False)
            with console.status("[cyan]Running consolidation...[/cyan]"):
                result = await agent.consolidator.consolidate(force=True, since=override)
            if result:
                console.print("[green]Consolidation complete.[/green] MEMORY.md has been updated.")
            else:
                console.print("[yellow]No daily files to consolidate.[/yellow]")
        finally:
            await agent.aclose()

    asyncio.run(_run())


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
    from .backends import get_backend_class, resolve_backend_name

    config: MemclawConfig = ctx.obj["config"]
    store = MemoryStore(config)
    index = MemoryIndex(config)

    files = store.list_files()
    stats = index.get_stats()
    index.close()

    backend_cls = get_backend_class(resolve_backend_name(config))
    rows = [
        ("Memory directory", config.memory_dir),
        ("Memory files", len(files)),
        ("Platform", config.platform or "terminal"),
        *backend_cls.status_rows(config),
        ("Indexed chunks", stats["chunks"]),
        ("Stored images", stats["images"]),
        ("Database", config.db_path),
    ]

    console.print(
        Panel(
            "\n".join(f"{label:<16} : {value}" for label, value in rows),
            title="Memclaw Status",
            border_style="bright_cyan",
        )
    )


@cli.command()
@click.pass_context
def configure(ctx):
    """Update API keys, agent backend, and front-end platform."""
    print_logo()
    run_setup(ctx.obj["config"], reconfigure=True)


@cli.command()
@click.pass_context
def doctor(ctx):
    """Check that your OpenAI key can reach every model Memclaw uses."""
    from .openai_client import probe_openai

    config: MemclawConfig = ctx.obj["config"]
    _require_openai(config)

    with console.status("[cyan]Probing OpenAI...[/cyan]"):
        report = asyncio.run(probe_openai(config.openai.api_key))
    print_probe_report(report, console)
    if not report.all_ok:
        raise SystemExit(1)
