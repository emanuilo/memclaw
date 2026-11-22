"""The `memclaw` command: reads the arguments and hands each command to the
code that does it (memclaw.bot, memclaw.memory, memclaw.setup)."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from . import backends, bot
from .config import MemclawConfig, reload_env_file
from .memory import MemoryVault
from .setup import needs_setup, print_logo, run_doctor, run_setup

console = Console()


def _require_backend(config: MemclawConfig) -> None:
    """Exit with a helpful message unless the chosen backend can run."""
    problem = backends.configuration_problem(config)
    if problem:
        console.print(f"[red]Error:[/red] {problem}")
        raise SystemExit(1)


def _require_openai(config: MemclawConfig) -> None:
    bot.require_setting(config.openai.api_key, "OPENAI_API_KEY")


def _config(ctx: click.Context) -> MemclawConfig:
    memory_dir = ctx.obj["memory_dir"]
    return MemclawConfig(memory_dir=memory_dir) if memory_dir else MemclawConfig()


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
    ctx.obj["config"] = _config(ctx)

    if ctx.invoked_subcommand is None:
        print_logo()
        if needs_setup():
            # First run: the wizard writes ~/.memclaw/.env; read it back in.
            run_setup(ctx.obj["config"])
            reload_env_file()
            ctx.obj["config"] = _config(ctx)
        config = ctx.obj["config"]
        _require_backend(config)
        _require_openai(config)
        bot.chosen_platform(config).run(config)


@cli.command()
@click.argument("content")
@click.pass_context
def save(ctx, content):
    """Save a memory directly (no agent needed)."""
    config: MemclawConfig = ctx.obj["config"]
    _require_openai(config)
    with MemoryVault(config) as vault:
        file_path = asyncio.run(vault.save(content))
    console.print(f"[green]✓[/green] Memory saved to [bold]{file_path.name}[/bold]")


@cli.command()
@click.argument("query_text")
@click.option("--limit", default=5, help="Number of results to return")
@click.pass_context
def search(ctx, query_text, limit):
    """Search your memories (no agent needed)."""
    config: MemclawConfig = ctx.obj["config"]
    _require_openai(config)
    with MemoryVault(config) as vault:
        results = asyncio.run(vault.search(query_text, limit=limit))

    if not results:
        console.print("[yellow]No matching memories found.[/yellow]")
        return
    for i, r in enumerate(results, 1):
        console.print(
            Panel(
                r.content.strip(),
                title=f"[{i}] {Path(r.file_path).stem} (score: {r.score:.2f}, {r.match_type})",
                border_style="blue",
            )
        )


@cli.command()
@click.option("--since", "since_date", default=None, help="Consolidate daily files after this date (YYYY-MM-DD)")
@click.pass_context
def consolidate(ctx, since_date):
    """Consolidate daily memory files into MEMORY.md."""
    config: MemclawConfig = ctx.obj["config"]
    _require_backend(config)
    _require_openai(config)

    since = None
    if since_date:
        try:
            since = date.fromisoformat(since_date)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid date format: {since_date}. Use YYYY-MM-DD.")
            raise SystemExit(1)

    async def _run(vault: MemoryVault) -> bool:
        backend = backends.build_backend(config)
        with console.status("[cyan]Syncing index...[/cyan]"):
            await vault.sync_or_warn()
        with console.status("[cyan]Running consolidation...[/cyan]"):
            return await vault.consolidate(backend, since=since)

    with MemoryVault(config) as vault:
        if asyncio.run(_run(vault)):
            console.print("[green]Consolidation complete.[/green] MEMORY.md has been updated.")
        else:
            console.print("[yellow]No daily files to consolidate.[/yellow]")


@cli.command(name="index")
@click.pass_context
def rebuild_index(ctx):
    """Rebuild the search index from all memory files."""
    config: MemclawConfig = ctx.obj["config"]
    _require_openai(config)
    with MemoryVault(config) as vault:
        changed, stats = asyncio.run(vault.reindex())

    console.print(f"[green]✓[/green] Index {'updated' if changed else 'already up to date'}")
    console.print(f"  Chunks: {stats['chunks']}  Files: {stats['files']}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show memory vault status."""
    config: MemclawConfig = ctx.obj["config"]
    with MemoryVault(config) as vault:
        rows = [
            *vault.store.status_rows(),
            *bot.status_rows(config),
            *backends.status_rows(config),
            *vault.index.status_rows(),
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
    config: MemclawConfig = ctx.obj["config"]
    _require_openai(config)
    if not run_doctor(config.openai.api_key, console):
        raise SystemExit(1)
