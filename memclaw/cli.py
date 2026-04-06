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
from .setup import needs_setup, run_setup
from .store import MemoryStore

console = Console()


def _ensure_setup(ctx):
    """Run first-time setup if ~/.memclaw/.env doesn't exist, then reload config."""
    if needs_setup():
        run_setup()
        # Reload .env so newly saved keys are picked up
        from dotenv import load_dotenv
        load_dotenv(Path.home() / ".memclaw" / ".env", override=True)
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
        _ensure_setup(ctx)
        config = ctx.obj["config"]
        if not config.anthropic_api_key:
            console.print("[red]Error:[/red] ANTHROPIC_API_KEY is not set.")
            console.print("Run [bold]memclaw configure[/bold] to set it.")
            raise SystemExit(1)
        if not config.openai_api_key:
            console.print("[red]Error:[/red] OPENAI_API_KEY is not set.")
            console.print("Run [bold]memclaw configure[/bold] to set it.")
            raise SystemExit(1)
        asyncio.run(_interactive(config))


# ------------------------------------------------------------------
# Interactive mode
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
            "Type [bold]/quit[/bold] to exit.",
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


@cli.command()
@click.option("--since", "since_date", default=None, help="Consolidate daily files after this date (YYYY-MM-DD)")
@click.pass_context
def consolidate(ctx, since_date):
    """Consolidate daily memory files into MEMORY.md."""
    from datetime import date as date_type

    from .agent import MemclawAgent

    config: MemclawConfig = ctx.obj["config"]

    if not config.anthropic_api_key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY is not set.")
        raise SystemExit(1)
    if not config.openai_api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY is not set.")
        raise SystemExit(1)

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
            with console.status("[cyan]Running consolidation...[/cyan]"):
                result = await agent._maybe_consolidate(
                    force=True, consolidated_through_override=override
                )
            if result:
                console.print("[green]Consolidation complete.[/green] MEMORY.md has been updated.")
            else:
                console.print("[yellow]No daily files to consolidate.[/yellow]")
        finally:
            agent.close()

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
            f"Stored images    : {stats['images']}\n"
            f"Database         : {config.db_path}",
            title="Memclaw Status",
            border_style="bright_cyan",
        )
    )


# ------------------------------------------------------------------
# Telegram bot
# ------------------------------------------------------------------

@cli.command()
@click.pass_context
def configure(ctx):
    """Update API keys and settings."""
    run_setup(reconfigure=True)


@cli.command()
@click.pass_context
def bot(ctx):
    """Start the Memclaw Telegram bot."""
    import sys

    from loguru import logger
    from openai import AsyncOpenAI
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    from .bot.handlers import MessageHandlers

    _ensure_setup(ctx)
    config: MemclawConfig = ctx.obj["config"]

    if not config.telegram_bot_token:
        console.print("[red]Error:[/red] TELEGRAM_BOT_TOKEN is not set.")
        console.print("Run [bold]memclaw configure[/bold] to set it.")
        raise SystemExit(1)

    if not config.openai_api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY is not set.")
        console.print("Run [bold]memclaw configure[/bold] to set it.")
        raise SystemExit(1)

    # Logging
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>")
    logger.add(
        str(config.memory_dir / "bot.log"),
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )

    async def post_init(application: Application) -> None:
        openai_client = AsyncOpenAI(api_key=config.openai_api_key)
        handlers = MessageHandlers(config, openai_client)
        application.bot_data["handlers"] = handlers

        # Spec #9: run startup sync and start periodic background sync
        await handlers.agent.start()
        await handlers.agent.start_background_sync(interval=60)

        logger.info("Memclaw bot initialized")

    async def post_shutdown(application: Application) -> None:
        handlers = application.bot_data.get("handlers")
        if handlers:
            handlers.close()
            logger.info("Memclaw bot shut down cleanly")

    app = (
        Application.builder()
        .token(config.telegram_bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # Thin wrappers that delegate to the handlers instance
    async def _start(update, context):
        await context.bot_data["handlers"].start_command(update, context)

    async def _text(update, context):
        await context.bot_data["handlers"].handle_text(update, context)

    async def _photo(update, context):
        await context.bot_data["handlers"].handle_photo(update, context)

    async def _voice(update, context):
        await context.bot_data["handlers"].handle_voice(update, context)

    app.add_handler(CommandHandler("start", _start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _text))
    app.add_handler(MessageHandler(filters.PHOTO, _photo))
    app.add_handler(MessageHandler(filters.VOICE, _voice))

    console.print(
        f"[green]Starting Memclaw Telegram bot...[/green]  "
        f"(allowed users: {config.allowed_user_ids_list or 'all'})"
    )
    app.run_polling(allowed_updates=["message"])


# ------------------------------------------------------------------
# Obsidian integration
# ------------------------------------------------------------------

@cli.command(name="obsidian-init")
@click.argument("vault_path", type=click.Path())
@click.option(
    "--subfolder",
    default="memclaw",
    help="Subfolder name within the vault (default: memclaw)",
)
def obsidian_init(vault_path, subfolder):
    """Set up Memclaw inside an Obsidian vault for cross-device sync.

    VAULT_PATH is the root of your Obsidian vault (the folder containing .obsidian/).

    This command creates a memclaw subfolder in the vault and configures
    Memclaw to write Obsidian-flavored Markdown with frontmatter, #tags,
    wikilinks, and callouts.

    Sync is handled by your Obsidian sync method (iCloud, Obsidian Sync,
    Obsidian Git, Remotely Save, etc.).
    """
    from .setup import ENV_FILE, _load_existing

    vault = Path(vault_path).expanduser().resolve()

    if not vault.is_dir():
        console.print(f"[red]Error:[/red] {vault} is not a directory.")
        raise SystemExit(1)

    obsidian_dir = vault / ".obsidian"
    if not obsidian_dir.exists():
        console.print(
            f"[yellow]Warning:[/yellow] No .obsidian/ found in {vault}.\n"
            "  This folder may not be an Obsidian vault yet.\n"
            "  You can still proceed — Obsidian will create .obsidian/ when you open it."
        )

    memclaw_dir = vault / subfolder
    memclaw_dir.mkdir(parents=True, exist_ok=True)
    (memclaw_dir / "memory").mkdir(exist_ok=True)

    # Update .env with Obsidian settings
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing()
    existing["OBSIDIAN_VAULT_PATH"] = str(memclaw_dir)
    existing["OBSIDIAN_MODE"] = "true"
    lines = [f"{k}={v}" for k, v in existing.items() if v]
    ENV_FILE.write_text("\n".join(lines) + "\n")

    # Create initial files with Obsidian frontmatter
    config = MemclawConfig(
        memory_dir=memclaw_dir,
        obsidian_mode=True,
        openai_api_key=existing.get("OPENAI_API_KEY", "placeholder"),
        anthropic_api_key=existing.get("ANTHROPIC_API_KEY", "placeholder"),
    )
    store = MemoryStore(config)

    # Create MEMORY.md with frontmatter if it doesn't exist
    if not config.memory_file.exists():
        config.memory_file.write_text(store._create_permanent_header())

    console.print(
        Panel(
            f"[bold green]Obsidian integration configured![/bold green]\n\n"
            f"Vault path    : {vault}\n"
            f"Memclaw folder: {memclaw_dir}\n"
            f"Obsidian mode : enabled\n\n"
            "Your memories will now be written with Obsidian-flavored Markdown:\n"
            "  - YAML frontmatter for properties\n"
            "  - #tags for categorization\n"
            "  - Callout blocks for visual structure\n"
            "  - [[wikilinks]] in consolidation\n\n"
            "[dim]Sync is handled by your Obsidian sync method.\n"
            "Supported: iCloud, Obsidian Sync, Obsidian Git,\n"
            "Remotely Save (S3/Dropbox/WebDAV), LiveSync (CouchDB).[/dim]",
            title="memclaw obsidian-init",
            border_style="bright_cyan",
        )
    )
