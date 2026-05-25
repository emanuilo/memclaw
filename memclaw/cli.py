from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .bot import mask_user_id
from .config import MemclawConfig
from .index import MemoryIndex
from .search import HybridSearch
from .setup import needs_setup, print_logo, run_setup
from .store import MemoryStore

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
    if not config.openai_api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY is not set.")
        console.print("Run [bold]memclaw configure[/bold] to set it.")
        raise SystemExit(1)


def _ensure_setup(ctx):
    """Run first-time setup if ~/.memclaw/.env doesn't exist, then reload config."""
    if needs_setup():
        run_setup(memory_dir=ctx.obj.get("memory_dir"))
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
        print_logo()
        _ensure_setup(ctx)
        config = ctx.obj["config"]
        _require_backend_auth(config)
        _require_openai(config)

        platform = config.platform or "terminal"
        if platform == "telegram":
            _run_telegram(config)
        elif platform == "slack":
            _run_slack(config)
        elif platform == "whatsapp":
            _run_whatsapp(config)
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
        await agent.aclose()
        console.print("\nGoodbye! Your memories are safe.")


# ------------------------------------------------------------------
# Bot launchers (dispatched from the bare `memclaw` command)
# ------------------------------------------------------------------

def _setup_bot_logging(log_file: Path) -> None:
    import sys

    from loguru import logger

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
    )
    logger.add(str(log_file), rotation="10 MB", retention="7 days", level="DEBUG")


def _run_telegram(config: MemclawConfig) -> None:
    from loguru import logger
    from openai import AsyncOpenAI
    from telegram.error import NetworkError, TimedOut
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    from .bot.handlers import MessageHandlers

    if not config.telegram_bot_token:
        console.print("[red]Error:[/red] TELEGRAM_BOT_TOKEN is not set.")
        console.print("Run [bold]memclaw configure[/bold] to set it.")
        raise SystemExit(1)

    _setup_bot_logging(config.memory_dir / "bot.log")

    async def post_init(application: Application) -> None:
        openai_client = AsyncOpenAI(api_key=config.openai_api_key)
        handlers = MessageHandlers(config, openai_client)
        application.bot_data["handlers"] = handlers

        await handlers.agent.start()
        await handlers.agent.start_background_sync(interval=60)

        handlers.attach_bot(application.bot)
        handlers.scheduler.start()

        logger.info("Memclaw bot initialized")

    async def post_shutdown(application: Application) -> None:
        handlers = application.bot_data.get("handlers")
        if handlers:
            await handlers.aclose()
            logger.info("Memclaw bot shut down cleanly")

    app = (
        Application.builder()
        .token(config.telegram_bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    async def _start(update, context):
        await context.bot_data["handlers"].start_command(update, context)

    async def _text(update, context):
        await context.bot_data["handlers"].handle_text(update, context)

    async def _photo(update, context):
        await context.bot_data["handlers"].handle_photo(update, context)

    async def _voice(update, context):
        await context.bot_data["handlers"].handle_voice(update, context)

    async def _on_error(update, context):
        err = context.error
        if isinstance(err, (NetworkError, TimedOut)):
            logger.warning(f"Network blip ({type(err).__name__}): {err} — polling will retry")
            return
        logger.exception("Unhandled error in Telegram handler", exc_info=err)

    app.add_error_handler(_on_error)
    app.add_handler(CommandHandler("start", _start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _text))
    app.add_handler(MessageHandler(filters.PHOTO, _photo))
    app.add_handler(MessageHandler(filters.VOICE, _voice))

    allowed = config.allowed_user_ids_list
    masked = [mask_user_id(u) for u in allowed] if allowed else "all"
    console.print(
        f"[green]Starting Memclaw Telegram bot...[/green]  "
        f"(allowed users: {masked})"
    )
    app.run_polling(allowed_updates=["message"])


def _run_whatsapp(config: MemclawConfig) -> None:
    from openai import AsyncOpenAI

    from .bot.whatsapp_handlers import WhatsAppBot

    _setup_bot_logging(config.memory_dir / "whatsapp.log")

    openai_client = AsyncOpenAI(api_key=config.openai_api_key)
    bot_ = WhatsAppBot(config, openai_client)

    console.print("[green]Starting Memclaw WhatsApp bot[/green]  (self-notes only)")
    if not config.whatsapp_session_db.exists():
        console.print(
            "[cyan]First run:[/cyan] a QR code will appear below. "
            "Open WhatsApp → Settings → Linked Devices → Link a Device, and scan it."
        )

    try:
        asyncio.run(bot_.start())
    except KeyboardInterrupt:
        pass
    finally:
        bot_.close()


def _run_slack(config: MemclawConfig) -> None:
    from openai import AsyncOpenAI

    from .bot.slack_handlers import SlackHandlers

    if not config.slack_bot_token:
        console.print("[red]Error:[/red] SLACK_BOT_TOKEN is not set.")
        console.print("Run [bold]memclaw configure[/bold] to set it.")
        raise SystemExit(1)
    if not config.slack_app_token:
        console.print("[red]Error:[/red] SLACK_APP_TOKEN is not set.")
        console.print("Run [bold]memclaw configure[/bold] to set it.")
        raise SystemExit(1)

    _setup_bot_logging(config.memory_dir / "slack.log")

    openai_client = AsyncOpenAI(api_key=config.openai_api_key)
    handlers = SlackHandlers(config, openai_client)

    console.print(
        f"[green]Starting Memclaw Slack bot (Socket Mode)...[/green]  "
        f"(allowed channels: {config.slack_allowed_channels_list or 'all'})"
    )

    async def _run():
        try:
            await handlers.start()
        except KeyboardInterrupt:
            pass
        finally:
            await handlers.aclose()
            console.print("\nMemclaw Slack bot shut down.")

    asyncio.run(_run())


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
                result = await agent._maybe_consolidate(
                    force=True, consolidated_through_override=override
                )
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
            f"Platform         : {config.platform or 'terminal'}\n"
            f"Indexed chunks   : {stats['chunks']}\n"
            f"Stored images    : {stats['images']}\n"
            f"Database         : {config.db_path}",
            title="Memclaw Status",
            border_style="bright_cyan",
        )
    )


@cli.command()
@click.pass_context
def configure(ctx):
    """Update API keys, agent backend, and front-end platform."""
    print_logo()
    run_setup(reconfigure=True, memory_dir=ctx.obj.get("memory_dir"))


@cli.command()
@click.pass_context
def doctor(ctx):
    """Check that your OpenAI key can reach every model Memclaw uses."""
    from .openai_health import print_probe_report, probe_openai

    config: MemclawConfig = ctx.obj["config"]
    _require_openai(config)

    with console.status("[cyan]Probing OpenAI...[/cyan]"):
        report = asyncio.run(probe_openai(config.openai_api_key))
    print_probe_report(report, console)
    if not report.all_ok:
        raise SystemExit(1)
