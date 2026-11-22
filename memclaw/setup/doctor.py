"""OpenAI model access: the check the wizard runs after saving, and the
report `memclaw doctor` prints."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..openai_client import ProbeReport, probe_openai


def check_openai(api_key: str | None, console: Console) -> None:
    """Verify OpenAI model access so the user finds out about disabled models
    right now, not when their first voice message silently disappears."""
    if not api_key:
        return
    console.print("\n[cyan]Checking OpenAI model access...[/cyan]")
    try:
        report = asyncio.run(probe_openai(api_key))
        print_probe_report(report, console)
    except Exception as exc:
        console.print(f"[yellow]Skipping model check:[/yellow] {exc}")


def print_probe_report(report: ProbeReport, console: Console) -> None:
    """Render a probe report (memclaw.openai_client.probe_openai) as a panel."""
    body = Text()
    for outcome in report:
        marker = "[green]✓[/green]" if outcome.ok else "[red]✗[/red]"
        line = f"{marker} {outcome.name}  [dim]({outcome.model})[/dim]"
        if not outcome.ok and outcome.error:
            line += f"\n  [red]{outcome.error}[/red]"
        body.append_text(Text.from_markup(line))
        body.append("\n")

    if not report.all_ok:
        body.append("\n", style="")
        body.append_text(
            Text.from_markup(
                "[yellow]Enable the missing models for your project at[/yellow]\n"
                "[bold]https://platform.openai.com/settings/organization/limits[/bold]"
            )
        )

    console.print(
        Panel(
            body,
            title="OpenAI model access",
            border_style="green" if report.all_ok else "yellow",
        )
    )
