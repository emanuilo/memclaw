"""Tests for the shared numbered picker and the setup pickers built on it."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from rich.panel import Panel

from memclaw import setup
from memclaw.prompts import choose


def _choose(answer: str, **kwargs):
    """Run `choose` with Prompt.ask mocked. Returns (index, panel, prompt kwargs)."""
    console = MagicMock()
    with patch("memclaw.prompts.Prompt.ask", return_value=answer) as ask:
        index = choose(console, **kwargs)
    panel = next(c.args[0] for c in console.print.call_args_list
                 if c.args and isinstance(c.args[0], Panel))
    return index, panel, ask.call_args.kwargs


class TestChoose:
    def test_returns_the_zero_based_index(self):
        index, _, _ = _choose("2", title="t", rows=["a", "b", "c"])
        assert index == 1

    def test_default_is_shown_one_based(self):
        _, _, kwargs = _choose("3", title="t", rows=["a", "b", "c"], default=2)
        assert kwargs["default"] == "3"
        assert kwargs["choices"] == ["1", "2", "3"]

    def test_rows_are_numbered_and_joined(self):
        _, panel, _ = _choose("1", title="Pick", rows=["a", "b"], separator="    ")
        assert panel.renderable == "[bold]1)[/bold] a    [bold]2)[/bold] b"
        assert panel.title == "Pick"
        assert panel.border_style == "bright_cyan"

    def test_footer_goes_under_the_rows(self):
        _, panel, _ = _choose("1", title="t", rows=["a"], footer="[dim]note[/dim]")
        assert panel.renderable == "[bold]1)[/bold] a\n\n[dim]note[/dim]"


class TestSetupPickers:
    def test_platform_defaults_to_the_saved_one(self):
        with patch("memclaw.prompts.Prompt.ask", return_value="3") as ask:
            platform = setup._select_platform({"MEMCLAW_PLATFORM": "whatsapp"})
        expected = [name for name, _ in setup.PLATFORMS].index("whatsapp") + 1
        assert ask.call_args.kwargs["default"] == str(expected)
        assert platform == setup.PLATFORMS[2][0]

    def test_backend_defaults_to_the_saved_one(self):
        backends = setup.list_backends()
        with patch("memclaw.prompts.Prompt.ask", return_value="2") as ask:
            name = setup._select_backend({"AGENT_BACKEND": "cursor"})
        expected = [cls.name for cls in backends].index("cursor") + 1
        assert ask.call_args.kwargs["default"] == str(expected)
        assert name == backends[1].name
