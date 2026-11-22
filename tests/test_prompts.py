"""Tests for the shared numbered picker and the setup pickers built on it."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.panel import Panel

from memclaw import setup
from memclaw.prompts import _read_masked, choose


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


class TestMaskedKeys:
    """The key loop masked_input runs on a real terminal."""

    def _read(self, keys: str, visible: int = 4) -> tuple[str, str]:
        out = io.StringIO()
        with patch("sys.stdin", io.StringIO(keys)), patch("sys.stdout", out):
            value = _read_masked(visible)
        return value, out.getvalue()

    def test_echoes_the_first_chars_then_stars(self):
        assert self._read("sk-abcdef\n") == ("sk-abcdef", "sk-a*****")

    def test_backspace_deletes_and_control_keys_are_ignored(self):
        value, echo = self._read("ab\x7f\x7f\x7fc\x01d\r")
        assert value == "cd"
        assert echo == "ab\b \b\b \bcd"

    def test_ctrl_c_interrupts(self):
        with pytest.raises(KeyboardInterrupt):
            self._read("ab\x03")


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
