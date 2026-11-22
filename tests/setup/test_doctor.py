"""Tests for the OpenAI model-access report (`memclaw doctor`, and the wizard after saving)."""
from __future__ import annotations

from unittest.mock import patch

from rich.console import Console

from memclaw.openai_client import ProbeOutcome, ProbeReport
from memclaw.setup import doctor, print_probe_report


def _report(chat_ok: bool) -> ProbeReport:
    return ProbeReport(
        ProbeOutcome("Embeddings (search)", "text-embedding-3-small", True),
        ProbeOutcome("Chat (link summaries)", "gpt-4o-mini", chat_ok, None if chat_ok else "no access"),
        ProbeOutcome("Audio (voice messages)", "whisper-1", True),
    )


def _console() -> Console:
    return Console(record=True, width=80, color_system=None)


class TestProbeReport:
    def test_all_models_reachable(self):
        console = _console()
        print_probe_report(_report(chat_ok=True), console)
        text = console.export_text()
        assert text.count("✓") == 3 and "✗" not in text
        assert "Enable the missing models" not in text

    def test_a_missing_model_shows_its_error_and_where_to_enable_it(self):
        console = _console()
        print_probe_report(_report(chat_ok=False), console)
        text = console.export_text()
        assert "✗ Chat (link summaries)  (gpt-4o-mini)" in text
        assert "no access" in text
        assert "platform.openai.com/settings/organization/limits" in text


class TestCheckOpenAI:
    def test_no_key_checks_nothing(self):
        console = _console()
        with patch.object(doctor, "probe_openai") as probe:
            doctor.check_openai(None, console)
        probe.assert_not_called()
        assert console.export_text() == ""

    def test_a_failing_probe_is_skipped_not_raised(self):
        console = _console()

        async def _fail(_key):
            raise RuntimeError("offline")

        with patch.object(doctor, "probe_openai", _fail):
            doctor.check_openai("sk-1", console)
        assert "Skipping model check: offline" in console.export_text()
