"""WhatsApp: only your notes to yourself are handled. (What WhatsApp shares
with the other platforms is tested in test_parity.py.)"""
from __future__ import annotations


# ────────────────────────────────────────────────────────────────────
# WhatsApp self-chat scoping (regression: outgoing DMs to friends
# were being processed because IsFromMe alone matches them too)
# ────────────────────────────────────────────────────────────────────

class TestWhatsAppSelfChatOnly:
    """Read whatsapp.py source directly to avoid needing neonize installed."""

    def _source(self) -> str:
        from pathlib import Path
        path = Path(__file__).parents[2] / "memclaw" / "bot" / "whatsapp.py"
        return path.read_text()

    def test_check_sender_requires_self_chat(self):
        """_check_sender must compare Chat.User to Sender.User, not just IsFromMe."""
        import ast
        src = self._source()
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_check_sender"
        )
        body_src = ast.unparse(fn)
        assert "Chat.User" in body_src and "Sender.User" in body_src, (
            "_check_sender must require Chat.User == Sender.User to scope to the "
            "self-chat — IsFromMe alone matches outgoing DMs to friends too"
        )

    def test_check_sender_behaviour(self):
        """Simulate the three relevant cases against the real method."""
        from types import SimpleNamespace
        import ast

        src = self._source()
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_check_sender"
        )
        ns: dict = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<_check_sender>", "exec"), ns)
        check = ns["_check_sender"]

        def ev(*, is_group: bool, is_from_me: bool, chat_user: str, sender_user: str):
            return SimpleNamespace(
                Info=SimpleNamespace(MessageSource=SimpleNamespace(
                    IsGroup=is_group,
                    IsFromMe=is_from_me,
                    Chat=SimpleNamespace(User=chat_user),
                    Sender=SimpleNamespace(User=sender_user),
                ))
            )

        self_note = ev(is_group=False, is_from_me=True, chat_user="me", sender_user="me")
        out_to_friend = ev(is_group=False, is_from_me=True, chat_user="friend", sender_user="me")
        in_from_friend = ev(is_group=False, is_from_me=False, chat_user="friend", sender_user="friend")
        group = ev(is_group=True, is_from_me=True, chat_user="grp", sender_user="me")

        assert check(None, self_note) is True
        assert check(None, out_to_friend) is False, "outgoing DM to friend must be ignored"
        assert check(None, in_from_friend) is False
        assert check(None, group) is False


# ────────────────────────────────────────────────────────────────────
# Shutdown: Ctrl+C runs the async shutdown (the AI backend's aclose)
# ────────────────────────────────────────────────────────────────────

class TestShutdown:
    def _run(self, monkeypatch, tmp_path, idle) -> list[str]:
        import tests.bot.test_parity  # noqa: F401  (stands in for neonize without libmagic)
        import memclaw.bot.whatsapp as whatsapp
        from memclaw.config import MemclawConfig

        events: list[str] = []

        class FakeInbound:
            def __init__(self, config, platform):
                pass

            def deliver_reminders(self, send):
                pass

            async def start(self):
                events.append("start")

            async def aclose(self):
                events.append("aclose")

            def close(self):
                events.append("close")

        class FakeClient:
            def __init__(self, path):
                pass

            def event(self, ev_type):
                return lambda fn: fn

            async def connect(self):
                pass

            async def idle(self):
                await idle()

        monkeypatch.setattr(whatsapp, "Inbound", FakeInbound)
        monkeypatch.setattr(whatsapp, "NewAClient", FakeClient)
        monkeypatch.setattr(whatsapp, "log_to", lambda *a: None)
        whatsapp.run(MemclawConfig(memory_dir=tmp_path))
        return events

    def test_ctrl_c_shuts_the_backend_down(self, monkeypatch, tmp_path):
        import asyncio
        import os
        import signal

        async def ctrl_c():
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.sleep(5)

        assert self._run(monkeypatch, tmp_path, ctrl_c) == ["start", "aclose"]

    def test_disconnect_shuts_the_backend_down(self, monkeypatch, tmp_path):
        async def disconnected():
            pass

        assert self._run(monkeypatch, tmp_path, disconnected) == ["start", "aclose"]
