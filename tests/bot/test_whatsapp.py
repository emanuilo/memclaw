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
