"""Tests for Cursor SDK hook installation and policy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from memclaw.backends.cursor.hooks import (
    cursor_hook_script_path,
    cursor_hooks_installed,
    cursor_hooks_json_path,
    cursor_hooks_status,
    ensure_cursor_hooks,
)
from memclaw.backends.tool_policy import HOOKS_VERSION, MEMCLAW_TOOL_NAMES
from memclaw.defaults.cursor_hooks.allow_memclaw_tools import (
    HOOKS_VERSION as INSTALLED_HOOKS_VERSION,
    MEMCLAW_MCP_TOOL_NAMES,
    _resolve_memory_dir,
    is_allowed_file_read,
    is_allowed_mcp_execution,
    is_allowed_shell_execution,
    is_allowed_tool,
)
from memclaw.tools import TOOL_DEFINITIONS


class TestHookPolicy:
    def test_hook_versions_match_tool_policy(self):
        assert INSTALLED_HOOKS_VERSION == HOOKS_VERSION

    def test_hook_tool_names_match_executor(self):
        defined = {defn["name"] for defn in TOOL_DEFINITIONS}
        assert MEMCLAW_MCP_TOOL_NAMES == defined
        assert MEMCLAW_TOOL_NAMES == defined

    @pytest.mark.parametrize(
        ("tool_name", "tool_input", "expected"),
        [
            ("mcp", {"providerIdentifier": "memclaw", "toolName": "memory_save"}, True),
            ("mcp", {"providerIdentifier": "memclaw", "toolName": "mcp_auth"}, False),
            (
                "mcp",
                {"toolName": "memory_save", "args": {"content": "hello"}},
                True,
            ),
            ("mcp", {"server": "memclaw", "toolName": "file_write"}, True),
            ("MCP:memclaw:memory_search", {}, True),
            ("MCP: memclaw:memory_search", {}, True),
            ("memclaw_memory_save", {}, True),
            ("memclaw_reminder_create", {}, True),
            ("memory_save", {"content": "hello"}, True),
            ("file_read", {"file_path": "notes.md"}, True),
            (
                "CallMcpTool",
                {
                    "server": "memclaw",
                    "toolName": "memory_save",
                    "args": {"content": "hello"},
                },
                True,
            ),
            (
                "call_mcp_tool",
                {"providerIdentifier": "memclaw", "toolName": "file_write"},
                True,
            ),
            (
                "CallMcpTool",
                {"toolName": "memory_save", "args": {"content": "hello"}},
                True,
            ),
            (
                "CallMcpTool",
                {"server": "other", "toolName": "memory_save"},
                False,
            ),
            ("Glob", {"globPattern": "**/*"}, False),
            (
                "glob",
                {"globPattern": "**/*", "targetDirectory": "/Users/x/.memclaw"},
                False,
            ),
            ("Read", {"path": "/tmp/x"}, False),
            ("read", {"path": "/tmp/x"}, False),
            ("Grep", {"pattern": "foo"}, False),
            ("grep", {"pattern": "foo"}, False),
            ("Shell", {"command": "ls"}, False),
            ("Task", {"description": "explore"}, False),
            ("mcp", {"providerIdentifier": "other", "toolName": "x"}, False),
            ("mcp", {"toolName": "Write", "args": {"path": "/tmp/x"}}, False),
            ("MCP:other:memory_save", {}, False),
        ],
    )
    def test_is_allowed_tool(self, tool_name, tool_input, expected):
        assert is_allowed_tool(tool_name, tool_input) is expected

    def test_mcp_shorthand_tool_name_from_real_cursor_payload(self):
        payload = {
            "tool_name": "MCP:memory_save",
            "tool_input": {
                "content": "AI video list",
                "permanent": True,
                "entry_type": "link",
                "tags": ["ai", "video"],
            },
            "hook_event_name": "preToolUse",
        }
        assert is_allowed_tool(payload["tool_name"], payload["tool_input"]) is True

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (
                {
                    "hook_event_name": "beforeMCPExecution",
                    "tool_name": "memory_save",
                    "tool_input": '{"content": "hello"}',
                    "url": "http://127.0.0.1:17373/mcp",
                },
                True,
            ),
            (
                {
                    "hook_event_name": "beforeMCPExecution",
                    "tool_name": "memory_save",
                    "tool_input": "{}",
                    "url": "http://localhost:17373/mcp",
                },
                True,
            ),
            (
                {
                    "hook_event_name": "beforeMCPExecution",
                    "tool_name": "memory_save",
                    "url": "http://127.0.0.1:17373",
                },
                True,
            ),
            (
                {
                    "hook_event_name": "beforeMCPExecution",
                    "tool_name": "memory_save",
                    "url": "http://127.0.0.1:17373/mcp/session/abc",
                },
                True,
            ),
            (
                {
                    "hook_event_name": "beforeMCPExecution",
                    "tool_name": "memory_save",
                    "url": "http://127.0.0.1:17374/mcp",
                },
                False,
            ),
            (
                {
                    "hook_event_name": "beforeMCPExecution",
                    "tool_name": "memory_save",
                    "url": "https://example.com/mcp",
                },
                False,
            ),
            (
                {
                    "hook_event_name": "beforeMCPExecution",
                    "command": "memclaw",
                    "tool_name": "memory_save",
                    "tool_input": "{}",
                },
                True,
            ),
            ({"hook_event_name": "beforeReadFile", "file_path": "/tmp/x"}, False),
            (
                {"hook_event_name": "beforeShellExecution", "command": "ls"},
                False,
            ),
        ],
    )
    def test_specialized_hook_events(self, payload, expected):
        event = payload["hook_event_name"]
        if event == "beforeMCPExecution":
            assert is_allowed_mcp_execution(payload) is expected
        elif event == "beforeReadFile":
            assert is_allowed_file_read(payload) is expected
        elif event == "beforeShellExecution":
            assert is_allowed_shell_execution(payload) is expected
        else:
            raise AssertionError(f"unexpected event {event}")

    def test_hook_script_subprocess(self, tmp_path: Path):
        ensure_cursor_hooks(tmp_path)
        script = cursor_hook_script_path(tmp_path)

        def run(payload: dict) -> dict:
            proc = subprocess.run(
                [sys.executable, str(script)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(proc.stdout)

        assert (
            run(
                {
                    "tool_name": "Read",
                    "tool_input": {"path": "/etc/passwd"},
                    "hook_event_name": "preToolUse",
                }
            )["permission"]
            == "deny"
        )
        assert (
            run(
                {
                    "tool_name": "glob",
                    "tool_input": {"globPattern": "**/*"},
                    "hook_event_name": "preToolUse",
                }
            )["permission"]
            == "deny"
        )
        assert (
            run(
                {
                    "tool_name": "mcp",
                    "tool_input": {
                        "providerIdentifier": "memclaw",
                        "toolName": "memory_save",
                        "args": {"content": "hello"},
                    },
                    "hook_event_name": "preToolUse",
                }
            )["permission"]
            == "allow"
        )
        assert (
            run(
                {
                    "tool_name": "mcp",
                    "tool_input": {
                        "toolName": "memory_save",
                        "args": {"content": "hello"},
                    },
                    "hook_event_name": "preToolUse",
                }
            )["permission"]
            == "allow"
        )
        assert (
            run(
                {
                    "hook_event_name": "beforeMCPExecution",
                    "tool_name": "memory_save",
                    "tool_input": '{"content": "hello"}',
                    "url": "http://127.0.0.1:17373/mcp",
                }
            )["permission"]
            == "allow"
        )
        assert (
            run(
                {
                    "tool_name": "CallMcpTool",
                    "tool_input": {
                        "server": "memclaw",
                        "toolName": "memory_save",
                        "args": {"content": "hello"},
                    },
                    "hook_event_name": "preToolUse",
                }
            )["permission"]
            == "allow"
        )
        assert (
            run(
                {
                    "tool_name": "CallMcpTool",
                    "tool_input": {"server": "other", "toolName": "memory_save"},
                    "hook_event_name": "preToolUse",
                }
            )["permission"]
            == "deny"
        )
        assert (
            run(
                {
                    "tool_name": "MCP:memory_save",
                    "tool_input": {
                        "content": "AI video list",
                        "permanent": True,
                        "entry_type": "link",
                        "tags": ["ai"],
                    },
                    "hook_event_name": "preToolUse",
                }
            )["permission"]
            == "allow"
        )
        assert (
            run({"hook_event_name": "beforeReadFile", "file_path": "/tmp/x"})[
                "permission"
            ]
            == "deny"
        )
        assert (
            run({"hook_event_name": "beforeShellExecution", "command": "ls"})[
                "permission"
            ]
            == "deny"
        )

    def test_audit_deny_writes_to_hook_install_memory_dir(
        self, tmp_path: Path, monkeypatch
    ):
        ensure_cursor_hooks(tmp_path)
        script = cursor_hook_script_path(tmp_path)
        monkeypatch.delenv("MEMCLAW_MEMORY_DIR", raising=False)

        proc = subprocess.run(
            [sys.executable, str(script)],
            input=json.dumps(
                {
                    "tool_name": "Read",
                    "tool_input": {"path": "/etc/passwd"},
                    "hook_event_name": "preToolUse",
                }
            ),
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(proc.stdout)["permission"] == "deny"

        log_path = tmp_path / "hook-deny.jsonl"
        assert log_path.is_file()
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert entry["payload"]["tool_name"] == "Read"

    def test_resolve_memory_dir_prefers_env_override(self, tmp_path: Path, monkeypatch):
        ensure_cursor_hooks(tmp_path)
        override = tmp_path / "other-vault"
        monkeypatch.setenv("MEMCLAW_MEMORY_DIR", str(override))
        assert _resolve_memory_dir() == override


class TestHookInstallation:
    def test_ensure_cursor_hooks_installs_files(self, tmp_path: Path):
        assert not cursor_hooks_installed(tmp_path)
        assert ensure_cursor_hooks(tmp_path) is True
        assert cursor_hooks_json_path(tmp_path).is_file()
        data = json.loads(cursor_hooks_json_path(tmp_path).read_text(encoding="utf-8"))
        assert data.get("version") == 1
        pre_tool_use = data["hooks"]["preToolUse"]
        assert len(pre_tool_use) == 2
        assert pre_tool_use[0].get("matcher")
        for hook_name in (
            "preToolUse",
            "beforeMCPExecution",
            "beforeReadFile",
            "beforeShellExecution",
        ):
            for entry in data["hooks"][hook_name]:
                command = entry["command"]
                assert str(cursor_hook_script_path(tmp_path).resolve()) in command
                assert entry.get("failClosed") is True
        assert cursor_hook_script_path(tmp_path).is_file()
        assert cursor_hooks_installed(tmp_path) is True
        assert cursor_hooks_status(tmp_path).startswith("ready")

    def test_ensure_cursor_hooks_refreshes_outdated_script(self, tmp_path: Path):
        ensure_cursor_hooks(tmp_path)
        script = cursor_hook_script_path(tmp_path)
        script.write_text("# memclaw-hooks-version: 0\n", encoding="utf-8")
        assert cursor_hooks_installed(tmp_path) is False
        assert ensure_cursor_hooks(tmp_path) is True
        assert cursor_hooks_installed(tmp_path) is True

    def test_installed_version_matches_package(self, tmp_path: Path):
        ensure_cursor_hooks(tmp_path)
        script = cursor_hook_script_path(tmp_path)
        assert f"memclaw-hooks-version: {HOOKS_VERSION}" in script.read_text(
            encoding="utf-8"
        )
