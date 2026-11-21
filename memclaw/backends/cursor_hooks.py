"""Install and verify Cursor SDK hooks that restrict tool use to Memclaw MCP."""

from __future__ import annotations

import importlib.resources
import json
import re
import shutil
import sys
from pathlib import Path

from .mcp_tools import MCP_SERVER_NAME
from .tool_policy import CURSOR_PRETOOLUSE_MATCHER, HOOKS_VERSION, hook_policy_payload

_HOOKS_JSON = "hooks.json"
_HOOK_SCRIPT = "allow_memclaw_tools.py"
_HOOK_POLICY_JSON = "hook_policy.json"
_VERSION_MARKER = re.compile(r"memclaw-hooks-version:\s*(\d+)")


def cursor_hooks_dir(memory_dir: Path) -> Path:
    return Path(memory_dir) / ".cursor"


def cursor_hook_script_path(memory_dir: Path) -> Path:
    return cursor_hooks_dir(memory_dir) / "hooks" / _HOOK_SCRIPT


def cursor_hooks_json_path(memory_dir: Path) -> Path:
    return cursor_hooks_dir(memory_dir) / _HOOKS_JSON


def cursor_hook_policy_path(memory_dir: Path) -> Path:
    return cursor_hooks_dir(memory_dir) / "hooks" / _HOOK_POLICY_JSON


def _packaged_defaults() -> Path:
    return Path(str(importlib.resources.files("memclaw.defaults") / "cursor_hooks"))


def _installed_hook_version(script_path: Path) -> int | None:
    if not script_path.is_file():
        return None
    match = _VERSION_MARKER.search(script_path.read_text(encoding="utf-8"))
    if not match:
        return None
    return int(match.group(1))


def _write_hook_policy(hooks_dir: Path, memory_dir: Path) -> None:
    from ..config import MemclawConfig  # noqa: PLC0415 - known violation, to fix: move to top

    cfg = MemclawConfig(memory_dir=memory_dir)
    policy_path = hooks_dir / _HOOK_POLICY_JSON
    policy_path.write_text(
        json.dumps(
            hook_policy_payload(mcp_http_port=cfg.mcp_http_port),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_hooks_json(memory_dir: Path, script_path: Path) -> None:
    """Write hooks.json with an absolute command path for reliable execution."""
    command = f"{sys.executable} {script_path}"
    hook_entry = {"command": command, "failClosed": True}
    config = {
        "version": 1,
        "hooks": {
            "preToolUse": [
                {
                    "command": command,
                    "matcher": CURSOR_PRETOOLUSE_MATCHER,
                    "failClosed": True,
                },
                hook_entry,
            ],
            "beforeMCPExecution": [hook_entry],
            "beforeReadFile": [hook_entry],
            "beforeShellExecution": [hook_entry],
        },
    }
    cursor_hooks_json_path(memory_dir).write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )


def cursor_hooks_installed(memory_dir: Path) -> bool:
    """Return True when Memclaw's Cursor preToolUse hook is present and current."""
    hooks_json = cursor_hooks_json_path(memory_dir)
    script_path = cursor_hook_script_path(memory_dir)
    policy_path = cursor_hook_policy_path(memory_dir)
    if (
        not hooks_json.is_file()
        or not script_path.is_file()
        or not policy_path.is_file()
    ):
        return False
    return _installed_hook_version(script_path) == HOOKS_VERSION


def ensure_cursor_hooks(memory_dir: Path) -> bool:
    """Install or refresh Cursor hooks under *memory_dir*.

    Returns True when hooks are ready to use after this call.
    """
    memory_dir = Path(memory_dir)
    packaged = _packaged_defaults()
    target_hooks = cursor_hooks_dir(memory_dir) / "hooks"
    target_hooks.mkdir(parents=True, exist_ok=True)

    script_path = cursor_hook_script_path(memory_dir)
    shutil.copy2(packaged / _HOOK_SCRIPT, script_path)
    script_path.chmod(0o755)
    _write_hook_policy(target_hooks, memory_dir)
    _write_hooks_json(memory_dir, script_path.resolve())

    return cursor_hooks_installed(memory_dir)


def cursor_hooks_status(memory_dir: Path) -> str:
    """Human-readable hook status for logs and configuration help."""
    hooks_json = cursor_hooks_json_path(memory_dir)
    script_path = cursor_hook_script_path(memory_dir)
    policy_path = cursor_hook_policy_path(memory_dir)
    if (
        not hooks_json.is_file()
        or not script_path.is_file()
        or not policy_path.is_file()
    ):
        return "missing"
    version = _installed_hook_version(script_path)
    if version != HOOKS_VERSION:
        return f"outdated (found v{version}, need v{HOOKS_VERSION})"
    try:
        config = json.loads(hooks_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid hooks.json"
    for hook_name in (
        "preToolUse",
        "beforeMCPExecution",
        "beforeReadFile",
        "beforeShellExecution",
    ):
        entries = config.get("hooks", {}).get(hook_name, [])
        if not entries:
            return f"{hook_name} hook not configured"
        for entry in entries:
            command = entry.get("command", "")
            if (
                str(script_path.resolve()) not in command
                and _HOOK_SCRIPT not in command
            ):
                return f"{hook_name} hook points elsewhere"
    return f"ready (memclaw MCP provider={MCP_SERVER_NAME!r}, setting_sources=project)"
