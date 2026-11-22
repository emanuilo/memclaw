#!/usr/bin/env python3
"""Claude Code PreToolUse hook: run the code-shape check before `git commit`.

Wired up in .claude/settings.json for the Bash tool. Reads the hook event (JSON)
from stdin. For a command that runs `git commit` in this repo, it runs
.githooks/pre-commit (`code-shape check`). If that fails, it exits 2 with the
check's output on stderr: Claude Code then blocks the commit and shows Claude
what to fix. Every other command passes through (exit 0).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# `git commit`, also after `cd x &&`, `;` or `|`, and with options before the
# subcommand (`git -c k=v commit`).
GIT_COMMIT = re.compile(r"(?:^|[\s;&|(])git(?:\s+-[cC]\s+\S+|\s+--?[\w-]+(?:=\S+)?)*\s+commit(?:\s|$)")


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    command = (event.get("tool_input") or {}).get("command") or ""
    if event.get("tool_name") != "Bash" or not GIT_COMMIT.search(command):
        return 0
    root = _repo_root(event.get("cwd") or ".")
    hook = root / ".githooks" / "pre-commit" if root else None
    if hook is None or not hook.is_file():  # not this project (or not a git repo)
        return 0
    if _git_hook_runs(root, command):  # git runs the same check and blocks on failure
        return 0
    check = subprocess.run(["sh", str(hook)], cwd=root, capture_output=True, text=True, check=False)
    if check.returncode == 0:
        return 0
    sys.stderr.write(check.stdout + check.stderr)
    return 2


def _repo_root(cwd: str) -> Path | None:
    out = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    return Path(out.stdout.strip()) if out.returncode == 0 else None


def _git_hook_runs(root: Path, command: str) -> bool:
    """True when git itself will run .githooks/pre-commit for this commit."""
    if "--no-verify" in command:
        return False
    out = subprocess.run(["git", "-C", str(root), "config", "core.hooksPath"], capture_output=True, text=True, check=False)
    hooks_path = out.stdout.strip()
    return bool(hooks_path) and (root / hooks_path).resolve() == (root / ".githooks").resolve()


if __name__ == "__main__":
    sys.exit(main())
