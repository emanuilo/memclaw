#!/usr/bin/env python3
"""Claude Code PreToolUse hook: run the code-shape check before `git commit`.

Wired up in .claude/settings.json for the Bash tool. Reads the hook event (JSON)
from stdin. For a command that runs `git commit` in this repo, it runs
.githooks/pre-commit (`code-shape check`), unless git will run that hook itself.
If the check fails, it exits 2 with the check's output on stderr: Claude Code
then blocks the commit and shows Claude what to fix. Every other command passes
through (exit 0).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Fallback for a command shlex can't split (e.g. an unclosed quote): `git commit`,
# also after `cd x &&`, `;` or `|`, and with options before the subcommand.
GIT_COMMIT = re.compile(r"(?:^|[\s;&|(])git(?:\s+-[cC]\s+\S+|\s+--?[\w-]+(?:=\S+)?)*\s+commit(?:\s|$)")
# Tokens that end one shell command and start the next.
SEPARATORS = {";", "&", "&&", "|", "||", "|&", "(", ")", "\n"}
# git options before the subcommand that take the next word as their value.
GIT_OPTIONS_WITH_VALUE = {"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--config-env"}
# `git commit` short options whose value is the rest of the word or the next word
# (in `-am msg` the `m` takes `msg`; in `-mn` the message is "n").
COMMIT_SHORT_WITH_VALUE = set("mFCct")
# Short options whose optional value can only be attached (`-S<key>`, `-uno`).
COMMIT_SHORT_ATTACHED_VALUE = set("Su")
# `git commit` long options that may take the next word as their value.
COMMIT_LONG_WITH_VALUE = {
    "--message", "--file", "--reuse-message", "--reedit-message", "--fixup", "--squash",
    "--author", "--date", "--template", "--cleanup", "--trailer", "--pathspec-from-file",
}


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    command = (event.get("tool_input") or {}).get("command") or ""
    if event.get("tool_name") != "Bash":
        return 0
    skips_hooks = commit_skips_git_hooks(command)
    if skips_hooks is None:  # no `git commit` in the command
        return 0
    root = _repo_root(event.get("cwd") or ".")
    hook = root / ".githooks" / "pre-commit" if root else None
    if hook is None or not hook.is_file():  # not this project (or not a git repo)
        return 0
    if not skips_hooks and _git_hook_enabled(root):  # git runs the same check and blocks on failure
        return 0
    check = subprocess.run(["sh", str(hook)], cwd=root, capture_output=True, text=True, check=False)
    if check.returncode == 0:
        return 0
    sys.stderr.write(check.stdout + check.stderr)
    return 2


def commit_skips_git_hooks(command: str) -> bool | None:
    """Whether the `git commit` in a shell command skips git's pre-commit hook.

    None when the command runs no `git commit`. True when one of its commits
    passes `--no-verify` / `-n` (also inside `-anm`), overrides `core.hooksPath`,
    or the command can't be parsed, so the caller runs the check itself.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()\n")
        lexer.whitespace = " \t\r"  # a newline separates commands, like `;`
        lexer.whitespace_split = True
        lexer.commenters = ""  # `#` mid-word (`-m fix#3`) is not a comment
        tokens = list(lexer)
    except ValueError:
        return True if GIT_COMMIT.search(command) else None
    commits = [_commit_skips_hooks(words) for words in _git_commands(tokens)]
    commits = [skips for skips in commits if skips is not None]
    return any(commits) if commits else None


def _git_commands(tokens: list[str]) -> list[list[str]]:
    """The words after `git` in each shell command that runs git."""
    commands, words = [], []
    for token in [*tokens, ";"]:
        if token not in SEPARATORS:
            words.append(token)
            continue
        git_at = next((i for i, word in enumerate(words) if os.path.basename(word) == "git"), None)
        if git_at is not None:
            commands.append(words[git_at + 1 :])
        words = []
    return commands


def _commit_skips_hooks(args: list[str]) -> bool | None:
    """For the words after `git`: None if it isn't `git commit`, else whether hooks are skipped."""
    hooks_path_overridden = False
    i = 0
    while i < len(args) and args[i].startswith("-"):
        option = args[i]
        if option in GIT_OPTIONS_WITH_VALUE and i + 1 < len(args):
            value = args[i + 1]
            hooks_path_overridden |= option == "-c" and value.lower().startswith("core.hookspath=")
            i += 1
        i += 1
    if i >= len(args) or args[i] != "commit":
        return None
    return hooks_path_overridden or _no_verify(args[i + 1 :])


def _no_verify(args: list[str]) -> bool:
    """Whether `git commit` options include --no-verify / -n (the last of --verify/--no-verify wins)."""
    no_verify = False
    words = iter(args)
    for word in words:
        if word == "--":
            break
        if word.startswith("--"):
            if word == "--verify":
                no_verify = False
            elif word.startswith("--no-veri") and "--no-verify".startswith(word):  # git accepts abbreviations
                no_verify = True
            elif word in COMMIT_LONG_WITH_VALUE:
                next(words, None)
        elif word.startswith("-") and len(word) > 1:
            for at, flag in enumerate(word[1:], start=1):
                if flag == "n":
                    no_verify = True
                if flag in COMMIT_SHORT_WITH_VALUE:
                    if at == len(word) - 1:  # value is the next word
                        next(words, None)
                    break
                if flag in COMMIT_SHORT_ATTACHED_VALUE:
                    break
    return no_verify


def _repo_root(cwd: str) -> Path | None:
    out = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    return Path(out.stdout.strip()) if out.returncode == 0 else None


def _git_hook_enabled(root: Path) -> bool:
    """True when git is set up to run .githooks/pre-commit for commits in this clone."""
    out = subprocess.run(["git", "-C", str(root), "config", "core.hooksPath"], capture_output=True, text=True, check=False)
    hooks_path = out.stdout.strip()
    return bool(hooks_path) and (root / hooks_path).resolve() == (root / ".githooks").resolve()


if __name__ == "__main__":
    sys.exit(main())
