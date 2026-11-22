"""Tests for the Claude Code commit hook's `git commit` detection (.claude/hooks/)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "shape_check_on_commit.py"


@pytest.fixture(scope="module")
def hook():
    spec = importlib.util.spec_from_file_location("shape_check_on_commit", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "command, expected",
    [
        # Not a commit: the hook does nothing.
        ("git status", None),
        ("git log -n 5", None),
        ("git add -A && git log -n 1", None),
        ("ls -n", None),
        # A commit that git's own hook will check.
        ("git commit -m x", False),
        ("cd repo && git add . && git commit -am 'fix stuff'", False),
        ('git commit -m "fix -n flag"', False),
        ("git commit -m '--no-verify is bad'", False),
        ("git commit -mn", False),  # the message is "n"
        ("git commit -m x -- -n", False),  # a path named -n
        ("git commit -n -m x --verify", False),  # the last one wins
        ("git log -n 3; git commit -m x", False),
        # A commit that skips git's hooks.
        ("git commit --no-verify -m x", True),
        ("git commit --no-veri -m x", True),
        ("git commit -n -m x", True),
        ("git commit -m x -n", True),
        ('git commit -nm "msg"', True),
        ("git commit -anm msg", True),
        ("git commit -a -n", True),
        ("git -c k=v commit -n", True),
        ("git -C . -c user.name=x commit -n -m x", True),
        ("git -c core.hooksPath=/dev/null commit -m x", True),
        ("git add x\ngit commit -n -m x", True),
        ("git commit -m x; git commit --amend -n --no-edit", True),
        ("git commit -m 'unclosed", True),  # can't parse: the caller checks itself
    ],
)
def test_commit_skips_git_hooks(hook, command, expected):
    assert hook.commit_skips_git_hooks(command) is expected
