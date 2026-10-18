"""Memclaw - A lightweight, local-first personal memory assistant."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("memclaw")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0"
