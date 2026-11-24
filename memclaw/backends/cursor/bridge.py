"""The `cursor-sdk-bridge` process (a Node subprocess) that every chat's
agent runs on: launched lazily, relaunched when it dies, shut down with
the backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


async def drain_stderr(stream: asyncio.StreamReader) -> None:
    """Keep reading the bridge's stderr. The SDK only reads it until the
    bridge is up, and a long-lived bridge would block once the pipe fills.
    Read in chunks: readline() fails on a line over 64 KiB, which would end
    the drain."""
    while chunk := await stream.read(65536):
        logger.debug("cursor-sdk-bridge: {out}", out=chunk.decode(errors="replace").rstrip())


def bridge_failed(exc: BaseException) -> bool:
    """True when *exc* came from not reaching the bridge process at all
    (it crashed), rather than from the agent or Cursor's API. A timeout
    doesn't count: a slow run says nothing about the bridge, and dropping it
    would abort other chats' runs."""
    cause = exc.__cause__
    return isinstance(cause, httpx.RequestError) and not isinstance(cause, httpx.TimeoutException)


def _bridge_process(client: Any) -> asyncio.subprocess.Process | None:
    # The SDK keeps the subprocess it launched on a private attribute.
    return getattr(getattr(client, "_owned_bridge", None), "process", None)


class CursorBridge:
    """The one bridge client the whole Memclaw process shares."""

    def __init__(self, cwd: str) -> None:
        self._cwd = cwd
        self.client: Any = None  # launched lazily
        self._lock = asyncio.Lock()
        self._background: set[asyncio.Task] = set()  # stderr drains

    async def launch(self) -> Any:
        from cursor_sdk import AsyncClient  # optional dependency (cursor-sdk)
        from cursor_sdk._vendor import resolve_bridge_path  # optional dependency (cursor-sdk)

        # Local agents are stored under a path derived from the bridge
        # process's working directory (not LocalAgentOptions.cwd), and
        # launch_bridge can't set it, so start the bridge through `sh` in
        # the memory dir. A restarted process then finds its agents again.
        Path(self._cwd).mkdir(parents=True, exist_ok=True)
        client = await AsyncClient.launch_bridge(
            ["/bin/sh", "-c", 'cd "$0" && exec "$@"', self._cwd, resolve_bridge_path()],
            workspace=self._cwd,
        )
        process = _bridge_process(client)
        if process is not None and process.stderr is not None:
            task = asyncio.create_task(drain_stderr(process.stderr))
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        return client

    async def get(self) -> Any:
        async with self._lock:
            client = self.client
            if client is not None:
                process = _bridge_process(client)
                if process is not None and process.returncode is not None:
                    # The bridge died between calls: relaunch now rather than
                    # failing the next message (or /model) first.
                    logger.warning("Cursor SDK bridge exited ({rc}); relaunching", rc=process.returncode)
                    await self.drop(client)
            if self.client is None:
                self.client = await self.launch()
            return self.client

    async def drop(self, client: Any) -> None:
        """Shut *client*'s bridge down; the next call launches a new one."""
        if self.client is client:
            self.client = None
        try:
            await client.aclose()
        except Exception as exc:
            logger.warning("Error while closing the Cursor SDK bridge: {exc}", exc=exc)
