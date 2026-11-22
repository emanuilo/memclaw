"""Persistent HTTP MCP bridge for Cursor SDK tool wiring."""

from __future__ import annotations

import asyncio
import contextlib
import socket
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import uvicorn
from loguru import logger
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Route

from .mcp_tools import MCP_SERVER_NAME, build_memclaw_mcp_server

if TYPE_CHECKING:
    from cursor_sdk import HttpMcpServerConfig

    from ..tools import ToolExecutor


async def _wait_until_listening(host: str, port: int, *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise TimeoutError(f"MCP server did not start on {host}:{port} within {timeout}s")


class _StreamableHTTPASGIApp:
    """Minimal ASGI adapter for StreamableHTTPSessionManager."""

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self._session_manager = session_manager

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        await self._session_manager.handle_request(scope, receive, send)


class HttpMcpServer:
    """Run Memclaw tools as a local HTTP MCP server for the process lifetime."""

    def __init__(self) -> None:
        self._host = "127.0.0.1"
        self._port = 0
        self._listen_sock: socket.socket | None = None
        self._session_manager: StreamableHTTPSessionManager | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._uvicorn_task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self._uvicorn_task is not None and not self._uvicorn_task.done()

    @property
    def config(self) -> HttpMcpServerConfig | None:
        if not self.is_running or self._port <= 0:
            return None
        from cursor_sdk import HttpMcpServerConfig  # optional dependency (cursor-sdk)

        return HttpMcpServerConfig(
            url=f"http://{self._host}:{self._port}/mcp",
            type="http",
        )

    async def start(
        self,
        executor: ToolExecutor,
        *,
        host: str = "127.0.0.1",
        port: int,
    ) -> None:
        if self.is_running:
            return
        self._host = host
        self._port = port

        mcp_server = build_memclaw_mcp_server(executor)
        self._session_manager = StreamableHTTPSessionManager(
            app=mcp_server,
            stateless=True,
        )
        asgi_app = _StreamableHTTPASGIApp(self._session_manager)
        session_manager = self._session_manager

        @asynccontextmanager
        async def lifespan(_app: Starlette):
            async with session_manager.run():
                yield

        starlette_app = Starlette(
            routes=[Route("/mcp", endpoint=asgi_app)],
            lifespan=lifespan,
        )

        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_sock.bind((host, port))
            listen_sock.listen(2048)
            self._port = int(listen_sock.getsockname()[1])
            self._listen_sock = listen_sock
            listen_sock = None
        except OSError as exc:
            await self.stop()
            raise OSError(
                f"Cannot bind MCP HTTP server to {host}:{port} — "
                f"is another process using the port? Set MEMCLAW_MCP_PORT to a free port. "
                f"({exc})"
            ) from exc
        finally:
            if listen_sock is not None:
                listen_sock.close()

        uvicorn_config = uvicorn.Config(
            starlette_app,
            host=host,
            port=self._port,
            fd=self._listen_sock.fileno(),
            log_level="warning",
        )
        self._uvicorn_server = uvicorn.Server(uvicorn_config)
        self._uvicorn_task = asyncio.create_task(self._uvicorn_server.serve())
        try:
            await _wait_until_listening(host, self._port)
        except BaseException:
            await self.stop()
            raise

        logger.info(
            "MCP server listening on http://{host}:{port}/mcp",
            host=host,
            port=self._port,
        )

    async def stop(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._uvicorn_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._uvicorn_task
        self._uvicorn_task = None
        self._uvicorn_server = None
        self._session_manager = None
        if self._listen_sock is not None:
            self._listen_sock.close()
            self._listen_sock = None
        self._port = 0


def mcp_servers_for(config: HttpMcpServerConfig) -> dict[str, HttpMcpServerConfig]:
    return {MCP_SERVER_NAME: config}
