"""
Silence a harmless Windows asyncio error.

On Windows the ProactorEventLoop logs ``ConnectionResetError`` (WinError 10054)
from ``_ProactorBasePipeTransport._call_connection_lost`` whenever the browser
aborts a connection - e.g. seeking a <video> cancels an in-flight range request.
The server is unaffected; only the traceback is noise. Everything else is
forwarded to the default handler unchanged.
"""
from __future__ import annotations

import asyncio
from typing import Any

_IGNORED = (ConnectionResetError, ConnectionAbortedError)


def is_client_disconnect(context: dict[str, Any]) -> bool:
    """True when the loop error is just a client dropping its connection."""
    return isinstance(context.get("exception"), _IGNORED)


def quiet_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    if is_client_disconnect(context):
        return
    loop.default_exception_handler(context)


async def serve_quietly(server: Any) -> None:
    """Run a ``uvicorn.Server`` with the quiet exception handler installed."""
    asyncio.get_running_loop().set_exception_handler(quiet_exception_handler)
    await server.serve()
