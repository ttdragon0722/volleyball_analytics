import asyncio
from unittest.mock import MagicMock

from videomae_labeler.quiet_loop import quiet_exception_handler, serve_quietly


def test_client_disconnect_is_swallowed():
    loop = MagicMock()
    quiet_exception_handler(loop, {"exception": ConnectionResetError(10054, "reset")})
    quiet_exception_handler(loop, {"exception": ConnectionAbortedError()})
    loop.default_exception_handler.assert_not_called()


def test_other_errors_are_forwarded():
    loop = MagicMock()
    ctx = {"exception": ValueError("boom"), "message": "x"}
    quiet_exception_handler(loop, ctx)
    loop.default_exception_handler.assert_called_once_with(ctx)


def test_context_without_exception_is_forwarded():
    loop = MagicMock()
    ctx = {"message": "Task was destroyed but it is pending!"}
    quiet_exception_handler(loop, ctx)
    loop.default_exception_handler.assert_called_once_with(ctx)


def test_serve_quietly_installs_handler_and_serves():
    seen = {}

    class FakeServer:
        async def serve(self):
            seen["handler"] = asyncio.get_running_loop().get_exception_handler()

    asyncio.run(serve_quietly(FakeServer()))
    assert seen["handler"] is quiet_exception_handler
