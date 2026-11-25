from __future__ import annotations

import asyncio
import selectors
import sys
from collections.abc import Coroutine
from typing import Any, TypeVar


T = TypeVar("T")


class coroutine(Coroutine[Any, Any, T]):
    """Concrete coroutine wrapper that delegates all await/send/throw/close calls.

    This makes it easy to provide a usable implementation for the abstract
    methods required by collections.abc.Coroutine without losing behavior of
    the wrapped native coroutine object.
    """

    def __init__(self, wrapped: Coroutine[Any, Any, T]) -> None:
        self._wrapped = wrapped

    def send(self, value: Any) -> Any:
        return self._wrapped.send(value)

    def throw(self, typ: Any, val: Any = None, tb: Any = None) -> Any:
        if val is None and tb is None:
            return self._wrapped.throw(typ)
        return self._wrapped.throw(typ, val, tb)

    def close(self) -> None:
        self._wrapped.close()

    def __await__(self) -> Any:
        return self._wrapped.__await__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._wrapped!r})"


def _windows_selector_loop() -> asyncio.AbstractEventLoop:
    """
    Create a SelectorEventLoop for Windows.

    psycopg async is not compatible with ProactorEventLoop in this project,
    so every top-level async entry point should run through run_async().
    """
    return asyncio.SelectorEventLoop(
        selectors.SelectSelector()
    )


def run_async(
    coroutine: Coroutine[Any, Any, T],
) -> T:
    """
    Central top-level async runner.

    Windows:
        force SelectorEventLoop for psycopg compatibility.

    Linux/macOS:
        use the normal asyncio.run() behavior.

    Important:
        Use this only at top-level entry points such as:
            cli.py
            sync_tools.py
            embed_tools.py

        Do not call run_async() from inside an already-running event loop.
    """

    if sys.platform == "win32":
        # asyncio.Runner(loop_factory=...) works on Python 3.11+
        # and avoids depending on asyncio.run(loop_factory=...), whose
        # availability differs across Python versions.
        with asyncio.Runner(
            loop_factory=_windows_selector_loop
        ) as runner:
            return runner.run(
                coroutine
            )

    return asyncio.run(
        coroutine
    )
