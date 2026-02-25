from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from types import TracebackType
from typing import Any, Literal, ParamSpec, TypeVar, overload, cast

P = ParamSpec("P")
R = TypeVar("R")


# ---------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------


class _TimedFlag:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover
        return "timed"


timed: _TimedFlag = _TimedFlag()


# ---------------------------------------------------------------------
# Public timing node
# ---------------------------------------------------------------------


@dataclass(slots=True)
class TimingNode:
    total: float = 0.0
    children: dict[str, "TimingNode"] = field(default_factory=dict)


@dataclass(slots=True)
class _TimingState:
    roots: dict[str, TimingNode] = field(default_factory=dict)
    stack: list[TimingNode] = field(default_factory=list)


_current_state: ContextVar[_TimingState | None] = ContextVar(
    "_nestedtiming_state",
    default=None,
)


# ---------------------------------------------------------------------
# Region context manager
# ---------------------------------------------------------------------


class Timer(
    AbstractContextManager[None],
    AbstractAsyncContextManager[None],
):
    __slots__ = ("_name", "_start", "_node")

    def __init__(self, name: str | None) -> None:
        self._name = name if name else None
        self._start: float | None = None
        self._node: TimingNode | None = None

    def __enter__(self) -> None:
        state = _current_state.get()
        if state is None or self._name is None:
            return None

        parent_dict = state.stack[-1].children if state.stack else state.roots

        node = parent_dict.get(self._name)
        if node is None:
            node = TimingNode()
            parent_dict[self._name] = node

        state.stack.append(node)
        self._node = node
        self._start = time.perf_counter()
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        state = _current_state.get()

        if (
            state is not None
            and self._node is not None
            and self._start is not None
        ):
            elapsed = time.perf_counter() - self._start
            self._node.total += elapsed

            if state.stack and state.stack[-1] is self._node:
                state.stack.pop()

        return False

    async def __aenter__(self) -> None:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        return self.__exit__(exc_type, exc, tb)


def timer(name: str | None) -> Timer:
    return Timer(name)


# ---------------------------------------------------------------------
# Overloads
# ---------------------------------------------------------------------


@overload
def timing(
    func: Callable[P, R],
) -> Callable[P, R | tuple[R, float, dict[str, TimingNode] | None]]: ...


@overload
def timing(
    func: Callable[P, Awaitable[R]],
) -> Callable[
    P,
    Awaitable[R | tuple[R, float, dict[str, TimingNode] | None]],
]: ...


# ---------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------


def timing(
    func: Callable[P, Any],
) -> Callable[P, Any]:
    if inspect.iscoroutinefunction(func):
        return _timing_async(cast(Callable[P, Awaitable[R]], func))
    return _timing_sync(cast(Callable[P, R], func))


def _timing_sync(
    func: Callable[P, R],
) -> Callable[P, R | tuple[R, float, dict[str, TimingNode] | None]]:
    @wraps(func)
    def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R | tuple[R, float, dict[str, TimingNode] | None]:
        do_timing = bool(args and args[-1] is timed)

        if not do_timing:
            return func(*args, **kwargs)

        state = _TimingState()
        token = _current_state.set(state)

        start = time.perf_counter()
        try:
            inner = cast(Callable[..., R], func)
            result = inner(*args[:-1], **kwargs)
        finally:
            total = time.perf_counter() - start
            _current_state.reset(token)

        tree = state.roots if state.roots else None
        return result, total, tree

    return wrapper


def _timing_async(
    func: Callable[P, Awaitable[R]],
) -> Callable[
    P,
    Awaitable[R | tuple[R, float, dict[str, TimingNode] | None]],
]:
    @wraps(func)
    async def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R | tuple[R, float, dict[str, TimingNode] | None]:
        do_timing = bool(args and args[-1] is timed)

        if not do_timing:
            return await func(*args, **kwargs)

        state = _TimingState()
        token = _current_state.set(state)

        start = time.perf_counter()
        try:
            inner = cast(Callable[..., Awaitable[R]], func)
            result = await inner(*args[:-1], **kwargs)
        finally:
            total = time.perf_counter() - start
            _current_state.reset(token)

        tree = state.roots if state.roots else None
        return result, total, tree

    return wrapper