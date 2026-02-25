"""Provide call-time opt-in hierarchical execution timing.

This module exposes:

- ``timing``: A decorator enabling optional execution timing.
- ``timer``: A context manager for defining named nested timing regions.
- ``timed``: A sentinel used to activate timing at call time.
- ``TimingNode``: A public representation of timing tree nodes.

Timing is disabled by default and incurs negligible overhead unless the
``timed`` sentinel is passed as the final positional argument.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from types import TracebackType
from typing import Any, Literal, ParamSpec, TypeVar, cast, overload

P = ParamSpec("P")
R = TypeVar("R")


class _TimedFlag:
    """Represent the sentinel that activates timing."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a readable sentinel representation.

        Returns:
            str: The string ``"timed"``.
        """
        return "timed"


timed: _TimedFlag = _TimedFlag()


@dataclass(slots=True)
class TimingNode:
    """Represent a node in the hierarchical timing tree.

    Attributes:
        total (float):
            Total accumulated execution time in seconds.
        children (dict[str, TimingNode]):
            Nested timing regions keyed by region name.
    """

    total: float = 0.0
    children: dict[str, TimingNode] = field(default_factory=dict)


@dataclass(slots=True)
class _TimingState:
    """Store internal timing state for a single decorated call.

    Attributes:
        roots (dict[str, TimingNode]):
            Top-level timing regions.
        stack (list[TimingNode]):
            Active region stack used to maintain nesting.
    """

    roots: dict[str, TimingNode] = field(default_factory=dict)
    stack: list[TimingNode] = field(default_factory=list)


_current_state: ContextVar[_TimingState | None] = ContextVar(
    "_nestedtiming_state",
    default=None,
)


class Timer(
    AbstractContextManager[None],
    AbstractAsyncContextManager[None],
):
    """Measure a named timing region within a timed function call.

    Use this via the ``timer()`` helper function.
    """

    __slots__ = ("_name", "_start", "_node")

    def __init__(self, name: str | None) -> None:
        """Initialize a timing region.

        Args:
            name (str | None):
                Name of the timing region. If ``None`` or empty,
                region timing is disabled.
        """
        self._name = name if name else None
        self._start: float | None = None
        self._node: TimingNode | None = None

    def __enter__(self) -> None:
        """Start measuring the timing region."""
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
        """Stop measuring the timing region and accumulate elapsed time.

        Args:
            exc_type (type[BaseException] | None):
                The exception type if raised.
            exc (BaseException | None):
                The exception instance if raised.
            tb (TracebackType | None):
                The traceback if raised.

        Returns:
            Literal[False]:
                Always returns False to propagate exceptions.
        """
        state = _current_state.get()

        if state is not None and self._node is not None and self._start is not None:
            elapsed = time.perf_counter() - self._start
            self._node.total += elapsed

            if state.stack and state.stack[-1] is self._node:
                state.stack.pop()

        return False

    async def __aenter__(self) -> None:
        """Start measuring the timing region in async context."""
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        """Stop measuring the timing region in async context."""
        return self.__exit__(exc_type, exc, tb)


def timer(name: str | None) -> Timer:
    """Create a timing region context manager.

    Args:
        name (str | None):
            Name of the timing region.

    Returns:
        Timer:
            A context manager that measures the named region.
    """
    return Timer(name)


@overload
def timing[**P, R](
    func: Callable[P, R],
) -> Callable[P, R | tuple[R, float, dict[str, TimingNode] | None]]: ...


@overload
def timing[**P, R](
    func: Callable[P, Awaitable[R]],
) -> Callable[
    P,
    Awaitable[R | tuple[R, float, dict[str, TimingNode] | None]],
]: ...


def timing[**P](
    func: Callable[P, Any],
) -> Callable[P, Any]:
    """Decorate a function to enable optional hierarchical timing.

    Timing activates only when the sentinel ``timed`` is passed as the
    final positional argument.

    When timing is enabled, the wrapped function returns:

        (result, total_time_seconds, timing_tree)

    Otherwise, it behaves identically to the original function.

    Args:
        func (Callable[P, Any]):
            The function to wrap.

    Returns:
        Callable[P, Any]:
            A wrapped function preserving the original signature.
    """
    if inspect.iscoroutinefunction(func):
        return _timing_async(cast(Callable[P, Awaitable[R]], func))
    return _timing_sync(cast(Callable[P, R], func))


def _timing_sync[**P, R](
    func: Callable[P, R],
) -> Callable[P, R | tuple[R, float, dict[str, TimingNode] | None]]:
    """Wrap a synchronous function with optional timing support.

    Args:
        func (Callable[P, R]):
            The synchronous function being wrapped.

    Returns:
        Callable[P, R | tuple[R, float, dict[str, TimingNode] | None]]:
            A wrapped function that optionally returns timing data.
    """

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


def _timing_async[**P, R](
    func: Callable[P, Awaitable[R]],
) -> Callable[
    P,
    Awaitable[R | tuple[R, float, dict[str, TimingNode] | None]],
]:
    """Wrap an asynchronous function with optional timing support.

    Args:
        func (Callable[P, Awaitable[R]]):
            The asynchronous function being wrapped.

    Returns:
        Callable[P, Awaitable[R | tuple[R, float, dict[str, TimingNode] | None]]]:
            A wrapped coroutine that optionally returns timing data.
    """

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