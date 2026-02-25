import time

import pytest

from nestedtiming import TimingNode, timed, timer, timing


@timing
def simple(x: int) -> int:
    return x * 2


def test_no_timing_returns_plain_result() -> None:
    assert simple(5) == 10


def test_sync_timing_basic() -> None:
    result, total, tree = simple(5, timed)

    assert result == 10
    assert total >= 0
    assert tree is None


@timing
def nested() -> None:
    with timer("outer"):
        time.sleep(0.01)
        with timer("inner"):
            time.sleep(0.01)


def test_nested_regions() -> None:
    _, total, tree = nested(timed)

    assert total > 0
    assert tree is not None
    assert "outer" in tree

    outer: TimingNode = tree["outer"]
    assert outer.total > 0
    assert "inner" in outer.children

    inner = outer.children["inner"]
    assert inner.total > 0


def test_sentinel_not_last_argument() -> None:
    with pytest.raises(TypeError):
        # timed must be last positional argument
        simple(timed, 5)  # type: ignore