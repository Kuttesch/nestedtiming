import asyncio

import pytest

from nestedtiming import timed, timer, timing


@timing
async def async_simple(x: int) -> int:
    await asyncio.sleep(0.01)
    return x + 1


@pytest.mark.asyncio
async def test_async_no_timing() -> None:
    result = await async_simple(5)
    assert result == 6


@pytest.mark.asyncio
async def test_async_timing() -> None:
    result, total, tree = await async_simple(5, timed)

    assert result == 6
    assert total > 0
    assert tree is None


@timing
async def async_nested() -> None:
    with timer("phase"):
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_async_nested_region() -> None:
    _, total, tree = await async_nested(timed)

    assert total > 0
    assert tree is not None
    assert "phase" in tree