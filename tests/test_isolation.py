from nestedtiming import timed, timer, timing


@timing
def region() -> None:
    with timer("a"):
        pass


def test_state_isolation() -> None:
    _, _, tree1 = region(timed)
    _, _, tree2 = region(timed)

    assert tree1 is not None
    assert tree2 is not None
    assert tree1 is not tree2