import numpy as np

from cube_search.tracker import CubeTracker


def test_fusion_rejects_outliers_and_gates_height():
    tr = CubeTracker(min_observations=3)
    rng = np.random.default_rng(1)
    for _ in range(20):
        assert tr.add(2, 1.0 + rng.normal(0, 0.02), 2.0 + rng.normal(0, 0.02), 0.05, 2.0)
    tr.add(2, 5.0, 5.0, 0.05, 2.0)           # falszywa detekcja z tym samym ID
    tr.add(2, 5.1, 5.0, 0.05, 2.0)
    assert not tr.add(2, 1.0, 2.0, 0.6, 2.0)  # nie lezy na podlodze
    assert not tr.add(2, 1.0, 2.0, 0.05, 9.0)  # za daleko
    e = tr.estimate(2)
    assert e.n == 20
    assert abs(e.x - 1.0) < 0.02 and abs(e.y - 2.0) < 0.02


def test_confirmation_threshold():
    tr = CubeTracker(min_observations=3)
    tr.add(1, 0.0, 0.0, 0.045, 1.0)
    tr.add(1, 0.01, 0.0, 0.045, 1.0)
    assert tr.confirmed() == []
    tr.add(1, 0.0, 0.01, 0.045, 1.0)
    assert [e.id for e in tr.confirmed()] == [1]
