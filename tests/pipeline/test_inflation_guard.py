"""Unit tests of the inflation guard blocks (inflation_guard_pipeline.py at the repo root). No GPU, no dataset."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import inflation_guard_pipeline as guard


def make_query(peaks, inliers):
    shifts = np.zeros((len(peaks), 2))
    return dict(peaks=np.array(peaks, float), stage2=dict(inliers=np.array(inliers, float), shifts=shifts))


def test_far_level_counts_only_places_far_from_the_choice():
    # candidate 0 is the choice (at 0, 0); candidate 1 is 100 m away (far at sigma 10); candidate 2 is 5 m away (near)
    query = make_query([[1, 0, 0], [1, 100, 0], [1, 5, 0]], [20, 4, 9])
    level = guard.far_level([query], [0], 0.0, 0.0, 10.0)
    assert level == 4.0, level


def test_far_level_is_zero_when_there_is_no_far_candidate():
    query = make_query([[1, 0, 0]], [20])
    assert guard.far_level([query], [0], 0.0, 0.0, 10.0) == 0.0


def test_factor_is_silent_inside_the_dead_zone():
    assert guard.guard_factor_from_level(guard.PRIOR * 1.4) == 1.0
    assert guard.guard_factor_from_level(guard.PRIOR * 1.5) == 1.0
    factor = guard.guard_factor_from_level(guard.PRIOR * 2.0)
    assert abs(factor - 2.0) < 1e-9, factor


def test_running_level_starts_at_the_prior_and_moves_slowly():
    class Stub(guard.InflationGuardPipeline):
        def __init__(self):
            self.prior = guard.PRIOR
            self.prior_weight = guard.PRIOR_WEIGHT
            self.dead_zone = guard.DEAD_ZONE
            self._level_sum = 0.0
            self._photo_count = 0
    stub = Stub()
    assert stub.update_level(guard.PRIOR) == 1.0
    for _ in range(5):
        factor = stub.update_level(guard.PRIOR * 3)
    # five photos at three times the prior with weight 30 are still inside the dead zone
    assert factor == 1.0, factor
    for _ in range(200):
        factor = stub.update_level(guard.PRIOR * 3)
    assert factor > 2.0, factor


if __name__ == "__main__":
    failed = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("PASS ", name)
            except AssertionError as error:
                failed += 1
                print("FAIL ", name, error)
    print("%d failed" % failed)
    sys.exit(1 if failed else 0)
