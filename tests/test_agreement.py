"""Tests of the optional agreement and inflation guard (geoloc/agreement.py) and of the decision with a factor. No GPU, no data. Run: python tests/test_agreement.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

import settings
from geoloc import agreement, decision


def make_result(places, inliers):
    peaks = np.array([[1.0, x, y, 0.0, 0.0] for x, y in places])
    shifts = np.zeros((len(places), 2))
    return dict(peaks=peaks, z=np.zeros(len(places)), stage2=dict(inliers=np.array(inliers, float), shifts=shifts))


def test_best_of_query_skips_zero_inliers():
    result = make_result([(0, 0), (50, 0)], [0, 0])
    assert agreement.best_of_query(result) is None
    result = make_result([(0, 0), (50, 0)], [2, 5])
    assert agreement.best_of_query(result) == (50.0, 0.0, 5.0)


def test_two_queries_on_one_place_agree():
    results = [make_result([(100, 100), (900, 900)], [4, 1]), make_result([(103, 100), (500, 500)], [5, 0]), make_result([(900, 900)], [3])]
    assert agreement.agreeing_queries(results, 100.0, 100.0, 10.0, 3) == 2
    assert agreement.agreeing_queries(results, 100.0, 100.0, 10.0, 5) == 1


def test_far_level_ignores_the_choice():
    results = [make_result([(0, 0), (500, 0), (0, 600)], [20, 4, 2])]
    assert agreement.far_level(results, 0.0, 0.0, 10.0) == 3.0
    assert agreement.far_level([make_result([(0, 0)], [20])], 0.0, 0.0, 10.0) == 0.0


def test_guard_is_silent_on_clean_levels_and_rises_when_inflated():
    guard = agreement.InflationGuard()
    for photo in range(20):
        assert guard.factor_after(settings.GUARD_PRIOR * 1.2) == 1.0
    guard = agreement.InflationGuard()
    factor = 1.0
    for photo in range(200):
        factor = guard.factor_after(settings.GUARD_PRIOR * 2.0)
    assert 1.5 < factor <= 2.0, factor


def test_bar_scales_with_the_factor():
    assert decision.should_answer(8, 1.0)
    assert not decision.should_answer(8, 1.5)
    assert decision.should_answer(12, 1.5)


if __name__ == "__main__":
    failed = 0
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            try:
                function()
                print("PASS ", name)
            except AssertionError as problem:
                failed += 1
                print("FAIL ", name, problem)
    print("%d failed" % failed)
    sys.exit(1 if failed else 0)
