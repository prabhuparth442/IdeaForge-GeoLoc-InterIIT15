"""Agreement block: each query's best place, how many queries agree on a place, and the answer-or-skip rule built on it."""
import numpy as np
from common import run_tests


def _result(xs, inliers, shifts=None):
    peaks = np.zeros((len(xs), 5))
    peaks[:, 1] = xs
    peaks[:, 2] = 50.0
    shifts = np.zeros((len(xs), 2)) if shifts is None else np.asarray(shifts, dtype=float)
    return dict(peaks=peaks, z=np.zeros(len(xs)), stage2=dict(inliers=np.array(inliers, dtype=float), shifts=shifts))


def test_best_place_of_each_query_includes_the_fine_shift():
    from semantic_rerank.agreement import query_best_places
    results = [_result([0, 100], [2, 9], shifts=[[0, 0], [3, -4]]), _result([500], [1])]
    places = query_best_places(results, [0, 1])
    assert places[0] == (103.0, 46.0, 9.0) and places[1] == (500.0, 50.0, 1.0)


def test_agreement_counts_queries_near_the_place_with_enough_inliers():
    from semantic_rerank.agreement import query_agreement
    results = [_result([100], [6]),            # at the place
               _result([105, 900], [4, 3]),    # its best (4 inliers) is 5 m away: agrees
               _result([108], [2]),            # near but only 2 inliers: does not count
               _result([100, 400], [3, 20])]   # its best is far away (400): does not agree, even with a weak candidate at the place
    assert query_agreement(results, [0, 1, 2, 3], 100.0, 50.0, radius_m=8.0) == 2
    assert query_agreement(results, [0, 1, 2, 3], 100.0, 50.0, radius_m=8.0, min_query_inliers=2) == 3
    assert query_agreement(results, [0, 1, 2, 3], 100.0, 50.0, radius_m=4.0) == 1
    assert query_agreement(results, [1, 2], 100.0, 50.0, radius_m=8.0) == 1


def test_decision_answers_at_the_bar_or_with_two_agreeing_queries():
    from semantic_rerank.agreement import decide_with_agreement
    assert decide_with_agreement(8, 1, min_inliers=8)
    assert decide_with_agreement(3, 2, min_inliers=8)
    assert not decide_with_agreement(7, 1, min_inliers=8)
    assert not decide_with_agreement(0, 0, min_inliers=8)
    try:
        decide_with_agreement(5, 1, min_inliers=8, min_agree=1)
    except AssertionError:
        return
    raise AssertionError("min_agree 1 is always met and must be refused")


run_tests(globals())
