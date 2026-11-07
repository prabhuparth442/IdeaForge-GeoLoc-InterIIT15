"""Heading-query block: the list of queries, the geometry override, and pooling the candidates of several queries."""
import numpy as np
from common import run_tests


def test_query_list_has_the_plain_query_once():
    from coarse_search.heading_queries import DEFAULT_HEADINGS, query_list
    queries = query_list(DEFAULT_HEADINGS)
    assert len(queries) == 9 and [q["id"] for q in queries] == list(range(9))
    assert [q for q in queries if q["heading"] == 0.0][0]["scale_factor"] == 1.0
    assert len(query_list([-5, 0, 5], [1.0, 1.05])) == 6
    for bad in ([-5, 5], [0, 0]):
        try:
            query_list(bad)
        except AssertionError:
            continue
        raise AssertionError("a list without exactly one plain query must be refused: %s" % bad)


def test_query_geometry_moves_the_yaw_and_scales_without_touching_the_settings():
    from coarse_search.heading_queries import query_geometry
    from header import settings
    before = dict(settings.GEOMETRY)
    geometry = query_geometry(5.0, 1.05)
    assert abs(geometry["yaw_offset"] - (before["yaw_offset"] + 5.0)) < 1e-9
    assert abs(geometry["scale"] - before["scale"] * 1.05) < 1e-9
    assert dict(settings.GEOMETRY) == before


def _result(xs, inliers, z):
    peaks = np.zeros((len(xs), 5))
    peaks[:, 1] = xs
    peaks[:, 2] = 7.0
    return dict(peaks=peaks, z=np.array(z, dtype=float), stage2=dict(inliers=np.array(inliers, dtype=float), shifts=np.zeros((len(xs), 2))))


def test_pooling_keeps_every_candidate_and_remembers_the_query():
    from coarse_search.heading_queries import pool_queries
    from semantic_rerank.decision import choose_candidate
    results = [_result([1, 2], [3, 4], [0.1, 0.2]), _result([10, 20, 30], [2, 40, 5], [0, 1, 2]), _result([100], [9], [0.5])]
    peaks, stage2, z, owner = pool_queries(results, [0, 1, 2])
    assert len(peaks) == 6 and list(owner) == [0, 0, 1, 1, 1, 2]
    x, y, inliers, chosen, z_chosen = choose_candidate(peaks, stage2, z, 0.0)
    assert x == 20 and inliers == 40 and owner[chosen] == 1                              # the best candidate wins whatever query it came from
    peaks, stage2, z, owner = pool_queries(results, [0, 2])                              # a subset of the queries: the other query's candidate is not there
    x, y, inliers, chosen, z_chosen = choose_candidate(peaks, stage2, z, 0.0)
    assert x == 100 and owner[chosen] == 2
    try:
        pool_queries(results, [])
    except AssertionError:
        return
    raise AssertionError("an empty list of queries must be refused")


def test_heading_penalty_prefers_the_smaller_heading_and_zero_penalty_changes_nothing():
    from coarse_search.heading_queries import choose_with_heading_penalty, pool_queries
    from semantic_rerank.decision import choose_candidate
    results = [_result([1], [30], [0.0]), _result([2], [34], [0.0])]
    headings = [0.0, 10.0]
    peaks, stage2, z, owner = pool_queries(results, [0, 1])
    same = choose_with_heading_penalty(peaks, stage2, z, owner, headings, 1.0, 0.0)
    assert same == choose_candidate(peaks, stage2, z, 1.0)                                # no penalty: identical to the plain choice
    assert same[0] == 2                                                                  # the 34-inlier candidate of the far heading wins
    penalised = choose_with_heading_penalty(peaks, stage2, z, owner, headings, 1.0, 0.5)  # 10 degrees cost 5 inliers: 34 - 5 = 29 < 30
    assert penalised[0] == 1 and penalised[2] == 30.0                                    # and the real inlier count comes back, not the penalised one


run_tests(globals())
