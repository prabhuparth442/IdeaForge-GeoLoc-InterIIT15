import numpy as np
from common import run_tests, world


def _peaks():
    p = np.zeros((5, 6)); p[:, 0] = [.9, .8, .7, .6, .5]; p[:, 1] = [100, 200, 300, 400, 500]; p[:, 2] = [10, 20, 30, 40, 50]
    return p


def test_most_inliers_wins_and_shift_applied():
    from result_thresholding.decision import final_answer
    s = dict(inliers=np.array([3, 15, 15, 40, 0.]), shifts=np.array([[0, 0], [1, 2], [3, 4], [-2, 1.5], [0, 0]]))
    x, y, n, k = final_answer(_peaks(), s)
    assert (x, y, n, k) == (398.0, 41.5, 40.0, 3)


def test_tie_goes_to_higher_correlation():
    from result_thresholding.decision import final_answer
    s = dict(inliers=np.array([0, 15, 15, 0, 0.]), shifts=np.zeros((5, 2)))
    assert final_answer(_peaks(), s)[3] == 1


def test_fallback_when_unverified():
    from result_thresholding.decision import final_answer
    s = dict(inliers=np.array([3, 5, 5, 8, 0.]), shifts=np.ones((5, 2)))
    x, y, n, k = final_answer(_peaks(), s)
    assert (x, y, k) == (100.0, 10.0, 0) and n == 8


def test_decide_threshold():
    from result_thresholding.decision import decide
    assert decide(12) and not decide(11)


def test_negative_inliers_rejected():
    from result_thresholding.decision import final_answer
    try:
        final_answer(_peaks(), dict(inliers=np.array([-1, 0, 0, 0, 0.]), shifts=np.zeros((5, 2))))
    except AssertionError:
        return
    raise AssertionError("negative inliers accepted")


def test_localise_photo_correlation_only():
    from result_thresholding.pipeline import localise_photo
    import coarse_search.search as search
    search.DISC_RADIUS = 600.0
    w = world(); t = w["testset"]
    for i in range(3):
        r = localise_photo(w["items"][i], t.truths[i] + np.array([250.0, -180.0]), w["camera"], w["store"], matcher=None)
        assert r["answer"] is False and t.error_m(i, r["x"], r["y"]) < 12


def test_localise_photo_is_order_independent():
    from result_thresholding.pipeline import localise_photo
    import coarse_search.search as search
    search.DISC_RADIUS = 600.0
    w = world(); t = w["testset"]
    run = lambda i: localise_photo(w["items"][i], t.truths[i] + np.array([250.0, -180.0]), w["camera"], w["store"])
    forward = [run(i) for i in range(3)]
    backward = [run(i) for i in (2, 1, 0)][::-1]
    for a, b in zip(forward, backward):
        assert (a["x"], a["y"]) == (b["x"], b["y"])


run_tests(globals())
