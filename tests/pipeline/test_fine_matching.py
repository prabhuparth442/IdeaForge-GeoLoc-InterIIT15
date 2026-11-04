import numpy as np
from common import run_tests, world


def test_ransac_recovers_shift_with_outliers():
    from fine_matching.ransac import fit_transform
    rng = np.random.default_rng(0)
    a = (rng.random((40, 2)) * 200).astype(np.float32)
    b = a + np.array([5, -3], dtype=np.float32); b[:10] = rng.random((10, 2)) * 200
    m, inl = fit_transform(a, b)
    assert np.allclose(m, [[1, 0, 5], [0, 1, -3]], atol=0.05) and inl.sum() >= 30


def test_ransac_gate_rejects_big_rotation():
    from fine_matching.ransac import fit_transform
    rng = np.random.default_rng(1)
    a = (rng.random((40, 2)) * 200 - 100).astype(np.float32)
    t = np.radians(25); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
    m, inl = fit_transform(a, a @ R.T)
    assert m is None and inl.sum() == 0


def test_ransac_needs_four_pairs():
    from fine_matching.ransac import fit_transform
    m, inl = fit_transform(np.zeros((3, 2), np.float32), np.zeros((3, 2), np.float32))
    assert m is None and len(inl) == 3


def _use_stand_in():
    import fine_matching.second_stage as stage
    import stand_in_matcher
    stage.find_keypoints = stand_in_matcher.find_keypoints
    stage.match_keypoints = stand_in_matcher.match_keypoints


def test_second_stage_runs_with_stand_in_matcher():
    from coarse_search.search import search_frame
    from fine_matching.second_stage import second_stage
    from image_processing.geometry import render_query
    import coarse_search.search as search
    search.DISC_RADIUS = 600.0
    _use_stand_in()
    w = world(); t = w["testset"]
    i = 0; level = int(t.query_levels[i])
    canvas, valid = render_query(w["items"][i], w["camera"], level)
    found, _ = search_frame(canvas, valid, level, t.truths[i] + np.array([250.0, -180.0]), w["store"], variants=[("zncc", 0.0)])
    peaks = found[("zncc", 0.0)]
    out = second_stage(object(), canvas, valid, level, peaks, w["store"]["levels"])
    assert out["inliers"].shape == (25,) and out["shifts"].shape == (25, 2)
    nearest = int(np.argmin([t.error_m(i, p[1], p[2]) if p[1] < 1e8 else 1e9 for p in peaks]))
    print("   inliers at the true candidate (index %d): %d, best other: %d" % (nearest, out["inliers"][nearest], np.delete(out["inliers"], nearest).max()))
    assert out["inliers"][nearest] >= out["inliers"].max() - 1e-9


run_tests(globals())
