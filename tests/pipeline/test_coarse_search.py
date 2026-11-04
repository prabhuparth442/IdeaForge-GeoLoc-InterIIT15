import numpy as np
import torch
from common import run_tests, world


def test_fast_equals_slow():
    from coarse_search.self_tests import fast_equals_slow
    fast_equals_slow()


def test_hidden_piece_is_found():
    from coarse_search.correlation import correlate_everywhere
    w = torch.rand(8, 60, 60); k = w[:, 20:30, 25:35].clone()
    z = correlate_everywhere(w, k, torch.ones(10, 10))["zncc"]
    assert divmod(int(z.argmax()), z.shape[1]) == (20, 25) and float(z.max()) > 0.999


def test_zncc_floor_is_near_zero_cosine_is_not():
    from coarse_search.correlation import correlate_everywhere
    w = torch.rand(8, 80, 80); k = w[:, 5:25, 5:25].clone()
    out = correlate_everywhere(w, k, torch.ones(20, 20))
    assert abs(float(out["zncc"].median())) < 0.1 and float(out["cosine"].median()) > 0.6


def test_fast_length():
    from coarse_search.correlation import fast_length
    assert [fast_length(n) for n in (2999, 1021, 3001)] == [3000, 1024, 3072]


def test_nms_separates_peaks():
    from coarse_search.search import pick_peaks
    y, x = np.mgrid[0:300, 0:300]
    bump = lambda r, c, h: h * np.exp(-((y - r) ** 2 + (x - c) ** 2) / 18.0)
    score = torch.tensor((bump(60, 70, .9) + bump(62, 76, .5) + bump(200, 220, .8)).astype(np.float32))
    pos = (torch.arange(300) * 4 + 112) * 3.39
    peaks = pick_peaks(score, dict(level=7, side=56, x_m=pos, y_m=pos.clone(), inside=torch.ones(300, 300, dtype=torch.bool)))
    assert peaks.shape == (25, 5) and peaks[0, 0] > 0.9 and abs(peaks[1, 0] - 0.8) < 0.02   # the 0.5 bump was suppressed


def test_sentinel_rows_when_prior_is_off_the_map():
    from coarse_search.search import search_frame
    from image_processing.geometry import render_query
    w = world()
    canvas, valid = render_query(w["items"][0], w["camera"], 3)
    found, _ = search_frame(canvas, valid, 3, np.array([-500000.0, -500000.0]), w["store"], variants=[("zncc", 0.0)])
    assert (found[("zncc", 0.0)][:, 1] > 1e8).all()


def test_true_place_is_in_the_top_25():
    from coarse_search.search import search_frame
    from image_processing.geometry import render_query
    import coarse_search.search as search
    search.DISC_RADIUS = 600.0
    w = world(); t = w["testset"]
    for i in range(3):
        level = int(t.query_levels[i])
        canvas, valid = render_query(w["items"][i], w["camera"], level)
        found, _ = search_frame(canvas, valid, level, t.truths[i] + np.array([250.0, -180.0]), w["store"], variants=[("zncc", 0.0)])
        p = found[("zncc", 0.0)]
        best = min(t.error_m(i, r[1], r[2]) for r in p if r[1] < 1e8)
        assert best < 12, "photo %d: nearest candidate %.1f m" % (i, best)


run_tests(globals())
