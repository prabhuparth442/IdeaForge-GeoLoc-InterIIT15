"""LoFTR adapter with a stand-in LoFTR model: matches -> inliers, the confidence cut, the RANSAC distance, and the stored-matches route."""
import numpy as np
import torch
from common import run_tests, world


class StandInLoftr:
    """Same call as kornia's LoFTR: a dict with image0 / image1 in, keypoints0 / keypoints1 / confidence out. The second picture is the first moved by (+4, -3);
    half of the matches are good (confidence 0.9), a quarter are off by 12 px (confidence 0.35), a quarter are random (confidence 0.25)."""
    def __call__(self, batch):
        k = batch["image0"].shape[2] / 224.0                  # the picture may have been enlarged: positions and the shift grow with it
        rng = np.random.default_rng(0)
        points = (rng.uniform(20, 200, size=(120, 2)) * k).astype(np.float32)
        moved = points + (np.array([4.0, -3.0]) * k).astype(np.float32)
        confidence = np.full(120, 0.9, dtype=np.float32)
        moved[60:90] += 12.0 * k
        confidence[60:90] = 0.35
        moved[90:] = (rng.uniform(20, 200, size=(30, 2)) * k).astype(np.float32)
        confidence[90:] = 0.25
        count = batch["image0"].shape[0]                      # every picture pair of the batch gets the same 120 matches; batch_indexes says which pair each belongs to
        return dict(keypoints0=torch.from_numpy(points).repeat(count, 1), keypoints1=torch.from_numpy(moved).repeat(count, 1),
                    confidence=torch.from_numpy(confidence).repeat(count), batch_indexes=torch.arange(count).repeat_interleave(120))


def _setup():
    from fine_matching.loftr_matcher import LoftrMatcher
    return LoftrMatcher(StandInLoftr(), 0.3)


def _peaks_and_world():
    from coarse_search.search import search_frame
    from image_processing.geometry import render_query
    import coarse_search.search as search
    search.DISC_RADIUS = 600.0
    w = world(); t = w["testset"]
    level = int(t.query_levels[0])
    canvas, valid = render_query(w["items"][0], w["camera"], level)
    found, _ = search_frame(canvas, valid, level, t.truths[0] + np.array([250.0, -180.0]), w["store"], variants=[("zncc", 0.0)])
    return w, canvas, valid, level, found[("zncc", 0.0)]


def test_fit_matches_confidence_cut_and_distance():
    from fine_matching.loftr_matcher import fit_matches, loftr_matches
    matcher = _setup()
    a, b, c = loftr_matches(matcher, np.zeros((224, 224)), np.zeros((224, 224)))
    reliable = np.ones((224, 224), dtype=bool)
    tight = fit_matches(a, b, c, reliable, 3, 0.3, pixels=3.0)
    assert 55 <= tight["n_inliers"] <= 65 and abs(tight["shift"][0] / 0.85) > 0       # the 60 good matches, 0.3 drops the random quarter
    assert abs(tight["n_pairs"] - 90) == 0
    strict = fit_matches(a, b, c, reliable, 3, 0.5, pixels=3.0)
    assert strict["n_pairs"] == 60                                                       # only the confidence-0.9 matches are left
    loose = fit_matches(a, b, c, reliable, 3, 0.3, pixels=20.0)
    assert loose["n_inliers"] > tight["n_inliers"]                                       # a looser distance lets the 12-px-off matches count


def test_stored_matches_give_the_same_answer_as_running_the_matcher():
    from fine_matching.loftr_matcher import candidate_matches, second_stage_loftr, stage2_from_matches
    w, canvas, valid, level, peaks = _peaks_and_world()
    matcher = _setup()
    levels = w["store"]["levels"]
    direct = second_stage_loftr(matcher, canvas, valid, level, peaks[:5], levels)
    stored = candidate_matches(matcher, canvas, valid, level, peaks[:5], levels)
    again = stage2_from_matches(stored, level, matcher.min_confidence)
    assert np.allclose(direct["inliers"], again["inliers"]) and np.allclose(direct["shifts"], again["shifts"])
    strict = stage2_from_matches(stored, level, 0.5)                                    # fewer matches, but the 60 good ones are all above 0.5
    assert np.abs(strict["inliers"] - again["inliers"]).max() <= 2
    nothing = stage2_from_matches(stored, level, 0.95)                                  # no match is that sure: no inliers at all
    assert (nothing["inliers"] == 0).all()


def test_upscale_maps_the_matches_back_to_the_224_frame():
    from fine_matching.loftr_matcher import LoftrMatcher, fit_matches, loftr_matches
    reliable = np.ones((224, 224), dtype=bool)
    plain = LoftrMatcher(StandInLoftr(), 0.3, upscale=1.0)
    double = LoftrMatcher(StandInLoftr(), 0.3, upscale=2.0)
    a1, b1, c1 = loftr_matches(plain, np.zeros((224, 224)), np.zeros((224, 224)))
    a2, b2, c2 = loftr_matches(double, np.zeros((224, 224)), np.zeros((224, 224)))
    assert np.allclose(a1, a2, atol=1e-3) and np.allclose(b1, b2, atol=1e-3)           # same positions in the 224 px frame whatever the enlargement
    assert np.allclose(fit_matches(a1, b1, c1, reliable, 3, 0.3)["shift"], fit_matches(a2, b2, c2, reliable, 3, 0.3)["shift"], atol=1e-3)
    try:
        LoftrMatcher(StandInLoftr(), 0.3, upscale=0.5)
    except AssertionError:
        return
    raise AssertionError("an upscale below 1 must be refused")


def test_empty_slot_is_skipped():
    from fine_matching.loftr_matcher import candidate_matches, stage2_from_matches
    w, canvas, valid, level, peaks = _peaks_and_world()
    empty = np.zeros((1, peaks.shape[1])); empty[0, 1] = empty[0, 2] = 1e9
    padded = np.concatenate([peaks[:2], empty])
    stored = candidate_matches(_setup(), canvas, valid, level, padded, w["store"]["levels"])
    assert stored["items"][2] is None
    assert stage2_from_matches(stored, level, 0.3)["inliers"][2] == 0


def test_border_px_is_applied_when_fitting_and_never_adds_matches():
    from fine_matching.loftr_matcher import candidate_matches, stage2_from_matches
    w, canvas, valid, level, peaks = _peaks_and_world()
    stored = candidate_matches(_setup(), canvas, valid, level, peaks[:3], w["store"]["levels"])
    base = stage2_from_matches(stored, level, 0.3)
    same = stage2_from_matches(stored, level, 0.3, border_px=0)
    near = stage2_from_matches(stored, level, 0.3, border_px=10)
    nothing = stage2_from_matches(stored, level, 0.3, border_px=200)                    # a 401 px erosion leaves no reliable pixel
    assert np.allclose(base["inliers"], same["inliers"])
    assert (near["inliers"] <= base["inliers"]).all() and (nothing["inliers"] == 0).all()


def test_margin_grows_both_pictures_and_keeps_the_shift_frame():
    from fine_matching.loftr_matcher import LoftrMatcher, candidate_matches, prepare_loftr_canvas, stage2_from_matches
    seen = []

    class Recorder(StandInLoftr):
        def __call__(self, batch):
            seen.append(tuple(batch["image0"].shape[2:]) + tuple(batch["image1"].shape[2:]))
            return super().__call__(batch)

    w, canvas, valid, level, peaks = _peaks_and_world()
    padded, inside = prepare_loftr_canvas(canvas, valid, 8)
    assert padded.shape == (240, 240) and inside[:8].sum() == 0 and inside[8:232, 8:232].sum() == valid.sum()
    stored = candidate_matches(LoftrMatcher(Recorder(), 0.3, margin=8), canvas, valid, level, peaks[:2], w["store"]["levels"])
    assert seen[0] == (240, 240, 240, 240) and stored["margin"] == 8
    result = stage2_from_matches(stored, level, 0.3)
    assert result["inliers"].shape == (2,)
    try:
        LoftrMatcher(StandInLoftr(), 0.3, margin=6)
    except AssertionError:
        return
    raise AssertionError("a margin that is not a multiple of 4 must be refused")


def test_batches_give_the_same_matches_as_one_pair_per_call():
    from fine_matching.loftr_matcher import LoftrMatcher, candidate_matches, candidate_matches_group, stage2_from_matches
    calls = []

    class Counter(StandInLoftr):
        def __call__(self, batch):
            calls.append(batch["image0"].shape[0])
            return super().__call__(batch)

    w, canvas, valid, level, peaks = _peaks_and_world()
    levels = w["store"]["levels"]
    alone = candidate_matches(LoftrMatcher(Counter(), 0.3, batch_size=1), canvas, valid, level, peaks[:5], levels)
    assert calls == [1, 1, 1, 1, 1]
    calls.clear()
    together = candidate_matches(LoftrMatcher(Counter(), 0.3, batch_size=3), canvas, valid, level, peaks[:5], levels)
    assert calls == [3, 2]                                                               # 5 pairs, 3 per call
    assert np.allclose(stage2_from_matches(alone, level, 0.3)["inliers"], stage2_from_matches(together, level, 0.3)["inliers"])
    calls.clear()
    jobs = [dict(canvas=canvas, valid=valid, level=level, peaks=peaks[:3]), dict(canvas=canvas, valid=valid, level=level, peaks=peaks[3:6])]
    group = candidate_matches_group(LoftrMatcher(Counter(), 0.3, batch_size=6), jobs, levels)
    assert calls == [6] and len(group) == 2                                              # the pairs of both jobs went through in one call
    assert len(group[0]["items"]) == 3 and len(group[1]["items"]) == 3
    try:
        LoftrMatcher(StandInLoftr(), 0.3, batch_size=0)
    except AssertionError:
        return
    raise AssertionError("a batch size below 1 must be refused")


run_tests(globals())
