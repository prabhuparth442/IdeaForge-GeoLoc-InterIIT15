"""Whole chain on the synthetic world, with the stand-in matcher (no dataset, no lightglue)."""
import numpy as np
from common import run_tests, world


def test_chain_with_stand_in_matcher():
    import coarse_search.search as search
    import fine_matching.second_stage as stage
    import stand_in_matcher
    from result_thresholding.pipeline import localise_photo
    search.DISC_RADIUS = 600.0
    stage.find_keypoints = stand_in_matcher.find_keypoints
    stage.match_keypoints = stand_in_matcher.match_keypoints
    w = world(); t = w["testset"]
    for i in range(3):
        r = localise_photo(w["items"][i], t.truths[i] + np.array([250.0, -180.0]), w["camera"], w["store"], matcher=object())
        err = t.error_m(i, r["x"], r["y"])
        print("   photo %d: error %.1f m, inliers %d, answer %s, %.1f s" % (i, err, r["inliers"], r["answer"], r["seconds"]))
        assert err < 12


def test_report_runs_on_synthetic_results():
    from evaluation.run_sample import run_correlation
    from evaluation.report import collect_answers
    w = world()
    disc, seconds = run_correlation(w["testset"], w["store"], [("zncc", 0.0)])
    results, closest, ratios, counts = collect_answers(w["testset"], disc, {}, {}, False)
    assert len(results["correlation"]["points"]) == 3 and len(closest) == 3


run_tests(globals())
