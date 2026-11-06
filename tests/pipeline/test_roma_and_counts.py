"""K / M candidate counts, the fine stage for any M, and the RoMa adapter (with a stand-in RoMa model; the real one is not run here)."""
import numpy as np
import torch
from common import run_tests, world


def _peaks_for_first_photo():
    from coarse_search.search import search_frame
    from image_processing.geometry import render_query
    import coarse_search.search as search
    search.DISC_RADIUS = 600.0
    w = world(); t = w["testset"]
    level = int(t.query_levels[0])
    canvas, valid = render_query(w["items"][0], w["camera"], level)
    found, _ = search_frame(canvas, valid, level, t.truths[0] + np.array([250.0, -180.0]), w["store"], variants=[("zncc", 0.0)])
    return w, canvas, valid, level, found[("zncc", 0.0)]


def test_second_stage_any_takes_any_number_of_candidates():
    import fine_matching.second_stage as stage
    import stand_in_matcher
    from fine_matching.second_stage import second_stage
    from fine_matching.second_stage_any import second_stage_any
    stage.find_keypoints = stand_in_matcher.find_keypoints
    stage.match_keypoints = stand_in_matcher.match_keypoints
    w, canvas, valid, level, peaks = _peaks_for_first_photo()
    levels = w["store"]["levels"]
    reference = second_stage(object(), canvas, valid, level, peaks, levels)
    for count in (3, 25):
        out = second_stage_any(object(), canvas, valid, level, peaks[:count], levels)
        assert out["inliers"].shape == (count,) and out["shifts"].shape == (count, 2)
        assert np.allclose(out["inliers"], reference["inliers"][:count])
    long_peaks = np.concatenate([peaks, peaks[:10]])                     # 35 candidates: two groups of 25
    out = second_stage_any(object(), canvas, valid, level, long_peaks, levels)
    assert out["inliers"].shape == (35,)
    assert np.allclose(out["inliers"][25:], out["inliers"][:10])


class StandInRoma:
    """Same three calls as romatch's model; the second picture is the first one moved by (+4, -3) pixels."""
    def match(self, picture_a, picture_b, device=None):
        return None, None

    def sample(self, warp, certainty, num=100):
        side = int(np.sqrt(num))
        grid = (torch.arange(side, dtype=torch.float32) + 0.5) / side * 1.2 - 0.6        # normalised -1..1, kept inside the picture
        ys, xs = torch.meshgrid(grid, grid, indexing="ij")
        shift_x, shift_y = 2 * 4 / 224, 2 * -3 / 224
        matches = torch.stack([xs.reshape(-1), ys.reshape(-1), xs.reshape(-1) + shift_x, ys.reshape(-1) + shift_y], dim=-1)
        return matches, torch.ones(len(matches))

    def to_pixel_coordinates(self, coords, H_A, W_A, H_B, W_B):
        a, b = coords[..., :2], coords[..., 2:]
        to_pixels = lambda p, H, W: torch.stack((W / 2 * (p[..., 0] + 1), H / 2 * (p[..., 1] + 1)), dim=-1)
        return to_pixels(a, H_A, W_A), to_pixels(b, H_B, W_B)


def test_roma_adapter_recovers_the_shift_and_dispatch_picks_it():
    from fine_matching.roma_matcher import RomaMatcher
    from fine_matching.second_stage_any import run_fine_stage
    from image_processing.map_levels import level_gsd
    w, canvas, valid, level, peaks = _peaks_for_first_photo()
    matcher = RomaMatcher(StandInRoma(), 400)
    out = run_fine_stage(matcher, canvas, valid, level, peaks[:5], w["store"]["levels"])
    assert out["inliers"].shape == (5,) and out["shifts"].shape == (5, 2)
    gsd = level_gsd(level)
    assert out["inliers"][0] >= 12, "the stand-in pairs are exact, RANSAC must keep them"
    assert np.allclose(out["shifts"][0], [4 * gsd, -3 * gsd], atol=0.2 * gsd), out["shifts"][0]


def test_roma_adapter_skips_empty_candidates():
    from fine_matching.roma_matcher import RomaMatcher, second_stage_roma
    w, canvas, valid, level, peaks = _peaks_for_first_photo()
    padded = peaks[:3].copy()
    empty = np.zeros((1, peaks.shape[1]))
    empty[0, 1] = empty[0, 2] = 1e9
    padded = np.concatenate([padded, empty])
    out = second_stage_roma(RomaMatcher(StandInRoma(), 400), canvas, valid, level, padded, w["store"]["levels"])
    assert out["inliers"][3] == 0 and (out["shifts"][3] == 0).all()


def test_semantic_layer_rejects_m_larger_than_k_and_coarse_top_returns_m():
    from coarse_search import search
    from header import settings
    from semantic_rerank.dino_rerank import coarse_top, semantic_layer
    torch.manual_seed(0)
    size = 400
    score = torch.rand(size, size, device=settings.DEVICE) * 0.5
    geometry = dict(inside=torch.ones(size, size, dtype=torch.bool, device=settings.DEVICE), side=56,
                    x_m=torch.arange(size, device=settings.DEVICE).float() * 16, y_m=torch.arange(size, device=settings.DEVICE).float() * 16)
    assert coarse_top(score, geometry, 7).shape == (7, 5)
    assert coarse_top(score, geometry, 40).shape == (40, 5)
    try:
        semantic_layer(None, score, geometry, None, None, 3, {}, shortlist=10, keep=20)
    except AssertionError:
        pass
    else:
        raise AssertionError("M > K must be refused")


run_tests(globals())
