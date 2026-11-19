"""Tests of LOFTR_REUSE_QUERY (the canvas goes through LoFTR's CNN once per query) and of the second pass options.
No data and no weights: kornia's LoFTR with random weights is enough to show the matches are the same. Run: python tests/test_query_reuse.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch

import settings
from geoloc import decision, fine_matching, queries


def small_loftr():
    from kornia.feature import LoFTR

    torch.manual_seed(0)
    model = LoFTR(pretrained=None)
    model.coarse_matching.thr = 0.0        # random weights: no cut, so that matches exist to compare
    return model.eval()


def pictures():
    rng = np.random.default_rng(1)
    canvas = rng.random((224, 224)).astype(np.float32)
    tiles = [np.clip(canvas + 0.05 * rng.random((224, 224)).astype(np.float32) * (k + 1), 0, 1) for k in range(3)]
    return canvas, tiles


def test_backbone_features_are_the_same():
    model = small_loftr()
    canvas, tiles = pictures()
    batch = torch.cat([fine_matching.as_tensor(canvas).repeat(3, 1, 1, 1), torch.cat([fine_matching.as_tensor(tile) for tile in tiles])])
    with torch.no_grad():
        plain = model.backbone(batch)
        wrapper = fine_matching.QueryBackbone(model.backbone)
        reused = wrapper(batch)
    assert len(plain) == len(reused) == 2
    for a, b in zip(plain, reused):
        assert a.shape == b.shape
        assert float((a - b).abs().max()) < 1e-5


def test_reuse_gives_the_same_matches_one_at_a_time():
    model = small_loftr()
    canvas, tiles = pictures()
    plain = [fine_matching.loftr_matches(model, canvas, tile) for tile in tiles]
    with fine_matching.cached_query_backbone(model):
        reused = [fine_matching.loftr_matches(model, canvas, tile) for tile in tiles]
    for a, b in zip(plain, reused):
        assert len(a[0]) > 0, "the random model found no match, the test would show nothing"
        for x, y in zip(a, b):
            assert x.shape == y.shape
            assert np.max(np.abs(x - y)) < 1e-4, np.max(np.abs(x - y))


def test_reuse_gives_the_same_matches_batched():
    model = small_loftr()
    canvas, tiles = pictures()
    plain = fine_matching.loftr_matches_batch(model, canvas, tiles)
    with fine_matching.cached_query_backbone(model):
        reused = fine_matching.loftr_matches_batch(model, canvas, tiles)
    compared = 0
    for a, b in zip(plain, reused):
        for x, y in zip(a, b):
            assert x.shape == y.shape
            if x.size > 0:
                assert np.max(np.abs(x - y)) < 1e-4
                compared += 1
    assert compared > 0, "the random model found no match, the test would show nothing"


def test_backbone_is_put_back_and_the_cache_refuses_another_canvas():
    model = small_loftr()
    original = model.backbone
    canvas, tiles = pictures()
    with fine_matching.cached_query_backbone(model):
        fine_matching.loftr_matches(model, canvas, tiles[0])
        try:
            fine_matching.loftr_matches(model, canvas * 0.5, tiles[1])
        except AssertionError:
            pass
        else:
            raise AssertionError("a different canvas must not use the cached features")
    assert model.backbone is original


def with_settings(**values):
    old = {key: getattr(settings, key) for key in values}
    for key, value in values.items():
        setattr(settings, key, value)
    return old


def restore(old):
    for key, value in old.items():
        setattr(settings, key, value)


def test_second_pass_default_is_the_frozen_set():
    old = with_settings(SECOND_PASS=True, QUERY_SET="standard", SECOND_PASS_HEADINGS=None, SECOND_PASS_SCALES=None)
    try:
        extra = queries.second_pass_queries()
        assert len(extra) == 12
        assert all(q["scale_factor"] in (1.0, 1.05) for q in extra)
        assert len(queries.query_list()) + len(extra) == 18
    finally:
        restore(old)


def test_second_pass_wider_scales_skips_what_the_first_pass_ran():
    old = with_settings(SECOND_PASS=True, QUERY_SET="standard", SECOND_PASS_HEADINGS=[-5.0, 0.0, 5.0], SECOND_PASS_SCALES=[0.8, 0.85, 0.9, 1.0, 1.05])
    try:
        extra = queries.second_pass_queries()
        assert len(extra) == 9                 # 15 queries in all, 6 already run
        assert all(q["scale_factor"] in (0.8, 0.85, 0.9) for q in extra)
    finally:
        restore(old)


def test_second_pass_bar():
    old = with_settings(MIN_INLIERS=8, SECOND_PASS_MIN_INLIERS=None)
    try:
        assert decision.should_answer(8, 1.0, True)
        settings.SECOND_PASS_MIN_INLIERS = 11
        assert not decision.should_answer(10, 1.0, True)
        assert decision.should_answer(11, 1.0, True)
        assert decision.should_answer(8, 1.0, False)       # the first pass keeps the normal bar
    finally:
        restore(old)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("all %d passed" % len(tests))
