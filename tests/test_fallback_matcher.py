"""Tests of the second-matcher fallback (Localiser.merge_fallback, other_model_for). No weights needed. Run: python tests/test_fallback_matcher.py"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch

import settings
from geoloc.check_settings import check_settings
from geoloc.pipeline import Localiser


class Fake(torch.nn.Module):
    """n matches that agree on one shift, confidence above any cut."""
    reuse_query = False
    min_confidence = 0.0

    def __init__(self, n, shift):
        super().__init__()
        self.n = n
        self.shift = np.array(shift, np.float32)
        self.calls = 0

    def forward(self, data):
        self.calls += 1
        points = np.random.default_rng(7).uniform(20, 200, (self.n, 2)).astype(np.float32)
        return dict(keypoints0=torch.from_numpy(points), keypoints1=torch.from_numpy(points + self.shift), confidence=torch.full((self.n,), 0.9))


def holder(primary, secondary):
    h = Localiser.__new__(Localiser)
    h.loftr = primary
    h.xoftr = secondary
    h.store = SimpleNamespace(levels={3: np.zeros((600, 600), np.uint8)})
    h.fallback_native = None
    h.fallback_queries = 0
    return h


def test_defaults_off_and_checked():
    assert settings.FALLBACK_MATCHER is False and settings.FALLBACK_NATIVE_FROM_HEIGHT_M is None
    check_settings()
    old = settings.FALLBACK_MATCHER
    try:
        settings.FALLBACK_MATCHER = True
        check_settings()
    finally:
        settings.FALLBACK_MATCHER = old


def test_other_model_is_the_one_the_photo_does_not_use():
    h = holder("loftr", "xoftr")
    old = settings.XOFTR_FROM_HEIGHT_M
    try:
        settings.XOFTR_FROM_HEIGHT_M = None
        assert h.fine_model_for(dict(height=2000.0)) == "loftr" and h.other_model_for(dict(height=2000.0)) == "xoftr"
        settings.XOFTR_FROM_HEIGHT_M = 800.0
        assert h.fine_model_for(dict(height=2000.0)) == "xoftr" and h.other_model_for(dict(height=2000.0)) == "loftr"
        assert h.fine_model_for(dict(height=300.0)) == "loftr" and h.other_model_for(dict(height=300.0)) == "xoftr"
    finally:
        settings.XOFTR_FROM_HEIGHT_M = old


def test_each_candidate_keeps_the_better_count_and_its_shift():
    primary, secondary = Fake(5, (1, 1)), Fake(40, (2, 3))
    h = holder(primary, secondary)
    canvas = np.random.default_rng(0).random((224, 224)).astype(np.float32)
    valid = np.ones((224, 224), bool)
    level_image = h.store.levels[3]
    peaks = np.array([[0.5, 300.0, 300.0, 0.0, 0.0], [0.4, 320.0, 320.0, 0.0, 0.0]])
    first = {"inliers": np.array([60.0, 0.0]), "shifts": np.array([[1.0, 1.0], [0.0, 0.0]]), "aligned": np.full(2, -np.inf)}
    old = settings.XOFTR_FROM_HEIGHT_M
    try:
        settings.XOFTR_FROM_HEIGHT_M = None
        merged = h.merge_fallback(first, dict(height=300.0), None, 3, peaks, canvas, valid, level_image, "picture", dict(heading=0.0, scale_factor=1.0))
    finally:
        settings.XOFTR_FROM_HEIGHT_M = old
    assert h.fallback_queries == 1 and secondary.calls == 2 and primary.calls == 0
    assert merged["inliers"][0] == 60.0 and merged["shifts"][0].tolist() == [1.0, 1.0]       # the first matcher was better on candidate 0: kept
    assert merged["inliers"][1] >= 30                                                           # the second matcher found candidate 1: taken, with its shift
    assert abs(merged["shifts"][1][0] - 2 * 0.85 * 1.0) < 3.0


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
