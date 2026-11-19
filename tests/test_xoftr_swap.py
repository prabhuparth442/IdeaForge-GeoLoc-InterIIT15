"""Tests of the XoFTR swap (geoloc/xoftr_matcher.py, fine_matching.check_places with a model that has reuse_query / min_confidence). No weights needed. Run: python tests/test_xoftr_swap.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch

import settings
from geoloc import fine_matching, xoftr_matcher
from geoloc.check_settings import check_settings
from geoloc.pipeline import Localiser


class FakeMatcher(torch.nn.Module):
    """Gives 40 matches that agree on one shift of (2, 3) pixels, with a confidence that the bar can cut."""
    reuse_query = False
    min_confidence = 0.0

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, data):
        self.calls += 1
        rng = np.random.default_rng(self.calls)
        points = rng.uniform(20, 200, (40, 2)).astype(np.float32)
        return dict(keypoints0=torch.from_numpy(points), keypoints1=torch.from_numpy(points + np.array([2, 3], np.float32)), confidence=torch.full((40,), 0.05))


def test_off_by_default_and_settings_check():
    assert settings.XOFTR_FROM_HEIGHT_M is None
    check_settings()
    old = settings.XOFTR_FROM_HEIGHT_M
    try:
        settings.XOFTR_FROM_HEIGHT_M = 800.0
        check_settings()
    finally:
        settings.XOFTR_FROM_HEIGHT_M = old


def test_the_height_gate_picks_the_model():
    holder = Localiser.__new__(Localiser)
    holder.loftr = "loftr"
    holder.xoftr = "xoftr"
    old = settings.XOFTR_FROM_HEIGHT_M
    try:
        settings.XOFTR_FROM_HEIGHT_M = 1000.0
        assert holder.fine_model_for(dict(height=999.0)) == "loftr"
        assert holder.fine_model_for(dict(height=1000.0)) == "xoftr"
        holder.xoftr = None
        assert holder.fine_model_for(dict(height=5000.0)) == "loftr"
    finally:
        settings.XOFTR_FROM_HEIGHT_M = old


def test_check_places_uses_the_models_own_confidence_cut_and_skips_query_reuse():
    canvas = np.random.default_rng(0).random((224, 224)).astype(np.float32)
    valid = np.ones((224, 224), bool)
    level_image = (np.random.default_rng(1).random((600, 600)) * 200 + 20).astype(np.uint8)
    peaks = np.array([[0.5, 300.0, 300.0, 0.0, 0.0]])
    fake = FakeMatcher()
    out = fine_matching.check_places(fake, canvas, valid, 3, peaks, level_image)
    assert fake.calls == 1 and out["inliers"][0] >= 30, out["inliers"]              # confidence 0.05 passes because min_confidence is 0
    fake.min_confidence = 0.2
    out = fine_matching.check_places(fake, canvas, valid, 3, peaks, level_image)
    assert out["inliers"][0] == 0                                                     # the same matches are cut by the model's own bar


def test_xoftr_model_interface_with_a_stub():
    class Stub(torch.nn.Module):
        def forward(self, batch):
            batch["mkpts0_f"] = torch.zeros(3, 2); batch["mkpts1_f"] = torch.ones(3, 2); batch["mconf_f"] = torch.full((3,), 0.5)
    model = xoftr_matcher.XoftrModel(Stub(), 0.0)
    out = model({"image0": torch.zeros(1, 1, 8, 8), "image1": torch.zeros(1, 1, 8, 8)})
    assert set(out) == {"keypoints0", "keypoints1", "confidence"} and out["keypoints0"].shape == (3, 2)
    assert model.reuse_query is False and model.min_confidence == 0.0


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
