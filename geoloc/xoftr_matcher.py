"""XoFTR (MINIMA weights) as a swap-in for the LoFTR model in the fine stage, for photos above a height (settings.XOFTR_FROM_HEIGHT_M).

Why (iteration 36, 2026-11-18): at the true place on the high-altitude development photos XoFTR finds the place more often than LoFTR (62 % against 46 % above 800 m, 49 % against 18 % above 1400 m,
with a bar set per matcher from its own wrong-place counts) at the same 224 px input and the same time per pair. XoFTR is LoFTR's cross-modality successor: the same coarse 1/8 grid, then a medium (1/4)
and a fine (1/2) refinement trained on thermal-visible pairs. On the low-height official set it was worse than LoFTR (Test_Iteration_4), so it is used only from the gate up.

XoftrModel looks like the kornia LoFTR the pipeline already uses: it takes {"image0", "image1"} (1, 1, H, W) pictures in 0..1 and returns keypoints0, keypoints1 (pixels) and confidence, so fine_matching
(loftr_matches, check_places, RANSAC, the shift rule) works unchanged. Two attributes tell check_places about the differences: reuse_query = False (LoFTR's trick of running the canvas through the CNN once
per query does not apply) and min_confidence (matches below it are dropped before RANSAC; 0 = keep every match XoFTR itself kept).
The model code is third_party/XoFTR (Apache-2.0, inference only, no yacs or lightning). minima_xoftr.ckpt is a training checkpoint with the weights under "matcher.".
"""
import os
import sys

import torch

import settings
from geoloc.coarse_search import DEVICE

THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "XoFTR")


def xoftr_config(coarse_threshold, fine_threshold):
    """The default XoFTR inference config (src/config/default.py of XoFTR with INFERENCE = True), written out so no config library is needed."""
    return dict(
        resolution=(8, 2), fine_window_size=5, medium_window_size=3,
        resnet=dict(initial_dim=128, block_dims=[128, 196, 256]),
        coarse=dict(inference=True, d_model=256, d_ffn=256, nhead=8, layer_names=["self", "cross"] * 4, attention="linear"),
        match_coarse=dict(inference=True, d_model=256, thr=coarse_threshold, border_rm=2, match_type="dual_softmax", dsmax_temperature=0.1,
                          train_coarse_percent=0.2, train_pad_num_gt_min=200),
        fine=dict(denser=False, inference=True, dsmax_temperature=0.1, thr=fine_threshold, mlp_hidden_dim_coef=2, nhead_fine_level=8, nhead_medium_level=7),
    )


class XoftrModel(torch.nn.Module):
    reuse_query = False

    def __init__(self, xoftr, min_confidence):
        super().__init__()
        self.xoftr = xoftr
        self.min_confidence = min_confidence

    def forward(self, data):
        batch = {"image0": data["image0"], "image1": data["image1"]}
        self.xoftr(batch)
        return dict(keypoints0=batch["mkpts0_f"], keypoints1=batch["mkpts1_f"], confidence=batch["mconf_f"])


def load_xoftr(weights_file):
    """XoFTR with the MINIMA weights, read from the local file (nothing is downloaded), strict loading."""
    if THIRD_PARTY not in sys.path:
        sys.path.insert(0, THIRD_PARTY)
    from xoftr import XoFTR

    saved = torch.load(weights_file, map_location="cpu", weights_only=False)      # our own checkpoint file
    weights = {}
    for name, value in saved["state_dict"].items():
        if name.startswith("matcher."):
            weights[name[len("matcher."):]] = value
    if len(weights) == 0:
        raise ValueError("no 'matcher.' weights in " + weights_file)
    model = XoFTR(config=xoftr_config(settings.XOFTR_COARSE_THRESHOLD, settings.XOFTR_FINE_THRESHOLD))
    model.load_state_dict(weights, strict=True)
    return XoftrModel(model, settings.XOFTR_MIN_CONFIDENCE).to(DEVICE).eval()
