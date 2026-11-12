"""XoFTR (with the MINIMA weights) as the fine matcher, a swap-in for LoFTR in loftr_matcher.py.

XoFTR is LoFTR's cross-modality successor: same coarse 1/8 grid, but a second (1/2) refinement with a medium level and a sub-pixel step that is
trained on thermal-visible pairs. We do not copy the LoFTR plumbing. XoftrModel is a thin module that accepts what the LoFTR model accepts
({"image0", "image1"}, pictures 0..1) and returns what loftr_matches reads (keypoints0, keypoints1, confidence). load_xoftr_matcher returns an
ordinary LoftrMatcher around it, so check_candidate_loftr, candidate_matches, stage2_from_matches, second_stage_loftr and run_fine_stage all work
unchanged (the same RANSAC, the same shift rule, the same result dict).

The model code is third_party/XoFTR (inference-only copy, no yacs or lightning needed: the config is written out below).
minima_xoftr.ckpt is a training checkpoint with the weights under "matcher."; the loader strips the prefix and loads strictly.
Pictures: sides must be multiples of 8 (CTX = 224 is). XoFTR normalises each picture itself (zero mean, unit std), so nothing to do here.
Two thresholds: coarse (the dual-softmax cut, MINIMA default 0.3) and fine (0.1). The "confidence" returned is the fine confidence.
"""
import sys
from pathlib import Path

import torch

from fine_matching.loftr_matcher import LoftrMatcher
from header.settings import DEVICE

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "XoFTR"


def xoftr_config(coarse_threshold, fine_threshold):
    """The default XoFTR inference config (same numbers as src/config/default.py of XoFTR with INFERENCE = True)."""
    return dict(
        resolution=(8, 2), fine_window_size=5, medium_window_size=3,
        resnet=dict(initial_dim=128, block_dims=[128, 196, 256]),
        coarse=dict(inference=True, d_model=256, d_ffn=256, nhead=8, layer_names=["self", "cross"] * 4, attention="linear"),
        match_coarse=dict(inference=True, d_model=256, thr=coarse_threshold, border_rm=2, match_type="dual_softmax", dsmax_temperature=0.1,
                          train_coarse_percent=0.2, train_pad_num_gt_min=200),
        fine=dict(denser=False, inference=True, dsmax_temperature=0.1, thr=fine_threshold, mlp_hidden_dim_coef=2, nhead_fine_level=8, nhead_medium_level=7),
    )


class XoftrModel(torch.nn.Module):
    """Makes XoFTR look like kornia's LoFTR: input dict of two (1, 1, H, W) pictures, output dict of matches in pixels with a confidence."""

    def __init__(self, xoftr):
        super().__init__()
        self.xoftr = xoftr
        self.coarse_matching = xoftr.coarse_matching        # so a notebook can set coarse_matching.thr, as it does for LoFTR

    def forward(self, data):
        batch = {"image0": data["image0"], "image1": data["image1"]}
        self.xoftr(batch)
        return dict(keypoints0=batch["mkpts0_f"], keypoints1=batch["mkpts1_f"], confidence=batch["mconf_f"])


def load_xoftr_matcher(weights_file, min_confidence=0.1, coarse_threshold=0.3, fine_threshold=0.1, upscale=1.0, border_px=0, margin=0):
    """XoFTR with the MINIMA weights (read from disk). min_confidence: matches below it are dropped when fitting (the fine threshold already cuts at
    fine_threshold, so a value under it changes nothing). coarse_threshold, fine_threshold: XoFTR's own cuts (MINIMA defaults 0.3 and 0.1)."""
    assert 0 < coarse_threshold < 1 and 0 < fine_threshold < 1, "thresholds must be between 0 and 1"
    if str(THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY))
    from xoftr import XoFTR

    saved = torch.load(str(weights_file), map_location="cpu", weights_only=False)     # our own checkpoint file
    assert "state_dict" in saved, "%s is not an XoFTR training checkpoint (no state_dict)" % weights_file
    weights = {name[len("matcher."):]: value for name, value in saved["state_dict"].items() if name.startswith("matcher.")}
    assert len(weights) > 0, "no 'matcher.' weights found in %s" % weights_file
    model = XoFTR(config=xoftr_config(coarse_threshold, fine_threshold))
    model.load_state_dict(weights, strict=True)
    wrapped = XoftrModel(model).to(DEVICE).eval()
    return LoftrMatcher(wrapped, min_confidence, upscale, border_px, margin)
