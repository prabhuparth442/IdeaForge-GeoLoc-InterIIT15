"""Block: XoFTR (MINIMA weights) as a drop-in for the batched LoFTR plumbing of src/fine_matching/loftr_matcher.py. No plotting.

src/fine_matching/xoftr_matcher.py was written before loftr_matcher.run_loftr_chunk learned to batch pairs; run_loftr_chunk now reads
`batch_indexes` (which pair each match belongs to), which XoftrModel does not return, so every XoFTR call failed with KeyError('batch_indexes').
This block wraps the same model and adds the batch index from XoFTR's own `m_bids` (all 0 for a single pair). src/ is left unchanged.

load_xoftr_batched(weights_file, ...)   same arguments as fine_matching.xoftr_matcher.load_xoftr_matcher; returns a LoftrMatcher
"""
import torch

from fine_matching.loftr_matcher import LoftrMatcher
from fine_matching import xoftr_matcher


class XoftrBatched(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner                                   # an xoftr_matcher.XoftrModel
        self.coarse_matching = inner.coarse_matching

    def forward(self, data):
        batch = {"image0": data["image0"], "image1": data["image1"]}
        self.inner.xoftr(batch)
        points0 = batch["mkpts0_f"]
        ids = batch.get("m_bids")
        if ids is None or len(ids) != len(points0):
            assert data["image0"].shape[0] == 1 or len(points0) == 0, "XoFTR returned no batch ids for a batch of several pairs"
            ids = torch.zeros(len(points0), dtype=torch.long, device=points0.device)
        return dict(keypoints0=points0, keypoints1=batch["mkpts1_f"], confidence=batch["mconf_f"], batch_indexes=ids)


def load_xoftr_batched(weights_file, min_confidence=0.1, coarse_threshold=0.3, fine_threshold=0.1, upscale=1.0, border_px=0, margin=0):
    plain = xoftr_matcher.load_xoftr_matcher(weights_file, min_confidence, coarse_threshold, fine_threshold, upscale, border_px, margin)
    return LoftrMatcher(XoftrBatched(plain.loftr_model), min_confidence, upscale, border_px, margin)
