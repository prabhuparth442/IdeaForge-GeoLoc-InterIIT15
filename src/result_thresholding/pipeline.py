import time

import cv2
import torch

from coarse_search.search import search_frame
from fine_matching.second_stage import second_stage
from header.settings import DEVICE, SEED
from image_processing.geometry import render_query
from image_processing.map_levels import level_for_height
from result_thresholding.decision import decide, final_answer


def localise_photo(item, prior_m, camera, store, matcher=None):
    clock = time.time()
    cv2.setRNGSeed(SEED)
    level = level_for_height(item["rel_height"], camera)
    canvas, valid = render_query(item, camera, level)
    found, _ = search_frame(canvas, valid, level, prior_m, store, variants=[("zncc", 0.0)])
    peaks = found[("zncc", 0.0)]
    ratio = peaks[0, 0] / max(peaks[1, 0], 1e-6)
    if matcher is not None:
        stage2 = second_stage(matcher, canvas, valid, level, peaks, store["levels"])
        x, y, inliers, chosen = final_answer(peaks, stage2)
    else:
        stage2, x, y, inliers, chosen = None, peaks[0, 1], peaks[0, 2], 0, 0
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return dict(x=float(x), y=float(y), answer=decide(inliers), inliers=int(inliers), chosen=int(chosen),
                ratio=float(ratio), peaks=peaks, stage2=stage2, seconds=time.time() - clock)
