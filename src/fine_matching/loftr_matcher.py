"""LoFTR (with the MINIMA weights) as the fine matcher: between SuperPoint + LightGlue (sparse, fast) and RoMa (dense, slow).

LoFTR is detector-free: it matches the two pictures at 1/8 resolution and refines each match at 1/2 resolution, so it finds matches on
low-texture, cross-modality pairs where keypoint detectors fail, at a fraction of RoMa's cost. The pair is (the north-up canvas, a map
window around a candidate). The matches go through the same RANSAC (fine_matching.ransac.fit_transform) and the same shift rule as
check_candidate, so the result dict is the one second_stage gives and final_answer / decide work unchanged.

The model is kornia's LoFTR (already a dependency). minima_loftr.ckpt is a training checkpoint whose weights are stored under the prefix
"matcher."; load_loftr_matcher strips it and requires every key to match (strict loading), so a wrong file fails loudly.
The picture sides must be multiples of 8 (CTX = 224 is).

Two settings about the query picture (both off by default, both tested in Stage_Evaluation.ipynb):
  border_px: matches within this many pixels of the edge of the valid (really photographed) area of the canvas are dropped. 0 = keep all.
             It is applied when the matches are fitted, so it can be swept without running the network again.
  margin:    the canvas and the map window are both grown by this many grey / map pixels on every side before LoFTR sees them (a multiple of 4),
             so matches near the canvas edge have context. The network must be run again for every value.

Inlier counts are on a different scale from LightGlue's (measured on 100 official photos: median 30 inliers on the correct place, 0 to 4 on
wrong places), so the answer bar (min_inliers) and the DINO decision constants (GAMMA, rescue rules) need their own values for this matcher.
"""
import math

import cv2
import numpy as np
import torch

from fine_matching.ransac import fit_transform
from header import settings
from header.settings import CELL, CONTEXT_MARGIN, CTX, DEVICE, MAX_SHIFT_CELLS
from image_processing.map_levels import level_gsd, map_piece


class LoftrMatcher:
    def __init__(self, loftr_model, min_confidence, upscale=1.0, border_px=0, margin=0):
        assert upscale >= 1.0, "upscale must be 1 or more (the pictures are only enlarged)"
        assert border_px >= 0, "border_px must be 0 or more"
        assert margin >= 0 and margin % 4 == 0, "margin must be a multiple of 4 pixels (LoFTR needs sides that are multiples of 8)"
        self.border_px = border_px
        self.margin = margin
        self.loftr_model = loftr_model
        self.min_confidence = min_confidence
        self.upscale = upscale


def load_loftr_matcher(weights_file, min_confidence=0.2, upscale=1.0, coarse_threshold=None, border_px=0, margin=0):
    """kornia LoFTR with the MINIMA weights (read from disk, nothing is downloaded). min_confidence: matches below it are dropped.
    upscale: both pictures are enlarged by this factor before LoFTR sees them (224 px -> a 28 x 28 coarse grid, 2.0 -> 56 x 56), the matches are
             mapped back to the 224 px frame. 1.0 = as before.
    border_px, margin: see the top of this file (0 = off).
    coarse_threshold: LoFTR's own cut on the coarse matching probability (None = kornia's 0.2). Lower = more, weaker matches."""
    from kornia.feature import LoFTR

    saved = torch.load(str(weights_file), map_location="cpu", weights_only=False)     # our own checkpoint file
    assert "state_dict" in saved, "%s is not a LoFTR training checkpoint (no state_dict)" % weights_file
    weights = {name[len("matcher."):]: value for name, value in saved["state_dict"].items() if name.startswith("matcher.")}
    assert len(weights) > 0, "no 'matcher.' weights found in %s" % weights_file
    model = LoFTR(pretrained=None)
    model.load_state_dict(weights, strict=True)
    if coarse_threshold is not None:
        assert 0 < coarse_threshold < 1, "coarse_threshold must be between 0 and 1"
        model.coarse_matching.thr = coarse_threshold
    model.to(DEVICE).eval()
    return LoftrMatcher(model, min_confidence, upscale, border_px, margin)


def as_tensor(gray):
    """(H, W) floats 0..1 -> (1, 1, H, W) on the device."""
    return torch.from_numpy(np.clip(np.asarray(gray, dtype=np.float32), 0.0, 1.0))[None, None].to(DEVICE)


@torch.no_grad()
def loftr_matches(loftr_matcher, picture_a, picture_b):
    """Every match LoFTR proposes (no confidence cut): positions in picture_a (n, 2), in picture_b (n, 2) and the confidence (n,).
    LoFTR itself drops matches below 0.2, so a min_confidence under 0.2 changes nothing."""
    assert picture_a.shape == picture_b.shape and picture_a.shape[0] % 8 == 0 and picture_a.shape[1] % 8 == 0, "LoFTR needs pictures with sides that are multiples of 8"
    size = picture_a.shape[0]
    big = int(round(size * loftr_matcher.upscale / 8)) * 8                     # a multiple of 8 again
    if big != size:
        picture_a = cv2.resize(np.asarray(picture_a, dtype=np.float32), (big, big), interpolation=cv2.INTER_CUBIC)
        picture_b = cv2.resize(np.asarray(picture_b, dtype=np.float32), (big, big), interpolation=cv2.INTER_CUBIC)
    found = loftr_matcher.loftr_model({"image0": as_tensor(picture_a), "image1": as_tensor(picture_b)})
    back = size / big                                                           # matches go back to the original picture frame
    return (found["keypoints0"].cpu().numpy().astype(np.float32) * back, found["keypoints1"].cpu().numpy().astype(np.float32) * back,
            found["confidence"].cpu().numpy().astype(np.float32))


def loftr_pairs(loftr_matcher, picture_a, picture_b):
    """Matched pixel positions (n, 2) in picture_a and picture_b (both CTX x CTX) with a confidence of at least min_confidence."""
    points_a, points_b, confidence = loftr_matches(loftr_matcher, picture_a, picture_b)
    keep = confidence >= loftr_matcher.min_confidence
    return points_a[keep], points_b[keep]


def settings_pixels(pixels):
    if pixels is None:
        return settings.RANSAC_PIXELS
    return pixels


def fit_matches(points_a, points_b, confidence, reliable, level, min_confidence, pixels=None, fit=None):
    """Matches -> inliers and the position correction: drop low confidence and unreliable-border matches, RANSAC, same shift rule as check_candidate.
    fit: another fitting function (points_a, points_b, pixels, confidence) -> (matrix, inliers), see fine_matching.robust_fit; None = RANSAC."""
    size = reliable.shape[0]
    keep = confidence >= min_confidence
    points_a, points_b, confidence = points_a[keep], points_b[keep], confidence[keep]
    columns = np.clip(np.round(points_a[:, 0]).astype(int), 0, size - 1)
    rows = np.clip(np.round(points_a[:, 1]).astype(int), 0, size - 1)
    keep = reliable[rows, columns]
    points_a, points_b, confidence = points_a[keep], points_b[keep], confidence[keep]
    if fit is None:
        matrix, inliers = fit_transform(points_a, points_b, pixels)
    else:
        matrix, inliers = fit(points_a, points_b, settings_pixels(pixels), confidence)
    result = dict(n_pairs=len(points_a), n_inliers=int(inliers.sum()), shift=np.zeros(2), points_a=points_a, points_b=points_b, inliers=inliers)
    if matrix is not None:
        middle = (size - 1) / 2
        mapped = matrix @ np.array([middle, middle, 1.0])
        result["shift"] = (mapped - middle) * level_gsd(level)
    return result


def prepare_loftr_canvas(canvas, valid, margin=0):
    """The canvas (TILE x TILE) grown by `margin` grey pixels on every side, and the bool mask of the pixels that were really photographed."""
    assert canvas.shape == valid.shape and valid.dtype == bool, "canvas and valid must have the same shape, valid must be bool"
    assert margin >= 0 and margin % 4 == 0, "margin must be a multiple of 4 pixels"
    fill = float(canvas[valid].mean()) if valid.any() else float(canvas.mean())
    size = canvas.shape[0] + 2 * margin
    padded = np.full((size, size), fill, dtype=np.float32)
    inside = np.zeros((size, size), dtype=bool)
    padded[margin:margin + canvas.shape[0], margin:margin + canvas.shape[1]] = np.where(valid, canvas, fill)
    inside[margin:margin + canvas.shape[0], margin:margin + canvas.shape[1]] = valid
    return padded, inside


def reliable_mask(inside, border_px):
    """Where a match in the canvas may lie: the photographed area, shrunk by border_px (0 = everywhere, also the grey filler)."""
    assert border_px >= 0, "border_px must be 0 or more"
    if border_px == 0:
        return np.ones(inside.shape, dtype=bool)
    side = 2 * int(border_px) + 1
    return cv2.erode(inside.astype(np.uint8), np.ones((side, side), np.uint8)).astype(bool)


def check_candidate_loftr(loftr_matcher, canvas_picture, reliable, level_images, level, x_m, y_m):
    assert level in level_images, "level %s was not built" % level
    assert math.isfinite(x_m) and math.isfinite(y_m) and x_m < 1e8, "candidate position must be finite metres (empty slots have x = 1e9)"
    tile = map_piece(level_images, level, x_m, y_m, CTX)
    points_a, points_b, confidence = loftr_matches(loftr_matcher, canvas_picture, tile)
    result = fit_matches(points_a, points_b, confidence, reliable, level, loftr_matcher.min_confidence)
    result["tile"] = tile
    return result


def candidate_matches(loftr_matcher, canvas, valid, level, peaks, level_images):
    """The raw LoFTR matches of every candidate, to be turned into inliers later with stage2_from_matches (any min_confidence, any RANSAC distance)
    without running the network again. Returns dict(inside=..., margin=..., border_px=..., items=[dict(points_a, points_b, confidence) or None for an empty slot])."""
    assert peaks.ndim == 2 and peaks.shape[1] >= 3, "peaks must be (M, >= 3), got %s" % (peaks.shape,)
    margin = loftr_matcher.margin
    canvas_picture, inside = prepare_loftr_canvas(canvas, valid, margin)
    items = []
    for k in range(len(peaks)):
        if peaks[k, 1] > 1e8:
            items.append(None)
            continue
        tile = map_piece(level_images, level, peaks[k, 1], peaks[k, 2], canvas_picture.shape[0])
        points_a, points_b, confidence = loftr_matches(loftr_matcher, canvas_picture, tile)
        items.append(dict(points_a=points_a, points_b=points_b, confidence=confidence))
    return dict(inside=inside, margin=margin, border_px=loftr_matcher.border_px, items=items)


def stage2_from_matches(stored, level, min_confidence, pixels=None, border_px=None, fit=None):
    """Same output as second_stage (inliers, shifts) from the matches of candidate_matches. border_px: None = the value the matches were made with."""
    reliable = reliable_mask(stored["inside"], stored["border_px"] if border_px is None else border_px)
    count = len(stored["items"])
    inliers = np.zeros(count)
    shifts = np.zeros((count, 2))
    limit = (MAX_SHIFT_CELLS * CELL + (CONTEXT_MARGIN + stored["margin"]) // 2) * level_gsd(level)
    for k, item in enumerate(stored["items"]):
        if item is None:
            continue
        result = fit_matches(item["points_a"], item["points_b"], item["confidence"], reliable, level, min_confidence, pixels, fit)
        inliers[k] = result["n_inliers"]
        if math.hypot(*result["shift"]) <= limit:
            shifts[k] = result["shift"]
        else:
            inliers[k] = 0
    return dict(inliers=inliers, shifts=shifts)


def second_stage_loftr(loftr_matcher, canvas, valid, level, peaks, level_images):
    """Same output as second_stage (inliers, shifts), for any number of candidates."""
    stored = candidate_matches(loftr_matcher, canvas, valid, level, peaks, level_images)
    return stage2_from_matches(stored, level, loftr_matcher.min_confidence)
