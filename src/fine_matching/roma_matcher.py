"""RoMa (with the MINIMA weights) as the fine matcher, instead of SuperPoint + LightGlue.

RoMa is a dense matcher: for a pair of pictures it gives, for every pixel of the first one, where it lands in the second and how sure it is.
Here the pair is (the north-up canvas, a map window around a candidate). From the dense result `num_matches` pairs are drawn in proportion to
their certainty, then the same RANSAC (fine_matching.ransac.fit_transform) and the same shift rule as check_candidate turn them into
`n_inliers` and a position correction, so the result dict is the one second_stage gives and final_answer / decide work unchanged.

Needs the `romatch` package (pip install romatch) and two local files: the MINIMA RoMa weights (minima_roma.pth) and the DINOv2 ViT-L
weights RoMa uses as its coarse encoder (dinov2_vitl14_pretrain.pth). No download happens when both paths are given.
NOT YET RUN against the real model on this machine: the adapter is tested with a stand-in model only (tests/pipeline/test_roma_and_counts.py).
"""
import math

import cv2
import numpy as np
import torch
from PIL import Image

from fine_matching.ransac import fit_transform
from header.settings import CELL, CONTEXT_MARGIN, CTX, DEVICE, MAX_SHIFT_CELLS, SEED, TILE, USE_BORDER_MASK
from image_processing.map_levels import level_gsd, map_piece


class RomaMatcher:
    def __init__(self, roma_model, num_matches):
        self.roma_model = roma_model
        self.num_matches = num_matches


def load_roma_matcher(weights_file, dinov2_file, coarse_res=560, upsample_res=(672, 672), num_matches=5000, use_custom_corr=None):
    """RoMa outdoor architecture with the MINIMA weights. coarse_res must be a multiple of 14; the 224 px inputs are resized to the model's
    working sizes (coarse_res, then upsample_res). Both files are read from disk, nothing is downloaded.
    use_custom_corr: RoMa's fast CUDA correlation needs the optional package `local_corr` (pip install fused-local-corr).
    None = use it when it is installed, otherwise the plain PyTorch version (slower, same result)."""
    import importlib.util

    from romatch import roma_outdoor

    if use_custom_corr is None:
        use_custom_corr = DEVICE == "cuda" and importlib.util.find_spec("local_corr") is not None
    amp = torch.float16 if DEVICE == "cuda" else torch.float32
    weights = torch.load(str(weights_file), map_location=DEVICE)
    dinov2_weights = torch.load(str(dinov2_file), map_location=DEVICE)
    model = roma_outdoor(DEVICE, weights=weights, dinov2_weights=dinov2_weights, coarse_res=coarse_res, upsample_res=upsample_res, amp_dtype=amp,
                         use_custom_corr=use_custom_corr)
    model.eval()
    return RomaMatcher(model, num_matches)


def as_rgb_picture(gray):
    """(H, W) floats 0..1 -> PIL RGB picture."""
    levels = np.clip(np.asarray(gray, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([levels, levels, levels], axis=-1))


def prepare_roma_canvas(canvas, valid):
    """Canvas as the matcher sees it: pixels outside the photo are filled with the mean of the photo (as prepare_canvas does).
    Returns the picture (CTX x CTX) and a bool mask of the pixels far enough inside the photo to be trusted."""
    assert canvas.shape == (TILE, TILE) and valid.shape == canvas.shape, "canvas and valid must be (%d, %d)" % (TILE, TILE)
    fill = float(canvas[valid].mean()) if valid.any() else float(canvas.mean())
    padded = np.full((CTX, CTX), fill, dtype=np.float32)
    padded[CONTEXT_MARGIN:CONTEXT_MARGIN + TILE, CONTEXT_MARGIN:CONTEXT_MARGIN + TILE] = np.where(valid, canvas, fill)
    reliable = np.ones((CTX, CTX), dtype=bool)
    if USE_BORDER_MASK:
        inside = np.zeros((CTX, CTX), dtype=np.uint8)
        inside[CONTEXT_MARGIN:CONTEXT_MARGIN + TILE, CONTEXT_MARGIN:CONTEXT_MARGIN + TILE] = valid
        reliable = cv2.erode(inside, np.ones((9, 9), np.uint8)).astype(bool)
    return padded, reliable


@torch.no_grad()
def roma_pairs(roma_matcher, picture_a, picture_b):
    """Matched pixel positions (n, 2) in picture_a and picture_b (both CTX x CTX), drawn by certainty."""
    model = roma_matcher.roma_model
    torch.manual_seed(SEED)
    warp, certainty = model.match(as_rgb_picture(picture_a), as_rgb_picture(picture_b), device=DEVICE)
    matches, _ = model.sample(warp, certainty, num=roma_matcher.num_matches)
    points_a, points_b = model.to_pixel_coordinates(matches, CTX, CTX, CTX, CTX)
    return points_a.cpu().numpy().astype(np.float32), points_b.cpu().numpy().astype(np.float32)


def check_candidate_roma(roma_matcher, canvas_picture, reliable, level_images, level, x_m, y_m):
    assert level in level_images, "level %s was not built" % level
    assert math.isfinite(x_m) and math.isfinite(y_m) and x_m < 1e8, "candidate position must be finite metres (empty slots have x = 1e9)"
    tile = map_piece(level_images, level, x_m, y_m, CTX)
    points_a, points_b = roma_pairs(roma_matcher, canvas_picture, tile)
    columns = np.clip(np.round(points_a[:, 0]).astype(int), 0, CTX - 1)
    rows = np.clip(np.round(points_a[:, 1]).astype(int), 0, CTX - 1)
    keep = reliable[rows, columns]
    points_a, points_b = points_a[keep], points_b[keep]
    matrix, inliers = fit_transform(points_a, points_b)
    result = dict(n_pairs=len(points_a), n_inliers=int(inliers.sum()), shift=np.zeros(2), points_a=points_a, points_b=points_b, inliers=inliers, tile=tile)
    if matrix is not None:
        middle = (CTX - 1) / 2
        mapped = matrix @ np.array([middle, middle, 1.0])
        result["shift"] = (mapped - middle) * level_gsd(level)
    return result


def second_stage_roma(roma_matcher, canvas, valid, level, peaks, level_images):
    """Same output as second_stage (inliers, shifts), for any number of candidates."""
    assert peaks.ndim == 2 and peaks.shape[1] >= 3, "peaks must be (M, >= 3), got %s" % (peaks.shape,)
    canvas_picture, reliable = prepare_roma_canvas(canvas, valid)
    count = len(peaks)
    inliers = np.zeros(count)
    shifts = np.zeros((count, 2))
    limit = (MAX_SHIFT_CELLS * CELL + CONTEXT_MARGIN // 2) * level_gsd(level)
    for k in range(count):
        if peaks[k, 1] > 1e8:
            continue
        result = check_candidate_roma(roma_matcher, canvas_picture, reliable, level_images, level, peaks[k, 1], peaks[k, 2])
        inliers[k] = result["n_inliers"]
        if math.hypot(*result["shift"]) <= limit:
            shifts[k] = result["shift"]
        else:
            inliers[k] = 0
    return dict(inliers=inliers, shifts=shifts)
