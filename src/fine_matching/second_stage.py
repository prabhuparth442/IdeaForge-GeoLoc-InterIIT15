import math

import cv2
import numpy as np
import torch

from fine_matching.matcher import find_keypoints, match_keypoints
from fine_matching.ransac import fit_transform
from header.settings import CELL, CONTEXT_MARGIN, CTX, DEVICE, MATCH_SIZE, MAX_SHIFT_CELLS, NUM_PEAKS, TILE, USE_BORDER_MASK
from image_processing.map_levels import level_gsd, map_piece


def prepare_canvas(matcher, canvas, valid):
    assert canvas.shape == (TILE, TILE) and valid.shape == canvas.shape, "canvas and valid must be (%d, %d)" % (TILE, TILE)
    assert valid.dtype == bool, "valid must be a bool array"
    fill = float(canvas[valid].mean()) if valid.any() else float(canvas.mean())
    padded = np.full((CTX, CTX), fill, dtype=np.float32)
    padded[CONTEXT_MARGIN:CONTEXT_MARGIN + TILE, CONTEXT_MARGIN:CONTEXT_MARGIN + TILE] = np.where(valid, canvas, fill)
    features = find_keypoints(matcher, padded)
    if USE_BORDER_MASK:
        reliable = np.zeros((CTX, CTX), dtype=np.uint8)
        reliable[CONTEXT_MARGIN:CONTEXT_MARGIN + TILE, CONTEXT_MARGIN:CONTEXT_MARGIN + TILE] = valid
        reliable = cv2.erode(reliable, np.ones((9, 9), np.uint8)).astype(bool)
        points = features["keypoints"][0].cpu().numpy() * (CTX / MATCH_SIZE)
        columns = np.clip(np.round(points[:, 0]).astype(int), 0, CTX - 1)
        rows = np.clip(np.round(points[:, 1]).astype(int), 0, CTX - 1)
        keep = torch.from_numpy(reliable[rows, columns]).to(DEVICE)
        for key in ("keypoints", "keypoint_scores", "descriptors"):
            features[key] = features[key][:, keep]
    return features


def check_candidate(matcher, canvas_features, level_images, level, x_m, y_m):
    assert level in level_images, "level %s was not built" % level
    assert math.isfinite(x_m) and math.isfinite(y_m) and x_m < 1e8, "candidate position must be finite metres (empty slots have x = 1e9)"
    tile = map_piece(level_images, level, x_m, y_m, CTX)
    tile_features = find_keypoints(matcher, tile)
    points_a, points_b, scores = match_keypoints(matcher, canvas_features, tile_features, CTX)
    matrix, inliers = fit_transform(points_a, points_b)
    result = dict(n_pairs=len(points_a), n_inliers=int(inliers.sum()), shift=np.zeros(2), points_a=points_a, points_b=points_b, inliers=inliers, tile=tile)
    if matrix is not None:
        middle = (CTX - 1) / 2
        mapped = matrix @ np.array([middle, middle, 1.0])
        result["shift"] = (mapped - middle) * level_gsd(level)
    return result


def second_stage(matcher, canvas, valid, level, peaks, level_images):
    assert peaks.ndim == 2 and peaks.shape[0] == NUM_PEAKS and peaks.shape[1] >= 3, "peaks must be (NUM_PEAKS, >= 3) from search_frame, got %s" % (peaks.shape,)
    assert level in level_images, "level %s was not built" % level
    canvas_features = prepare_canvas(matcher, canvas, valid)
    inliers = np.zeros(NUM_PEAKS)
    shifts = np.zeros((NUM_PEAKS, 2))
    limit = (MAX_SHIFT_CELLS * CELL + CONTEXT_MARGIN // 2) * level_gsd(level)
    for k in range(NUM_PEAKS):
        if peaks[k, 1] > 1e8:
            continue
        result = check_candidate(matcher, canvas_features, level_images, level, peaks[k, 1], peaks[k, 2])
        inliers[k] = result["n_inliers"]
        if math.hypot(*result["shift"]) <= limit:
            shifts[k] = result["shift"]
        else:
            inliers[k] = 0
    return dict(inliers=inliers, shifts=shifts)
