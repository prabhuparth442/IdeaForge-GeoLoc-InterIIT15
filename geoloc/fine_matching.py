"""Fine matching: LoFTR (kornia code, MINIMA weights) matches the photo canvas with a map window around each candidate place,
RANSAC keeps the matches that agree on one shift/rotation/scale. The number of agreeing matches (inliers) says how good the place is,
and the fitted shift corrects the position by a few metres.
"""
import math

import cv2
import numpy as np
import torch

import settings
from geoloc.coarse_search import DEVICE
from geoloc.levels import level_gsd, map_piece


def load_loftr(weights_file):
    """kornia LoFTR with the MINIMA weights, read from the local file (nothing is downloaded)."""
    from kornia.feature import LoFTR

    saved = torch.load(weights_file, map_location="cpu", weights_only=False)      # our own checkpoint file
    weights = {}
    for name, value in saved["state_dict"].items():
        if name.startswith("matcher."):
            weights[name[len("matcher."):]] = value
    if len(weights) == 0:
        raise ValueError("no 'matcher.' weights in " + weights_file)
    model = LoFTR(pretrained=None)
    model.load_state_dict(weights, strict=True)
    model.coarse_matching.thr = settings.LOFTR_COARSE_THRESHOLD
    return model.to(DEVICE).eval()


def as_tensor(gray):
    return torch.from_numpy(np.clip(np.asarray(gray, dtype=np.float32), 0.0, 1.0))[None, None].to(DEVICE)


@torch.no_grad()
def loftr_matches(model, canvas_picture, tile):
    """Matched positions in the canvas (n, 2), in the map window (n, 2) and the confidence of each match (n,)."""
    found = model({"image0": as_tensor(canvas_picture), "image1": as_tensor(tile)})
    return found["keypoints0"].cpu().numpy(), found["keypoints1"].cpu().numpy(), found["confidence"].cpu().numpy()


@torch.no_grad()
def loftr_matches_batch(model, canvas_picture, tiles):
    """The same for several map windows at once. Returns one (points_a, points_b, confidence) per tile."""
    count = len(tiles)
    first = as_tensor(canvas_picture).repeat(count, 1, 1, 1)
    second = torch.cat([as_tensor(tile) for tile in tiles], dim=0)
    found = model({"image0": first, "image1": second})
    which = found["batch_indexes"].cpu().numpy()
    points_a = found["keypoints0"].cpu().numpy()
    points_b = found["keypoints1"].cpu().numpy()
    confidence = found["confidence"].cpu().numpy()
    parts = []
    for i in range(count):
        mine = which == i
        parts.append((points_a[mine], points_b[mine], confidence[mine]))
    return parts


def fit_similarity(points_a, points_b):
    """RANSAC fit of a shift + rotation + scale. Returns the 2x3 matrix (or None) and which matches are inliers."""
    if len(points_a) < 4:
        return None, np.zeros(len(points_a), dtype=bool)
    matrix, inliers = cv2.estimateAffinePartial2D(points_a, points_b, method=cv2.RANSAC, ransacReprojThreshold=settings.RANSAC_PIXELS,
                                                  maxIters=settings.RANSAC_MAX_ITERS, confidence=0.995)
    if matrix is None:
        return None, np.zeros(len(points_a), dtype=bool)
    scale = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    if abs(scale - 1) > settings.RANSAC_MAX_SCALE_CHANGE or abs(angle) > settings.RANSAC_MAX_ROTATION:
        return None, np.zeros(len(points_a), dtype=bool)
    return matrix, inliers.ravel().astype(bool)


def grey_canvas(canvas, valid):
    # the part of the canvas the photo does not cover is filled with the photo's own mean grey
    fill = float(canvas[valid].mean()) if valid.any() else float(canvas.mean())
    return np.where(valid, canvas, fill).astype(np.float32)


def check_places(model, canvas, valid, level, peaks, level_image):
    """Fine stage for every candidate row of peaks. Returns inliers (M,) and shifts (M, 2) in metres.
    A candidate whose correction is longer than MAX_SHIFT_CELLS cells counts as failed (0 inliers)."""
    canvas_picture = grey_canvas(canvas, valid)
    size = canvas_picture.shape[0]
    middle = (size - 1) / 2
    gsd = level_gsd(level)
    limit = settings.MAX_SHIFT_CELLS * settings.CELL * gsd

    wanted = [k for k in range(len(peaks)) if peaks[k, 1] <= 1e8]
    tiles = {}
    for k in wanted:
        tiles[k] = map_piece(level_image, level, peaks[k, 1], peaks[k, 2], size)

    # LOFTR_BATCH windows go through the network together (1 = one at a time)
    matches = {}
    step = settings.LOFTR_BATCH
    for start in range(0, len(wanted), step):
        group = wanted[start:start + step]
        if step == 1:
            matches[group[0]] = loftr_matches(model, canvas_picture, tiles[group[0]])
        else:
            parts = loftr_matches_batch(model, canvas_picture, [tiles[k] for k in group])
            for k, part in zip(group, parts):
                matches[k] = part

    inliers = np.zeros(len(peaks))
    shifts = np.zeros((len(peaks), 2))
    for k in wanted:
        points_a, points_b, confidence = matches[k]
        keep = confidence >= settings.LOFTR_CONFIDENCE
        matrix, picked = fit_similarity(points_a[keep], points_b[keep])
        if matrix is None:
            continue
        mapped = matrix @ np.array([middle, middle, 1.0])
        shift = (mapped - middle) * gsd
        if math.hypot(shift[0], shift[1]) <= limit:
            inliers[k] = int(picked.sum())
            shifts[k] = shift
    return dict(inliers=inliers, shifts=shifts)
