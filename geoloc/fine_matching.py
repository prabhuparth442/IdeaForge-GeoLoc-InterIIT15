"""Fine matching: LoFTR (kornia code, MINIMA weights) matches the photo canvas with a map window around each candidate place,
RANSAC keeps the matches that agree on one shift/rotation/scale. The number of agreeing matches (inliers) says how good the place is,
and the fitted shift corrects the position by a few metres.
"""
import math
from contextlib import contextmanager, nullcontext

import cv2
import numpy as np
import torch
from torch import nn

import settings
from geoloc import aligned_count
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


class QueryBackbone(nn.Module):
    """LoFTR runs its CNN on the photo canvas and on the map window in one batch. Within one query the canvas is always the same picture, so its CNN
    features are computed once and reused for every candidate place. Same numbers as without (checked in tests/test_query_reuse.py), fewer network passes.
    Only for eval mode and a batch whose first half is the canvas repeated."""

    def __init__(self, original):
        super().__init__()
        assert not original.training, "query reuse needs eval mode"
        self.original = original
        self.query = None
        self.saved = None

    def forward(self, x):
        assert len(x) % 2 == 0
        n = len(x) // 2
        query = x[:n]
        assert torch.equal(query, query[:1].expand_as(query)), "mixed queries in one cached batch"
        if self.query is None:
            self.query = query[:1].detach().clone()
            self.saved = self.original(query[:1])
        else:
            assert torch.equal(query[:1], self.query), "the cache was reused for a different canvas"
        candidates = self.original(x[n:])
        return tuple(torch.cat((saved.expand(n, *saved.shape[1:]), new), dim=0) for saved, new in zip(self.saved, candidates))


@contextmanager
def cached_query_backbone(model):
    """Inside the block the model computes the canvas features once. One block per canvas (one query)."""
    original = model.backbone
    model.backbone = QueryBackbone(original)
    try:
        yield
    finally:
        model.backbone = original


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


def tile_centre_offset(x_m, y_m, gsd, size):
    """(east, south) metres from the candidate place to the true centre of its map tile. map_piece puts the tile corner on a whole map pixel, so the centre is off by up to half a pixel."""
    top = int(round(y_m / gsd - size / 2))
    left = int(round(x_m / gsd - size / 2))
    return np.array([(left + size / 2) * gsd - x_m, (top + size / 2) * gsd - y_m])


def check_places(model, canvas, valid, level, peaks, level_image, tile_filter=None, limit_level=None):
    """Fine stage for every candidate row of peaks. Returns inliers (M,) and shifts (M, 2) in metres.
    A candidate whose correction is longer than MAX_SHIFT_CELLS cells counts as failed (0 inliers).
    tile_filter: None, or a function that turns each map tile into the tile LoFTR should get (native-resolution path).
    limit_level: the level whose cell size sets the shift limit (None = this level); the native path passes the coarse level so the limit in metres stays the same."""
    canvas_picture = grey_canvas(canvas, valid)
    size = canvas_picture.shape[0]
    middle = (size - 1) / 2
    gsd = level_gsd(level)
    limit = settings.MAX_SHIFT_CELLS * settings.CELL * level_gsd(level if limit_level is None else limit_level)

    wanted = [k for k in range(len(peaks)) if peaks[k, 1] <= 1e8]
    tiles = {}
    for k in wanted:
        tiles[k] = map_piece(level_image, level, peaks[k, 1], peaks[k, 2], size)
        if tile_filter is not None:
            tiles[k] = tile_filter(tiles[k])

    # LOFTR_BATCH windows go through the network together (1 = one at a time)
    matches = {}
    step = settings.LOFTR_BATCH
    can_reuse = getattr(model, "reuse_query", True)            # False for XoFTR (no shared CNN, no batched call)
    if not can_reuse:
        step = 1
    reuse = cached_query_backbone(model) if (settings.LOFTR_REUSE_QUERY and can_reuse) else nullcontext()
    with reuse:
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
    aligned = np.full(len(peaks), -np.inf)       # only filled when settings.USE_ALIGNED_RESCUE is on
    for k in wanted:
        points_a, points_b, confidence = matches[k]
        if settings.USE_ALIGNED_RESCUE:
            aligned[k] = aligned_count.aligned_z(points_a, points_b, confidence, valid, gsd)
        keep = confidence >= getattr(model, "min_confidence", settings.LOFTR_CONFIDENCE)
        matrix, picked = fit_similarity(points_a[keep], points_b[keep])
        if matrix is None:
            continue
        mapped = matrix @ np.array([middle, middle, 1.0])
        shift = (mapped - middle) * gsd
        if math.hypot(shift[0], shift[1]) <= limit:
            inliers[k] = int(picked.sum())
            shifts[k] = shift
            if settings.TILE_CENTRE_FIX:
                shifts[k] = shift + tile_centre_offset(peaks[k, 1], peaks[k, 2], gsd, size)
    return dict(inliers=inliers, shifts=shifts, aligned=aligned)
