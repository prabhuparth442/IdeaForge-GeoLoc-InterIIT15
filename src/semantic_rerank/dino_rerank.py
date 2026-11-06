"""Semantic layer between coarse_search and fine_matching: DINO re-orders the shortlist of the coarse stage (patch v0.5.5, idea F3 used as a judge).

The coarse stage gives a score map. Its best SHORTLIST (50) places are looked at again with DINO:
a 224 px map window is cut exactly at each peak, the query's valid mask is put on it, and DINO compares it with the query.
Query and window both start their token grid at their own corner, so the 14 px grid problem of a dense DINO map does not arise.
The shortlist is re-ordered by  z(coarse score) + weight * z(DINO score)  and the first NUM_PEAKS (25) go to fine_matching.second_stage
in the same (25, 5) layout as pick_peaks gives, so nothing after this layer changes.

Measured in tests/Test_Iteration_2 (379 official photos, coarse stage only): top-1 within 4 sigma goes from 0.897 to 0.966 for the fused
score and from 0.802 to 0.913 for edge ZNCC; no photo got worse. The ceiling (right place anywhere in the 25) does not change.
"""
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from coarse_search import dino_hint, search
from header.settings import DEVICE, NUM_PEAKS, TILE
from image_processing.map_levels import map_piece

SHORTLIST = 50          # peaks taken from the coarse score map
WEIGHT = 100.0          # 100 = DINO decides the order inside the shortlist; 0 = keep the coarse order
LAYER = 8               # DINO layer used, same as dino_hint.py
PATCH = 14
MISSING = 1e8           # pick_peaks writes x = 1e9 for a peak it could not find


def load_encoder():
    """The DINOv2 ViT-S/14 of dino_hint.py (one network can serve both)."""
    return dino_hint.load_encoder()


@contextmanager
def _peak_count(count):
    # pick_peaks reads NUM_PEAKS from the search module; it is set for one call only and always put back
    old = search.NUM_PEAKS
    search.NUM_PEAKS = count
    try:
        yield
    finally:
        search.NUM_PEAKS = old


def shortlist_peaks(score_map, geometry, count=SHORTLIST):
    """The first `count` peaks of a coarse score map (same rules as pick_peaks). Returns (count, 5)."""
    assert count >= 1, "the shortlist must hold at least one peak"
    with _peak_count(count):
        return search.pick_peaks(score_map, geometry)


@torch.no_grad()
def token_grids(encoder, pictures):
    """pictures (batch, 224, 224) floats 0..1. Returns DINO tokens (batch, 384, 16, 16)."""
    assert pictures.ndim == 3 and pictures.shape[1:] == (TILE, TILE), "pictures must be (batch, %d, %d)" % (TILE, TILE)
    x = torch.from_numpy(np.ascontiguousarray(pictures)).float().to(DEVICE).unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - dino_hint.IMAGE_MEAN) / dino_hint.IMAGE_STD
    if DEVICE == "cuda":
        x = x.half()
    tokens = encoder.get_intermediate_layers(x, n=[LAYER], norm=True)[0]
    side = TILE // PATCH
    tokens = tokens.float().reshape(len(pictures), side, side, -1)
    return tokens.permute(0, 3, 1, 2)


def valid_tokens(valid):
    """(16, 16) bool: token cells that lie fully inside the valid part of the query."""
    mask = torch.from_numpy(np.asarray(valid, dtype=np.float32))[None, None]
    share = F.avg_pool2d(mask, PATCH)[0, 0]
    return share > 0.99


def dino_candidate_scores(encoder, canvas, valid, level_images, level, peaks, batch=25):
    """Two numbers per shortlisted place (NaN where there is no peak):
    pooled  cosine of the mean token of the query and of the window (what kind of place)
    dense   mean over valid tokens of the token-to-token cosine (what lies where)"""
    assert canvas.shape == (TILE, TILE) and valid.shape == canvas.shape, "canvas and valid must be (%d, %d)" % (TILE, TILE)
    count = len(peaks)
    pooled = np.full(count, np.nan)
    dense = np.full(count, np.nan)
    real = []
    for i in range(count):
        if peaks[i, 1] < MISSING:
            real.append(i)
    if len(real) == 0:
        return pooled, dense

    mask = np.asarray(valid, dtype=np.float32)
    query_tokens = token_grids(encoder, (np.asarray(canvas, dtype=np.float32) * mask)[None])[0]
    inside = valid_tokens(valid).to(query_tokens.device)
    query_unit = F.normalize(query_tokens, dim=0)
    query_mean = F.normalize(query_tokens[:, inside].mean(dim=1), dim=0)

    for start in range(0, len(real), batch):
        chunk = real[start:start + batch]
        windows = []
        for i in chunk:
            windows.append(map_piece(level_images, level, peaks[i, 1], peaks[i, 2]) * mask)
        tokens = token_grids(encoder, np.stack(windows))
        unit = F.normalize(tokens, dim=1)
        per_token = (unit * query_unit[None]).sum(dim=1)
        means = F.normalize(tokens[:, :, inside].mean(dim=2), dim=1)
        for slot in range(len(chunk)):
            dense[chunk[slot]] = float(per_token[slot][inside].mean())
            pooled[chunk[slot]] = float((means[slot] * query_mean).sum())
    return pooled, dense


def _z(values):
    """Plain z-score over the finite values; NaN becomes 0."""
    result = np.zeros(len(values))
    good = np.isfinite(values)
    if good.sum() > 1:
        spread = values[good].std()
        if spread < 1e-9:
            spread = 1.0
        result[good] = (values[good] - values[good].mean()) / spread
    return result


def rerank_peaks(peaks, dino_score, weight=WEIGHT, keep=NUM_PEAKS):
    """Re-order the shortlist by z(peak score) + weight * z(dino_score). Returns the first `keep` rows, rank column (4) = combined score."""
    assert len(peaks) == len(dino_score), "one DINO score per peak"
    assert weight >= 0, "weight must be >= 0"
    real = peaks[:, 1] < MISSING
    first = np.where(real, peaks[:, 0], np.nan)
    judge = np.where(real, dino_score, np.nan)
    combined = _z(first) + weight * _z(judge)
    combined[~real] = -1e9
    order = np.argsort(-combined, kind="stable")[:keep]
    result = peaks[order].copy()
    result[:, 4] = combined[order]
    return result


def semantic_layer(encoder, score_map, geometry, canvas, valid, level, level_images, weight=WEIGHT, shortlist=SHORTLIST, keep=NUM_PEAKS):
    """The whole layer: coarse score map in, the best `keep` (M) of the `shortlist` (K) places out, in the layout of pick_peaks.
    K = how many places the coarse search returns, M = how many go on to the fine stage (M <= K).
    Also returns the DINO scores of the shortlist, for logging."""
    assert 1 <= keep <= shortlist, "need 1 <= keep (M) <= shortlist (K), got M=%d K=%d" % (keep, shortlist)
    short = shortlist_peaks(score_map, geometry, shortlist)
    pooled, dense = dino_candidate_scores(encoder, canvas, valid, level_images, level, short)
    return rerank_peaks(short, dense, weight, keep=keep), dict(shortlist=short, pooled=pooled, dense=dense)


def coarse_top(score_map, geometry, keep=NUM_PEAKS):
    """Without the layer: the first `keep` (M) peaks of the coarse score map, in coarse order."""
    return shortlist_peaks(score_map, geometry, keep)
