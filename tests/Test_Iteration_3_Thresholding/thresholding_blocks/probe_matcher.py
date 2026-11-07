"""Runs the REAL fine matcher of the pipeline on any map tile and returns the diagnostics of probe_blocks.

Needs torch and the repo's src/ on sys.path (the notebook does that). It only READS from src/: it calls the same functions the pipeline
calls (prepare_canvas / prepare_roma_canvas, find_keypoints, match_keypoints, loftr_matches, map_piece) and never changes them.
Supported matchers: SuperPoint + LightGlue and LoFTR (MINIMA weights). RoMa is not supported here.
"""
import cv2
import numpy as np

from header import settings
from header.settings import CELL, CONTEXT_MARGIN, CTX, MAX_SHIFT_CELLS
from image_processing.map_levels import level_gsd, map_piece
from thresholding_blocks.probe_blocks import fit_with_reasons


def shift_limit_m(level):
    """The believable-shift limit of fine_matching.second_stage, in metres."""
    return (MAX_SHIFT_CELLS * CELL + CONTEXT_MARGIN // 2) * level_gsd(level)


def prepare_state(matcher, canvas, valid):
    """The canvas as the matcher sees it, prepared once per photo. canvas, valid: from render_query (TILE x TILE)."""
    if hasattr(matcher, "loftr_model"):
        from fine_matching.roma_matcher import prepare_roma_canvas
        picture, reliable = prepare_roma_canvas(canvas, valid)
        return dict(kind="loftr", picture=picture, reliable=reliable)
    assert hasattr(matcher, "glue"), "this probe supports the LightGlue and LoFTR matchers only"
    from fine_matching.second_stage import prepare_canvas
    features = prepare_canvas(matcher, canvas, valid)
    return dict(kind="lightglue", features=features)


def raw_matches(matcher, state, tile):
    """All matches the matcher proposes between the prepared canvas and one tile, before RANSAC.
    Returns points_a (n, 2) in the canvas, points_b (n, 2) in the tile, and a dict of counts."""
    if state["kind"] == "loftr":
        from fine_matching.loftr_matcher import loftr_matches
        points_a, points_b, confidence = loftr_matches(matcher, state["picture"], tile)
        keep = confidence >= matcher.min_confidence
        points_a, points_b = points_a[keep], points_b[keep]
        columns = np.clip(np.round(points_a[:, 0]).astype(int), 0, CTX - 1)
        rows = np.clip(np.round(points_a[:, 1]).astype(int), 0, CTX - 1)
        keep = state["reliable"][rows, columns]
        return points_a[keep], points_b[keep], dict(n_keypoints_canvas=-1, n_keypoints_tile=-1)
    from fine_matching.matcher import find_keypoints, match_keypoints
    tile_features = find_keypoints(matcher, tile)
    points_a, points_b, _ = match_keypoints(matcher, state["features"], tile_features, CTX)
    return points_a, points_b, dict(n_keypoints_canvas=int(state["features"]["keypoints"].shape[1]),
                                    n_keypoints_tile=int(tile_features["keypoints"].shape[1]))


def probe_positions(matcher, canvas, valid, level, level_images, positions, seed):
    """Run the matcher on the tile at each position. positions: list of (label, x_m, y_m) (x east, y south, map metres).
    Returns rows (one dict per position), tiles (label -> tile picture) and matches (label -> (points_a, points_b, inlier_mask))."""
    state = prepare_state(matcher, canvas, valid)
    gsd = level_gsd(level)
    limit = shift_limit_m(level)
    rows, tiles, matches = [], {}, {}
    for label, x_m, y_m in positions:
        tile = map_piece(level_images, level, x_m, y_m, CTX)
        points_a, points_b, counts = raw_matches(matcher, state, tile)
        cv2.setRNGSeed(seed)                       # the same RANSAC seed for every tile: repeatable
        fit = fit_with_reasons(points_a, points_b, settings.RANSAC_PIXELS, CTX, gsd, limit)
        mask = fit.pop("inlier_mask")
        row = dict(label=label, x_m=x_m, y_m=y_m)
        row.update(counts)
        row.update(fit)
        rows.append(row)
        tiles[label] = tile
        matches[label] = (points_a, points_b, mask)
    return rows, tiles, matches
