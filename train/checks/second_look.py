"""Block: a second, better-aimed LoFTR look at one candidate (needs the GPU and a set-up pipeline). No plotting.

The idea (more lines for right places): the first look matches a canvas drawn with a guessed heading and scale against a map tile centred on a
coarse peak, so a right place is seen off-centre (up to 12 px), slightly rotated and slightly scaled. LoFTR is not rotation- or scale-invariant
and loses lines at the tile border, so a right place yields fewer lines than it could. The first fit tells how far off the guess was. The second
look re-draws the canvas with the corrected heading and scale and centres the map tile on the corrected position; the pair is then near-identical
for a right place, which should give LoFTR many more lines. For a wrong place the first fit is chance, so the correction aims nowhere and the
second look should not gain (or should lose) lines. The ratio second / first is then a signal in itself.

signed_fit(match, min_confidence)                 the pipeline's similarity fit of stored matches: (inliers, angle_deg signed, scale, shift_px (dx, dy))
look_again(pipeline, item, level, heading, scale_factor, x_m, y_m, angle_sign, angle_deg, fit_scale)
    render the corrected query, match it against the tile at (x_m, y_m) (already shifted), and return the signals of
    train/checks/decision_signals.candidate_signals plus the pipeline's inliers and the new position (x, y after the new shift)
"""
import math

import cv2
import numpy as np

from coarse_search.heading_queries import query_geometry
from fine_matching.loftr_matcher import candidate_matches, stage2_from_matches
from fine_matching.ransac import fit_transform
from header.settings import RANSAC_PIXELS, SEED
from image_processing.geometry import render_query
from train.checks.decision_signals import candidate_signals

CONF = 0.22


def signed_fit(match, min_confidence=CONF):
    """(inliers, signed angle in degrees, scale, (dx, dy) of the canvas centre in px) of the stored matches; zeros when there is no fit."""
    if match is None:
        return 0, 0.0, 1.0, (0.0, 0.0)
    keep = match["confidence"] >= min_confidence
    a, b = match["points_a"][keep].astype(np.float32), match["points_b"][keep].astype(np.float32)
    if len(a) < 4:
        return 0, 0.0, 1.0, (0.0, 0.0)
    cv2.setRNGSeed(SEED)
    matrix, inl = fit_transform(a, b, RANSAC_PIXELS)
    if matrix is None:
        return 0, 0.0, 1.0, (0.0, 0.0)
    middle = 111.5
    centre = matrix @ np.array([middle, middle, 1.0]) - middle
    return int(inl.sum()), math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])), math.hypot(matrix[0, 0], matrix[1, 0]), (float(centre[0]), float(centre[1]))


def look_again(pipeline, item, level, heading, scale_factor, x_m, y_m, angle_sign=0, angle_deg=0.0, fit_scale=1.0, scale_power=0):
    """angle_sign 0 / scale_power 0: same drawing as the first look (only the tile is re-centred). angle_sign +-1: heading moved by
    +-angle_deg. scale_power +-1: scale factor multiplied by fit_scale ** scale_power."""
    geometry = query_geometry(heading + angle_sign * angle_deg, scale_factor * fit_scale ** scale_power)
    canvas, valid = render_query(item, pipeline.camera, level, geometry)
    peaks = np.array([[1.0, x_m, y_m, 0.0, 0.0]])
    stored = candidate_matches(pipeline.matcher, canvas, valid, level, peaks, pipeline.level_images)
    cv2.setRNGSeed(SEED)
    stage2 = stage2_from_matches(stored, level, pipeline.matcher.min_confidence)
    match = stored["items"][0]
    out = candidate_signals(match, stored["inside"], with_lo=False)
    out.update(pl_inliers=float(stage2["inliers"][0]), x=float(x_m + stage2["shifts"][0, 0]), y=float(y_m + stage2["shifts"][0, 1]),
               valid_fraction=float(valid.mean()))
    return out
