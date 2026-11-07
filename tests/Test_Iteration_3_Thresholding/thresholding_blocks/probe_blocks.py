"""Pure blocks for Step 1 (numpy and OpenCV only, no torch, no dataset): they can be tested anywhere.

fit_with_reasons : the same RANSAC fit as src/fine_matching/ransac.fit_transform, but it also says WHY a candidate ended with
                   0 inliers (too few pairs / RANSAC failed / scale-angle gate / shift gate) and keeps the raw inlier count.
classify_cause   : one plain label per photo: why did the candidate sent to the fine stage get so few inliers?
offset_label, offset_positions : the tile centres around the true position, used to see how fast inliers fall with distance.
This file prints nothing, plots nothing and never changes src/.
"""
import math

import cv2
import numpy as np

SCALE_TOLERANCE = 0.15      # the same limits as src/fine_matching/ransac.py
ANGLE_LIMIT_DEG = 12.0
MIN_PAIRS = 4


def empty_fit(n_pairs):
    return dict(n_pairs=int(n_pairs), raw_inliers=0, inliers=0, inlier_ratio=0.0, scale=float("nan"), angle_deg=float("nan"),
                shift_east_m=0.0, shift_south_m=0.0, shift_m=0.0, residual_rms_px=float("nan"), spread=0.0,
                reason="too_few_pairs", inlier_mask=np.zeros(int(n_pairs), dtype=bool))


def fit_with_reasons(points_a, points_b, ransac_pixels, canvas_size, metres_per_pixel, shift_limit_m):
    """points_a, points_b: (n, 2) matched pixel positions in the canvas and in the map tile.
    ransac_pixels: inlier distance in pixels (settings.RANSAC_PIXELS). canvas_size: side of the picture in pixels (CTX).
    metres_per_pixel: level_gsd(level). shift_limit_m: the believable-shift limit of second_stage, in metres.
    Returns a dict. 'raw_inliers' = RANSAC count before the gates; 'inliers' = what the pipeline would report (0 when a gate rejects the fit).
    'reason' is one of too_few_pairs, ransac_failed, scale_angle_gate, shift_gate, ok.
    shift_east_m / shift_south_m = how far the canvas centre moved on the tile (signed, metres)."""
    assert points_a.ndim == 2 and points_a.shape[1] == 2, "points_a must be (n, 2), got %s" % (points_a.shape,)
    assert points_a.shape == points_b.shape, "points_a and points_b must have the same shape"
    assert ransac_pixels > 0 and canvas_size > 0 and metres_per_pixel > 0 and shift_limit_m > 0, "pixels, size, metres per pixel and limit must be positive"
    assert np.isfinite(points_a).all() and np.isfinite(points_b).all(), "points must be finite"
    result = empty_fit(len(points_a))
    if len(points_a) < MIN_PAIRS:
        return result
    matrix, mask = cv2.estimateAffinePartial2D(points_a.astype(np.float32), points_b.astype(np.float32), method=cv2.RANSAC,
                                               ransacReprojThreshold=ransac_pixels, maxIters=3000, confidence=0.995)
    if matrix is None:
        result["reason"] = "ransac_failed"
        return result
    mask = mask.ravel().astype(bool)
    scale = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    middle = (canvas_size - 1) / 2
    mapped = matrix @ np.array([middle, middle, 1.0])
    east = (mapped[0] - middle) * metres_per_pixel
    south = (mapped[1] - middle) * metres_per_pixel
    result.update(raw_inliers=int(mask.sum()), inlier_ratio=float(mask.sum()) / len(points_a), scale=scale, angle_deg=angle,
                  shift_east_m=east, shift_south_m=south, shift_m=math.hypot(east, south), inlier_mask=mask)
    if mask.sum() > 0:
        moved = points_a[mask] @ matrix[:, :2].T + matrix[:, 2]
        result["residual_rms_px"] = float(np.sqrt(np.mean(np.sum((moved - points_b[mask]) ** 2, axis=1))))
    if mask.sum() >= 3:
        hull = cv2.convexHull(points_a[mask].astype(np.float32))
        result["spread"] = float(cv2.contourArea(hull)) / float(canvas_size * canvas_size)
    if abs(scale - 1) > SCALE_TOLERANCE or abs(angle) > ANGLE_LIMIT_DEG:
        result["reason"] = "scale_angle_gate"
    elif result["shift_m"] > shift_limit_m:
        result["reason"] = "shift_gate"
    else:
        result["reason"] = "ok"
        result["inliers"] = int(mask.sum())
    return result


def classify_cause(oracle_raw_inliers, candidate_raw_inliers, candidate_reason, candidate_offset_px, good_count=8, near_px=12.0):
    """One label for one photo. oracle_* = the tile centred on the true position; candidate_* = the candidate the pipeline chose.
    good_count: raw inliers that count as a real match. near_px: a candidate closer than this to the truth counts as 'near'."""
    assert good_count >= 1 and near_px >= 0, "good_count must be >= 1 and near_px >= 0"
    if candidate_raw_inliers >= good_count and candidate_reason != "ok":
        return "gate_removed_a_good_fit"
    if candidate_raw_inliers >= good_count:
        return "candidate_is_fine"
    if oracle_raw_inliers < good_count:
        return "weak_even_at_truth"
    if candidate_offset_px > near_px:
        return "candidate_too_far_from_truth"
    return "weak_at_candidate_near_truth"


def offset_label(dx, dy):
    """The name of a tile centred dx, dy map pixels from the truth. (0, 0) is called 'truth'."""
    return "truth" if (dx, dy) == (0, 0) else "truth%+d,%+d" % (dx, dy)


def offset_positions(true_x_m, true_y_m, metres_per_pixel, offsets_px):
    """Tile centres around the truth: a list of (label, x_m, y_m). offsets_px: list of (dx, dy) in map pixels (x east, y south)."""
    assert metres_per_pixel > 0, "metres_per_pixel must be positive"
    assert len(offsets_px) > 0 and tuple(offsets_px[0]) == (0, 0), "offsets_px must start with (0, 0), the tile at the truth"
    return [(offset_label(dx, dy), true_x_m + dx * metres_per_pixel, true_y_m + dy * metres_per_pixel) for dx, dy in offsets_px]
