"""Optional rescue of a photo that would be answered no_match: count the LoFTR matches that sit at the same pixel in the photo and in the map window
(settings.USE_ALIGNED_RESCUE, off by default).

The coarse search already puts a candidate place within about one cell of the truth, so at a right place the photo and the map window are nearly aligned
and the weak matches (confidence down to ALIGNED_CONFIDENCE) cluster at the same pixel. At a wrong place they scatter. RANSAC cannot use them, it needs many
agreeing matches; this count needs only the few that sit still.

aligned_z   matches within ALIGNED_TOLERANCE_M metres of the same pixel, minus what a random scatter would give, divided by the square root of that plus one
pick_place  the candidate with the highest z, if the z is high enough and no place far away is nearly as good
Research repo, Idea 1: on the official photos that were no_match, +5.3 of 60 with 0 wrong answers; on three NUDT sets +0.4 to +0.5 with 0 wrong.
"""
import math

import numpy as np

import settings


def aligned_z(points_a, points_b, confidence, valid, gsd):
    """points_a: matches in the photo canvas, points_b: the same matches in the map window (pixels). valid: bool mask of canvas pixels the photo covers."""
    size = valid.shape[0]
    columns = np.clip(np.round(points_a[:, 0]).astype(int), 0, size - 1)
    rows = np.clip(np.round(points_a[:, 1]).astype(int), 0, size - 1)
    chosen = valid[rows, columns] & (confidence >= settings.ALIGNED_CONFIDENCE)
    count = int(chosen.sum())
    gap_m = np.hypot(points_b[chosen, 0] - points_a[chosen, 0], points_b[chosen, 1] - points_a[chosen, 1]) * gsd
    near = int((gap_m <= settings.ALIGNED_TOLERANCE_M).sum())
    area_share = math.pi * (settings.ALIGNED_TOLERANCE_M / gsd) ** 2 / (size * size)
    chance = count * min(1.0, area_share)
    return (near - chance) / math.sqrt(chance + 1.0)


def pick_place(peaks, shifts, aligned, sigma):
    """Returns (x, y) of the place to answer at, or None. peaks (n, 5), shifts (n, 2) in metres, aligned (n,) with -inf for places that were not matched."""
    if len(aligned) == 0:
        return None
    best = int(np.argmax(aligned))
    best_z = aligned[best]
    if not np.isfinite(best_z) or best_z < settings.ALIGNED_MIN_Z:
        return None
    x = peaks[best, 1] + shifts[best, 0]
    y = peaks[best, 2] + shifts[best, 1]
    for k in range(len(aligned)):
        if not np.isfinite(aligned[k]):
            continue
        other_x = peaks[k, 1] + shifts[k, 0]
        other_y = peaks[k, 2] + shifts[k, 1]
        far = math.hypot(other_x - x, other_y - y) / sigma > settings.ALIGNED_FAR_SIGMA
        if far and aligned[k] > best_z - settings.ALIGNED_MARGIN:
            return None
    return x, y
