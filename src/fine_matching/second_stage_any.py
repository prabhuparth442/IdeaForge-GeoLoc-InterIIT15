"""The fine stage for any number of candidates M (second_stage itself is fixed to NUM_PEAKS = 25 rows).

run_fine_stage is the one entry point: it takes the matcher and the candidates and returns the same dict as second_stage
(inliers, shifts: one entry per candidate). A RoMa matcher (roma_matcher.py) is run candidate by candidate; the SuperPoint + LightGlue
matcher goes through second_stage in groups of NUM_PEAKS, the last group padded with empty rows (x = 1e9), which second_stage skips.
"""
import numpy as np

from fine_matching.second_stage import second_stage
from header.settings import NUM_PEAKS


def second_stage_any(matcher, canvas, valid, level, peaks, level_images):
    assert peaks.ndim == 2 and peaks.shape[1] >= 3 and len(peaks) >= 1, "peaks must be (M, >= 3) with M >= 1, got %s" % (peaks.shape,)
    count = len(peaks)
    inliers = np.zeros(count)
    shifts = np.zeros((count, 2))
    for start in range(0, count, NUM_PEAKS):
        part = peaks[start:start + NUM_PEAKS]
        padded = np.zeros((NUM_PEAKS, peaks.shape[1]))
        padded[:, 1] = 1e9                      # empty rows: second_stage skips candidates with x > 1e8
        padded[:, 2] = 1e9
        padded[:len(part)] = part
        found = second_stage(matcher, canvas, valid, level, padded, level_images)
        inliers[start:start + len(part)] = found["inliers"][:len(part)]
        shifts[start:start + len(part)] = found["shifts"][:len(part)]
    return dict(inliers=inliers, shifts=shifts)


def run_fine_stage(matcher, canvas, valid, level, peaks, level_images):
    """Fine stage for M candidates with either matcher."""
    if hasattr(matcher, "loftr_model"):
        from fine_matching.loftr_matcher import second_stage_loftr
        return second_stage_loftr(matcher, canvas, valid, level, peaks, level_images)
    if hasattr(matcher, "roma_model"):
        from fine_matching.roma_matcher import second_stage_roma
        return second_stage_roma(matcher, canvas, valid, level, peaks, level_images)
    return second_stage_any(matcher, canvas, valid, level, peaks, level_images)
