import numpy as np

from header.settings import ANSWER_MIN_INLIERS, MIN_INLIERS


def final_answer(peaks, stage2, min_inliers=None):
    """min_inliers: the least inliers the best candidate needs (None = settings.MIN_INLIERS)."""
    bar = MIN_INLIERS if min_inliers is None else min_inliers
    assert peaks.ndim == 2 and peaks.shape[1] >= 3, "peaks must be (n, >= 3): score, x, y, ..."
    assert len(stage2["inliers"]) == len(peaks) == len(stage2["shifts"]), "peaks, inliers and shifts must have one entry per candidate"
    assert stage2["shifts"].shape[1] == 2, "shifts must be (n, 2) metres"
    assert (stage2["inliers"] >= 0).all(), "inlier counts cannot be negative"
    order = np.lexsort((-peaks[:, 0], -stage2["inliers"]))
    best = int(order[0])
    if stage2["inliers"][best] < bar:
        return peaks[0, 1], peaks[0, 2], stage2["inliers"][best], 0
    return peaks[best, 1] + stage2["shifts"][best, 0], peaks[best, 2] + stage2["shifts"][best, 1], stage2["inliers"][best], best


def decide(inliers, min_inliers=None):
    """Answer when inliers >= min_inliers (None = settings.ANSWER_MIN_INLIERS)."""
    bar = ANSWER_MIN_INLIERS if min_inliers is None else min_inliers
    assert inliers >= 0, "inlier count cannot be negative, got %s" % inliers
    return bool(inliers >= bar)
