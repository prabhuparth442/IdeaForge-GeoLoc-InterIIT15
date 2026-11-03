import numpy as np

from header.settings import ANSWER_MIN_INLIERS, MIN_INLIERS


def final_answer(peaks, stage2):
    order = np.lexsort((-peaks[:, 0], -stage2["inliers"]))
    best = int(order[0])
    if stage2["inliers"][best] < MIN_INLIERS:
        return peaks[0, 1], peaks[0, 2], stage2["inliers"][best], 0
    return peaks[best, 1] + stage2["shifts"][best, 0], peaks[best, 2] + stage2["shifts"][best, 1], stage2["inliers"][best], best


def decide(inliers):
    return bool(inliers >= ANSWER_MIN_INLIERS)
