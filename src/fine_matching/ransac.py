import math

import cv2
import numpy as np

from header import settings


def fit_transform(points_a, points_b, pixels=None):
    """pixels: a match is an inlier within this distance of the fitted transform (None = settings.RANSAC_PIXELS)."""
    threshold = settings.RANSAC_PIXELS if pixels is None else pixels
    assert points_a.ndim == 2 and points_a.shape[1] == 2, "points_a must be (n, 2), got %s" % (points_a.shape,)
    assert points_a.shape == points_b.shape, "points_a and points_b must have the same shape (row k matches row k)"
    assert np.isfinite(points_a).all() and np.isfinite(points_b).all(), "points must be finite"
    if len(points_a) < 4:
        return None, np.zeros(len(points_a), dtype=bool)
    matrix, inliers = cv2.estimateAffinePartial2D(points_a, points_b, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=3000, confidence=0.995)
    if matrix is None:
        return None, np.zeros(len(points_a), dtype=bool)
    scale = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    if abs(scale - 1) > 0.15 or abs(angle) > 12:
        return None, np.zeros(len(points_a), dtype=bool)
    return matrix, inliers.ravel().astype(bool)
