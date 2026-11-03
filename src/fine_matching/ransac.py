import math

import cv2
import numpy as np

from header.settings import RANSAC_PIXELS


def fit_transform(points_a, points_b):
    if len(points_a) < 4:
        return None, np.zeros(len(points_a), dtype=bool)
    matrix, inliers = cv2.estimateAffinePartial2D(points_a, points_b, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PIXELS, maxIters=3000, confidence=0.995)
    if matrix is None:
        return None, np.zeros(len(points_a), dtype=bool)
    scale = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    if abs(scale - 1) > 0.15 or abs(angle) > 12:
        return None, np.zeros(len(points_a), dtype=bool)
    return matrix, inliers.ravel().astype(bool)
