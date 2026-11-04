"""Example replacement for fine_matching.ransac.fit_transform: plain OpenCV partial-affine RANSAC with a looser threshold.
Try:  python tests/pipeline/test_fine_matching.py --swap fine_matching.ransac.fit_transform=tests/pipeline/examples/my_fit.py:fit_transform
Same signature in, same outputs: (matrix 2x3 or None, inlier bool array).
It has no scale/angle gate, so test_ransac_gate_rejects_big_rotation is expected to FAIL: the tests tell you what your replacement is missing.
"""
import cv2
import numpy as np


def fit_transform(points_a, points_b):
    if len(points_a) < 4:
        return None, np.zeros(len(points_a), dtype=bool)
    matrix, inliers = cv2.estimateAffinePartial2D(points_a, points_b, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if matrix is None:
        return None, np.zeros(len(points_a), dtype=bool)
    return matrix, inliers.ravel().astype(bool)
