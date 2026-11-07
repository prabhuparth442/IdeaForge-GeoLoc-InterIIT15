"""Block: other ways to fit the match pairs than plain RANSAC, to compare them in the LoFTR fine stage.

Every function has the signature (points_a, points_b, pixels, confidence=None) -> (matrix, inliers), like fine_matching.ransac.fit_transform plus the LoFTR confidence of each pair (only the guided one uses it), with a 2 x 3 similarity
matrix (or None) and a bool array saying which pairs are inliers. The matrix is always a similarity (shift, rotation, scale), the same model as
the current fit. The methods that fit a homography (the MAGSAC family) only pick the pairs; the similarity is then fitted to those pairs, and the
inliers are counted again as "within `pixels` of that similarity", so all methods count the same way.

METHODS is the list used by the test notebook. Not here: NG-RANSAC and DSAC need a trained network, which does not exist for this data.
"""
import math

import cv2
import numpy as np
import pymagsac

from fine_matching.ransac import fit_transform

MAX_ITERS = 3000
CONFIDENCE = 0.995


def similarity_is_sensible(matrix):
    """Same sanity rule as fit_transform: scale within 15 % of 1, rotation within 12 degrees."""
    scale = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    return abs(scale - 1) <= 0.15 and abs(angle) <= 12


def inliers_of(matrix, points_a, points_b, pixels):
    mapped = points_a @ matrix[:, :2].T + matrix[:, 2]
    distance = np.hypot(mapped[:, 0] - points_b[:, 0], mapped[:, 1] - points_b[:, 1])
    return distance <= pixels


def nothing(count):
    return None, np.zeros(count, dtype=bool)


def finish_with_similarity(picked, points_a, points_b, pixels):
    """The pairs a method picked -> a similarity fitted to them (RANSAC inside the picked pairs) -> inliers counted by distance."""
    count = len(points_a)
    if picked.sum() < 4:
        return nothing(count)
    matrix, _ = cv2.estimateAffinePartial2D(points_a[picked], points_b[picked], method=cv2.RANSAC, ransacReprojThreshold=pixels, maxIters=MAX_ITERS, confidence=CONFIDENCE)
    if matrix is None or not similarity_is_sensible(matrix):
        return nothing(count)
    return matrix, inliers_of(matrix, points_a, points_b, pixels)


def fit_ransac(points_a, points_b, pixels, confidence=None):
    return fit_transform(points_a, points_b, pixels)


def fit_lmeds(points_a, points_b, pixels, confidence=None):
    count = len(points_a)
    if count < 4:
        return nothing(count)
    matrix, _ = cv2.estimateAffinePartial2D(points_a, points_b, method=cv2.LMEDS, maxIters=MAX_ITERS, confidence=CONFIDENCE)
    if matrix is None or not similarity_is_sensible(matrix):
        return nothing(count)
    return matrix, inliers_of(matrix, points_a, points_b, pixels)


def fit_magsac_pymagsac(points_a, points_b, pixels, confidence=None, plus_plus=False):
    count = len(points_a)
    if count < 4:
        return nothing(count)
    pairs = np.hstack([points_a, points_b]).astype(np.float64)
    size = 224.0
    _, picked = pymagsac.findHomography(pairs, size, size, size, size, np.ones(count), 4, plus_plus, pixels, 0.99, 50, 1000, 5)
    if picked is None:
        return nothing(count)
    return finish_with_similarity(np.asarray(picked, dtype=bool), points_a, points_b, pixels)


def fit_magsac_plus_plus_pymagsac(points_a, points_b, pixels, confidence=None):
    return fit_magsac_pymagsac(points_a, points_b, pixels, confidence, plus_plus=True)


def fit_magsac_opencv(points_a, points_b, pixels, confidence=None):
    count = len(points_a)
    if count < 4:
        return nothing(count)
    _, picked = cv2.findHomography(points_a, points_b, cv2.USAC_MAGSAC, pixels, maxIters=MAX_ITERS, confidence=CONFIDENCE)
    if picked is None:
        return nothing(count)
    return finish_with_similarity(picked.ravel().astype(bool), points_a, points_b, pixels)


def fit_confidence_guided(points_a, points_b, pixels, confidence=None, tries=500):
    """RANSAC that draws its two sample pairs with a chance proportional to the LoFTR confidence (the idea of NG-RANSAC / PROSAC, without a trained
    network). Needs `confidence` (n,). Two pairs fix a similarity exactly: b = m * a + t with m a complex number."""
    count = len(points_a)
    if count < 4:
        return nothing(count)
    chance = np.ones(count) / count
    if confidence is not None:
        chance = np.asarray(confidence, dtype=float) + 1e-6
        chance = chance / chance.sum()
    rng = np.random.default_rng(0)
    a = points_a[:, 0] + 1j * points_a[:, 1]
    b = points_b[:, 0] + 1j * points_b[:, 1]
    best_picked = None
    best_count = 0
    for _ in range(tries):
        first, second = rng.choice(count, size=2, replace=False, p=chance)
        if abs(a[second] - a[first]) < 1e-6:
            continue
        factor = (b[second] - b[first]) / (a[second] - a[first])
        if abs(abs(factor) - 1) > 0.15 or abs(math.degrees(np.angle(factor))) > 12:
            continue
        shift = b[first] - factor * a[first]
        picked = np.abs(factor * a + shift - b) <= pixels
        if picked.sum() > best_count:
            best_count = int(picked.sum())
            best_picked = picked
    if best_picked is None:
        return nothing(count)
    return finish_with_similarity(best_picked, points_a, points_b, pixels)


METHODS = {
    "RANSAC (current)": fit_ransac,
    "LMEDS": fit_lmeds,
    "MAGSAC (pymagsac)": fit_magsac_pymagsac,
    "MAGSAC++ (pymagsac)": fit_magsac_plus_plus_pymagsac,
    "MAGSAC (OpenCV USAC)": fit_magsac_opencv,
    "confidence-guided RANSAC": fit_confidence_guided,
}
