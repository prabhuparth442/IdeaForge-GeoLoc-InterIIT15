"""A keypoint matcher built on OpenCV SIFT, with the same two function signatures as fine_matching.matcher.
Use it to run second_stage and localise_photo on a computer without lightglue or the MINIMA weights.
It works on the synthetic world (same sensor on both sides); it will NOT bridge thermal and visible images.
"""
import cv2
import numpy as np


def find_keypoints(matcher, picture):
    assert picture.ndim == 2 and picture.shape[0] == picture.shape[1], "picture must be square and 2-D"
    big = cv2.resize((np.clip(picture, 0, 1) * 255).astype(np.uint8), (448, 448), interpolation=cv2.INTER_CUBIC)
    sift = cv2.SIFT_create(nfeatures=1500)
    keypoints, descriptors = sift.detectAndCompute(big, None)
    points = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2)
    return dict(points=points, descriptors=descriptors)


def match_keypoints(matcher, features_a, features_b, picture_size):
    empty = np.zeros((0, 2), np.float32)
    if features_a["descriptors"] is None or features_b["descriptors"] is None or len(features_a["points"]) < 2 or len(features_b["points"]) < 2:
        return empty, empty, np.zeros(0)
    pairs = cv2.BFMatcher().knnMatch(features_a["descriptors"], features_b["descriptors"], k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    scale = picture_size / 448
    points_a = np.array([features_a["points"][m.queryIdx] for m in good], dtype=np.float32).reshape(-1, 2) * scale
    points_b = np.array([features_b["points"][m.trainIdx] for m in good], dtype=np.float32).reshape(-1, 2) * scale
    return points_a, points_b, np.ones(len(good))
