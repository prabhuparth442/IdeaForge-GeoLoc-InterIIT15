"""Block: features of a chosen candidate for a learned answer-or-skip verifier, from a rich dump (no GPU, no plotting).

candidate_fit_features   re-fit one candidate's stored matches (same cut and RANSAC as the pipeline) and describe the fit: pairs, inliers, inlier
                         ratio, fitted scale and angle, inlier residual, spread and convex-hull share of the inliers, mean inlier confidence, shift
decision_rows            one row per photo and decision step (standard 6 / wide 27): the chosen candidate's features + photo features + truth label

Why these features (each has a reason to separate right from wrong independently of the raw inlier count, which does not transfer between data sets):
- a right match covers the photo (large hull, wide spread); a chance match on a look-alike building is local (small hull)
- the query already applies the heading and scale guess, so a right fit has scale near 1 and angle near 0; a chance fit is anywhere in the gate
- a right fit is tight (small residual) and uses most of the confident pairs (high ratio)
- agreement between queries that see different pictures (other headings / scales)
"""
import math

import cv2
import numpy as np

from fine_matching.ransac import fit_transform
from header.settings import RANSAC_PIXELS, SEED
from train.checks.replay import STANDARD, WIDE, agreement, choose, error_sigma, pooled, query_ids

MIN_CONFIDENCE = 0.22


def candidate_fit_features(match, canvas_size, min_confidence=MIN_CONFIDENCE, pixels=RANSAC_PIXELS):
    """match: dict(points_a, points_b, confidence) or None. canvas_size: side of the LoFTR canvas in px."""
    empty = dict(n_pairs=0, n_inl=0, ratio=0.0, fit_scale_dev=0.15, fit_angle=12.0, resid=pixels, hull=0.0, spread=0.0, conf_inl=0.0)
    if match is None:
        return empty
    keep = match["confidence"] >= min_confidence
    a, b, c = match["points_a"][keep], match["points_b"][keep], match["confidence"][keep]
    out = dict(empty, n_pairs=int(len(a)))
    if len(a) < 4:
        return out
    cv2.setRNGSeed(SEED)
    matrix, inl = fit_transform(a.astype(np.float32), b.astype(np.float32), pixels)
    if matrix is None or inl.sum() < 3:
        return out
    ai, bi = a[inl], b[inl]
    mapped = ai @ matrix[:, :2].T + matrix[:, 2]
    hull = cv2.convexHull(ai.astype(np.float32))
    out.update(n_inl=int(inl.sum()), ratio=float(inl.mean()),
               fit_scale_dev=abs(math.hypot(matrix[0, 0], matrix[1, 0]) - 1.0),
               fit_angle=abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))),
               resid=float(np.sqrt(np.mean(np.sum((mapped - bi) ** 2, axis=1)))),
               hull=float(cv2.contourArea(hull)) / float(canvas_size ** 2),
               spread=float(np.sqrt(np.mean(np.var(ai, axis=0)))) / canvas_size,
               conf_inl=float(c[inl].mean()))
    return out


def _step_row(photo, ids, gamma=1.0):
    cand = pooled(photo, ids)
    k = choose(cand, gamma)
    row = dict(n_queries=len(ids))
    if k < 0:
        row.update(has_choice=False, inliers=0.0, z=0.0, rank=-1, err_sigma=float("nan"), agree=0, agree_any=0, runner_far=0.0,
                   q_heading=0.0, q_scale=1.0, **candidate_fit_features(None, 1))
        return row
    owner, rank = int(cand["owner"][k]), int(cand["rank"][k])
    q = photo["queries"][owner]
    x, y = cand["x"][k], cand["y"][k]
    sigma = photo["sigma"]
    far = np.hypot(cand["x"] - x, cand["y"] - y) > 4 * sigma
    row.update(has_choice=True, inliers=float(cand["inliers"][k]), z=float(cand["z"][k]), rank=rank, coarse=float(cand["coarse"][k]),
               err_sigma=error_sigma(photo, x, y),
               agree=agreement(photo, ids, x, y, sigma),
               agree_any=int(sum(np.any((cand["owner"] == i) & (cand["inliers"] >= 3) & ~far & (np.hypot(cand["x"] - x, cand["y"] - y) <= sigma)) for i in ids)),
               runner_far=float(cand["inliers"][far].max()) if far.any() else 0.0,
               q_heading=float(q["heading"]), q_scale=float(q["scale"]), valid_fraction=float(q["valid_fraction"]),
               **candidate_fit_features(q["matches"][rank], q["inside"].shape[0]))
    return row


def decision_rows(photo, photo_id, data_name):
    """Two rows: the standard-6 decision and the wide-27 decision for this photo."""
    base = dict(photo=photo_id, data=data_name, height=photo["height"], level=photo["level"], sigma=photo["sigma"],
                pitch=photo["pitch"], roll=photo["roll"])
    rows = []
    for step, keys in (("std6", STANDARD), ("wide27", WIDE)):
        row = dict(base, step=step)
        row.update(_step_row(photo, query_ids(photo, keys)))
        rows.append(row)
    return rows
