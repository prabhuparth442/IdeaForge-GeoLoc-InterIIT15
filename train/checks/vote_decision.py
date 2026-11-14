"""Block: re-make the answer-or-skip decision offline from a dump of every query's candidates (no GPU, no plotting).

load_photo        one photo's dump (photo_<i>.npz from Decision_Dump_Run) as a candidate table
choose_max        today's choice: candidate with the most  inliers + gamma * z  (0-inlier candidates never chosen)
query_support     how many queries put a candidate with >= min_inliers within radius of a place (their best one, or any one)
choose_vote       choice by number of agreeing queries first, summed inliers second
mean_position     inlier-weighted mean place of the queries that agree with the chosen place
"""
import math

import numpy as np


def load_photo(path):
    """Returns dict(truth, sigma, height, level, cand) where cand is a dict of arrays x, y, inliers, z, query, heading, scale (one row per candidate)."""
    data = np.load(path)
    rows = dict(x=[], y=[], inliers=[], z=[], query=[], heading=[], scale=[])
    q = 0
    while ("q%d_peaks" % q) in data:
        peaks = data["q%d_peaks" % q]
        shifts = data["q%d_shifts" % q]
        count = len(peaks)
        rows["x"].append(peaks[:, 1] + shifts[:, 0])
        rows["y"].append(peaks[:, 2] + shifts[:, 1])
        rows["inliers"].append(data["q%d_inliers" % q])
        rows["z"].append(data["q%d_z" % q])
        rows["query"].append(np.full(count, q))
        rows["heading"].append(np.full(count, float(data["q%d_heading" % q])))
        rows["scale"].append(np.full(count, float(data["q%d_scale" % q])))
        q += 1
    assert q > 0, "no queries in %s" % path
    cand = {key: np.concatenate(parts) for key, parts in rows.items()}
    return dict(truth=data["truth"], sigma=float(data["sigma"]), height=float(data["height"]), level=int(data["level"]), cand=cand, n_queries=q)


def choose_max(cand, gamma=1.0):
    """Index of the candidate with the most inliers + gamma * z among candidates with at least 1 inlier (-1 when none)."""
    score = cand["inliers"] + gamma * cand["z"]
    score = np.where(cand["inliers"] > 0, score, -np.inf)
    best = int(np.argmax(score))
    return best if np.isfinite(score[best]) else -1


def query_support(cand, x, y, radius, min_inliers=3, best_only=True):
    """Number of queries with a candidate of >= min_inliers inliers within `radius` metres of (x, y).
    best_only: only each query's own best candidate counts (as in gate_features.agree_queries)."""
    distance = np.hypot(cand["x"] - x, cand["y"] - y)
    support = 0
    for q in np.unique(cand["query"]):
        mine = cand["query"] == q
        if best_only:
            index = np.flatnonzero(mine)[int(np.argmax(cand["inliers"][mine]))]
            if cand["inliers"][index] >= min_inliers and distance[index] <= radius:
                support += 1
        elif np.any(mine & (cand["inliers"] >= min_inliers) & (distance <= radius)):
            support += 1
    return support


def choose_vote(cand, radius, min_inliers=3, best_only=False):
    """Index of the candidate whose place most queries agree on (ties: most inliers summed over the agreeing candidates, then own inliers). -1 when no
    candidate has min_inliers. Returns (index, support)."""
    eligible = np.flatnonzero(cand["inliers"] >= min_inliers)
    if len(eligible) == 0:
        return -1, 0
    best, best_key = -1, None
    for index in eligible:
        x, y = cand["x"][index], cand["y"][index]
        support = query_support(cand, x, y, radius, min_inliers, best_only)
        near = (np.hypot(cand["x"] - x, cand["y"] - y) <= radius) & (cand["inliers"] >= min_inliers)
        key = (support, float(cand["inliers"][near].sum()), float(cand["inliers"][index]))
        if best_key is None or key > best_key:
            best, best_key = int(index), key
    return best, best_key[0]


def mean_position(cand, index, radius, min_inliers=3):
    """Inlier-weighted mean place of each query's best candidate within `radius` of candidate `index` (each query counts once)."""
    x0, y0 = cand["x"][index], cand["y"][index]
    xs, ys, ws = [], [], []
    for q in np.unique(cand["query"]):
        mine = np.flatnonzero((cand["query"] == q) & (cand["inliers"] >= min_inliers) &
                              (np.hypot(cand["x"] - x0, cand["y"] - y0) <= radius))
        if len(mine) == 0:
            continue
        top = mine[int(np.argmax(cand["inliers"][mine]))]
        xs.append(cand["x"][top]); ys.append(cand["y"][top]); ws.append(cand["inliers"][top])
    if not ws:
        return float(x0), float(y0)
    ws = np.asarray(ws)
    return float(np.dot(xs, ws) / ws.sum()), float(np.dot(ys, ws) / ws.sum())


def error_sigma(x, y, truth, sigma):
    return math.hypot(x - truth[0], y - truth[1]) / sigma
