"""Decision-only prototypes. Selectors do not read truth/error/point columns."""
import math

import cv2
import numpy as np

from fine_matching.ransac import fit_transform
from train.checks.candidate_judge import STANDARD


def heldout_geometry(match, inside):
    if match is None:
        return False
    keep = match["confidence"] >= 0.22
    first, second = match["points_a"][keep], match["points_b"][keep]
    if len(first) < 8:
        return False
    indices = np.rint(first).astype(int)
    size = inside.shape[0]
    indices = np.clip(indices, 0, size - 1)
    valid = inside[indices[:, 1], indices[:, 0]]
    first, second = first[valid], second[valid]
    cells = np.floor(first / (size / 4)).astype(int)
    split = (cells.sum(axis=1) % 2) == 0
    centres = []
    middle = (size - 1) / 2
    for training in (split, ~split):
        if training.sum() < 4 or (~training).sum() < 4:
            return False
        cv2.setRNGSeed(7)
        matrix, inliers = fit_transform(first[training], second[training], pixels=3.0)
        if matrix is None or inliers.sum() < 4:
            return False
        predicted = first[~training] @ matrix[:, :2].T + matrix[:, 2]
        count = (np.linalg.norm(predicted - second[~training], axis=1) <= 3).sum()
        if count < 4:
            return False
        centre = matrix @ np.array([middle, middle, 1]) - middle
        if np.linalg.norm(centre) > 12:
            return False
        centres.append(centre)
    return bool(np.linalg.norm(centres[0] - centres[1]) <= 3)


def choose(table, policy, threshold):
    candidates = table[table["inliers"] >= 4]
    if candidates.empty:
        return None
    if policy.startswith("unique"):
        order = np.lexsort((-candidates["coarse"].to_numpy(),
                           -(candidates["inliers"] + candidates["z"]).to_numpy(),
                           -candidates["p"].to_numpy()))
    else:
        order = np.lexsort((-candidates["coarse"].to_numpy(),
                           -(candidates["inliers"] + candidates["z"]).to_numpy()))
    winner = candidates.iloc[int(order[0])]
    if winner["p"] < threshold:
        return None
    if policy.startswith("unique"):
        distance = np.hypot(candidates["x"] - winner["x"], candidates["y"] - winner["y"])
        rivals = (candidates["p"] >= 0.9) & (distance > 2 * winner["sigma"])
        if rivals.any():
            return None
    return winner


def decide_photo(table, photo, policy):
    standard = set(STANDARD)
    std = table[[key in standard for key in zip(table.heading, table.scale)]]
    verified = {}
    for stage, candidates, count in (("standard", std, 6), ("wide", table, 27)):
        threshold = 0.98
        if policy == "heldout_geometry":
            threshold = 0.9 if stage == "standard" else 0.95
        winner = choose(candidates, policy, threshold)
        if winner is None:
            continue
        if "geometry" in policy:
            key = (int(winner["query"]), int(winner["rank"]))
            if key not in verified:
                query = photo["queries"][key[0]]
                verified[key] = heldout_geometry(query["matches"][key[1]], query["inside"])
            if not verified[key]:
                continue
        return dict(answered=True, stage=stage, n_queries=count, query=int(winner["query"]),
                    rank=int(winner["rank"]), x=float(winner["x"]), y=float(winner["y"]),
                    probability=float(winner["p"]), inliers=int(winner["inliers"]))
    return dict(answered=False, stage="skip", n_queries=27, query=-1, rank=-1,
                x=math.nan, y=math.nan, probability=math.nan, inliers=0)
