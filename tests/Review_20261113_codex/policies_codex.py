"""Isolated decision experiments. No truth is read until an answer is scored."""
import math

import numpy as np

from train.checks import replay


def selected_answer(photo, ids, stage, minimum=1):
    candidates = replay.pooled(photo, ids)
    eligible = candidates["inliers"] >= minimum
    if not eligible.any():
        return None
    scores = np.where(eligible, candidates["inliers"] + candidates["z"], -np.inf)
    best = int(np.lexsort((-candidates["coarse"], -scores))[0])
    return dict(x=float(candidates["x"][best]), y=float(candidates["y"][best]),
                inliers=float(candidates["inliers"][best]), n_queries=len(ids), stage=stage)


def score_answer(photo, answer):
    result = dict(answer)
    result["answered"] = result["stage"] != "skip"
    result["err_sigma"] = replay.error_sigma(photo, result["x"], result["y"])
    result["points"] = 0
    if result["answered"]:
        result["points"] = replay.points(result["err_sigma"])
    return result


def eligible_rescue(photo):
    original = replay.decide_current(photo)
    if original["answered"]:
        return original
    # Reconsider only the final stage, preserving all baseline answers and cost.
    ids = replay.query_ids(photo, replay.WIDE)
    answer = selected_answer(photo, ids, "eligible_rescue", minimum=replay.ESCALATION_BAR)
    if answer is None:
        return original
    return score_answer(photo, answer)


def early_twelve(photo):
    # Try the plain query first, as headless does. Research normally starts at -5 degrees.
    order = [(0.0, 1.0)]
    for key in replay.STANDARD:
        if key not in order:
            order.append(key)
    ids = replay.query_ids(photo, order)
    for count in range(1, len(ids)):
        answer = selected_answer(photo, ids[:count], "early12")
        if answer is not None and answer["inliers"] >= 12:
            return score_answer(photo, answer)
    return replay.decide_current(photo)


def headless_policy(photo, cap=18, stop_second=False):
    photo = replay.truncate_m(photo, 5)
    standard = [(0.0, 1.0)]
    for key in replay.STANDARD:
        if key not in standard:
            standard.append(key)
    extra = []
    for scale in (1.0, 1.05):
        for heading in (-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0):
            key = (heading, scale)
            if key not in standard:
                extra.append(key)
    ids = replay.query_ids(photo, standard[:cap])
    answer = selected_answer(photo, ids, "standard")
    if answer is not None and answer["inliers"] >= 8:
        return score_answer(photo, answer)
    for query in replay.query_ids(photo, extra):
        if len(ids) >= cap:
            break
        ids.append(query)
        answer = selected_answer(photo, ids, "second")
        if stop_second and answer is not None and answer["inliers"] >= 8:
            return score_answer(photo, answer)
    if answer is None:
        return dict(answered=False, points=0, err_sigma=math.nan, x=math.nan, y=math.nan,
                    inliers=0, stage="skip", n_queries=len(ids))
    if answer["inliers"] < 8:
        answer["stage"] = "skip"
    return score_answer(photo, answer)
