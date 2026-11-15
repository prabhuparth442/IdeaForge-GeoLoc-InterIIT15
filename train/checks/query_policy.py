"""Block: time-aware query policies, replayed offline from a rich dump (all 27 queries stored). No GPU, no plotting.

A policy runs the queries one at a time in a fixed order and decides after each one whether to stop. Its cost is the number of queries it ran
(on the Jetson the time per photo is about proportional to it: every query pays its own coarse FFT, DINO and LoFTR).

run_policy(photo, std_order, wide_order, early_bar, agree_min, escalation_bar, max_queries)
  1. standard queries in std_order; after each, if the pooled choice has >= early_bar inliers -> answer now (early exit; None = never)
  2. after all standard queries: today's rule (bar 8 or agreement >= agree_min)
  3. escalation: the other wide queries in wide_order, after each the pooled choice over everything run so far is answered at >= escalation_bar
  4. max_queries caps the total (the time budget); when hit, the photo is skipped unless step 2's rule already answered
Returns dict(answered, points, err_sigma, n_queries, stage, x, y). agree_query_inliers: inliers a query's own best needs to count.
z_veto = (z0, safe): every answer also needs the chosen place's DINO z >= z0 or >= safe inliers (Iteration 8).
"""
from train.checks.replay import (AGREE_RADIUS_SIGMA, BAR, ESCALATION_BAR, STANDARD, WIDE, agreement, choose, error_sigma, points, pooled,
                                 query_ids)


def _choice(photo, ids, with_z=False):
    cand = pooled(photo, ids)
    k = choose(cand)
    if k < 0:
        return (None, None, 0.0, -99.0) if with_z else (None, None, 0.0)
    if with_z:
        return cand["x"][k], cand["y"][k], float(cand["inliers"][k]), float(cand["z"][k])
    return cand["x"][k], cand["y"][k], float(cand["inliers"][k])


def _passes(z, inl, z_veto):
    """z_veto = (z0, safe): an answer needs DINO z >= z0 or inliers >= safe (None = no veto)."""
    return z_veto is None or z >= z_veto[0] or inl >= z_veto[1]


def _result(photo, x, y, n, stage):
    if stage == "skip" or x is None:
        return dict(answered=False, points=0, err_sigma=float("nan") if x is None else error_sigma(photo, x, y), n_queries=n, stage="skip",
                    x=x, y=y)
    err = error_sigma(photo, x, y)
    return dict(answered=True, points=points(err), err_sigma=err, n_queries=n, stage=stage, x=x, y=y)


def run_policy(photo, std_order=STANDARD, wide_order=None, early_bar=None, agree_min=2, bar=BAR, escalation_bar=ESCALATION_BAR,
               max_queries=27, agree_query_inliers=3, z_veto=None):
    std_ids = query_ids(photo, std_order)
    run = []
    x = y = None
    for qid in std_ids:
        if len(run) >= max_queries:
            break
        run.append(qid)
        x, y, inl, z = _choice(photo, run, True)
        if early_bar is not None and inl >= early_bar and len(run) < len(std_ids) and _passes(z, inl, z_veto):
            return _result(photo, x, y, len(run), "early")
    if x is not None and _passes(z, inl, z_veto):
        if inl >= bar:
            return _result(photo, x, y, len(run), "bar")
        if agreement(photo, run, x, y, AGREE_RADIUS_SIGMA * photo["sigma"], agree_query_inliers) >= agree_min:
            return _result(photo, x, y, len(run), "agreement")
    if wide_order is None:
        wide_order = [k for k in WIDE if k not in std_order]
    for qid in query_ids(photo, [k for k in wide_order if k not in std_order]):
        if len(run) >= max_queries:
            break
        run.append(qid)
        x, y, inl, z = _choice(photo, run, True)
        if inl >= escalation_bar and _passes(z, inl, z_veto):
            return _result(photo, x, y, len(run), "escalation")
    return _result(photo, x, y, len(run), "skip")


def policy_table(photos, **policy):
    """One row per photo for this policy: photo, answered, points, err_sigma, n_queries, stage."""
    import pandas as pd
    rows = []
    for pid, photo in photos.items():
        r = run_policy(photo, **policy)
        r["photo"] = pid
        r["height"] = photo["height"]
        rows.append(r)
    return pd.DataFrame(rows)


def greedy_extra_queries(photo_sets, k, candidates=None, wrong_penalty=1000, **policy):
    """Choose up to k escalation queries one at a time, each time adding the one that most raises the summed points over all photo_sets
    (list of {id: photo}); a wrong answer (-20) costs wrong_penalty extra, so the choice keeps wrong answers at 0 when it can.
    Returns the ordered list of (heading, scale) and the points after each step."""
    from train.checks.replay import WIDE
    std = policy.get("std_order", STANDARD)
    pool = [q for q in (candidates or WIDE) if q not in std]
    chosen, trace = [], []
    for _ in range(k):
        best, best_value = None, None
        for q in pool:
            if q in chosen:
                continue
            value = 0
            for photos in photo_sets:
                for photo in photos.values():
                    r = run_policy(photo, wide_order=chosen + [q], max_queries=len(std) + len(chosen) + 1, **policy)
                    value += r["points"] - (wrong_penalty if r["points"] < 0 else 0)
            if best_value is None or value > best_value:
                best, best_value = q, value
        chosen.append(best)
        trace.append(best_value)
    return chosen, trace
