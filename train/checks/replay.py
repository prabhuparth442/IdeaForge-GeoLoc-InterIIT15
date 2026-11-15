"""Block: replay the decision of agreement_query_pipeline.py offline from a rich dump (train/checks/rich_dump.py). No GPU, no plotting.

pooled            the candidates of a set of queries as flat arrays (x, y after the shift, inliers, z, coarse score, owner query, rank)
choose            today's choice: most inliers + gamma * z (0-inlier candidates never chosen; ties by the coarse score), as choose_candidate
agreement         number of queries whose own best candidate (>= min inliers) lies within a radius of a place
decide_current    bar 8 or agreement >= 2 on the 6 standard queries, else escalate to all 27 queries with bar 11 (today's best rule)
points / score    scoring of src/header/scoring.py from the error in sigma
paired_bootstrap  95 % interval of the mean gain of b over a (same photos)
refit_queries     inliers and shifts of every candidate re-made from the stored matches with any fit (fine_matching.loftr_matcher.stage2_from_matches)

Every function takes `photo` = load_rich(path). `inliers` / `shifts` overrides (lists, one array per query) let a re-fit replace the stored ones.
"""
import math

import numpy as np

STANDARD = [(h, s) for s in (1.0, 1.05) for h in (-5.0, 0.0, 5.0)]
WIDE = [(h, s) for s in (0.95, 1.0, 1.05) for h in (-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0)]
BAR, AGREE_MIN, AGREE_RADIUS_SIGMA, AGREE_QUERY_INLIERS, ESCALATION_BAR = 8, 2, 1.0, 3, 11


def query_ids(photo, keys):
    """Query numbers of the (heading, scale) pairs in `keys` (all must be in the dump)."""
    found = {(round(q["heading"], 3), round(q["scale"], 3)): i for i, q in enumerate(photo["queries"])}
    ids = []
    for h, s in keys:
        key = (round(h, 3), round(s, 3))
        assert key in found, "query %s not in the dump" % (key,)
        ids.append(found[key])
    return ids


def pooled(photo, ids, inliers=None, shifts=None):
    parts = {k: [] for k in ("x", "y", "inliers", "z", "coarse", "owner", "rank")}
    for i in ids:
        q = photo["queries"][i]
        inl = np.asarray(q["inliers"] if inliers is None else inliers[i], float)
        sh = np.asarray(q["shifts"] if shifts is None else shifts[i], float)
        parts["x"].append(q["peaks"][:, 1] + sh[:, 0])
        parts["y"].append(q["peaks"][:, 2] + sh[:, 1])
        parts["inliers"].append(inl)
        parts["z"].append(np.asarray(q["z"], float))
        parts["coarse"].append(q["peaks"][:, 0])
        parts["owner"].append(np.full(len(inl), i))
        parts["rank"].append(np.arange(len(inl)))
    return {k: np.concatenate(v) for k, v in parts.items()}


def choose(cand, gamma=1.0):
    """Index of the chosen candidate, or -1 when no candidate has an inlier (the pipeline then reports peak 0 with 0 inliers: a skip)."""
    usable = cand["inliers"] > 0
    if not usable.any():
        return -1
    score = np.where(usable, cand["inliers"] + gamma * cand["z"], -1e9)
    return int(np.lexsort((-cand["coarse"], -score))[0])


def agreement(photo, ids, x, y, radius_m, min_query_inliers=AGREE_QUERY_INLIERS, inliers=None, shifts=None):
    count = 0
    for i in ids:
        c = pooled(photo, [i], inliers, shifts)
        best = int(np.argmax(c["inliers"]))
        if c["inliers"][best] >= min_query_inliers and math.hypot(c["x"][best] - x, c["y"][best] - y) <= radius_m:
            count += 1
    return count


def query_seconds(photo, ids):
    return float(sum(sum(photo["queries"][i]["seconds"].values()) for i in ids))


def error_sigma(photo, x, y):
    return math.hypot(x - photo["truth"][0], y - photo["truth"][1]) / photo["sigma"]


def points(err_sigma):
    if err_sigma <= 1:
        return 10
    if err_sigma <= 2:
        return 6
    if err_sigma <= 4:
        return 3
    if err_sigma <= 8:
        return 0
    return -20


def score(points_list):
    """Score out of 60 (skips count as 0)."""
    return 60.0 * max(np.sum(points_list), 0) / (10.0 * len(points_list))


def decide_current(photo, inliers=None, shifts=None, bar=BAR, agree_min=AGREE_MIN, agree_radius_sigma=AGREE_RADIUS_SIGMA,
                   escalation_bar=ESCALATION_BAR, escalate=True, gamma=1.0, agree_query_inliers=AGREE_QUERY_INLIERS):
    """Returns dict(answered, x, y, inliers, stage, n_queries, seconds, err_sigma, points, chosen_owner, agree)."""
    ids = query_ids(photo, STANDARD)
    cand = pooled(photo, ids, inliers, shifts)
    k = choose(cand, gamma)
    x, y, inl = (cand["x"][k], cand["y"][k], cand["inliers"][k]) if k >= 0 else (cand["x"][0], cand["y"][0], 0.0)
    agree = agreement(photo, ids, x, y, agree_radius_sigma * photo["sigma"], agree_query_inliers, inliers=inliers, shifts=shifts) if k >= 0 else 0
    used = ids
    if inl >= bar:
        stage = "bar"
    elif agree >= agree_min:
        stage = "agreement"
    elif escalate:
        used = query_ids(photo, WIDE)
        cand = pooled(photo, used, inliers, shifts)
        k = choose(cand, gamma)
        x, y, inl = (cand["x"][k], cand["y"][k], cand["inliers"][k]) if k >= 0 else (cand["x"][0], cand["y"][0], 0.0)
        stage = "escalation" if inl >= escalation_bar else "skip"
    else:
        stage = "skip"
    answered = stage != "skip"
    err = error_sigma(photo, x, y)
    return dict(answered=answered, x=x, y=y, inliers=inl, stage=stage, n_queries=len(used), seconds=query_seconds(photo, used),
                err_sigma=err, points=points(err) if answered else 0, chosen_owner=int(cand["owner"][k]) if k >= 0 else -1, agree=agree)


def paired_bootstrap(a, b, n=4000, seed=0):
    """Mean gain of b over a in score-out-of-60 units (score = 6 x mean points per photo) with a 95 % percentile interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "paired: same photos"
    d = (b - a) * 6.0
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def refit_queries(photo, min_confidence=0.22, pixels=None, fit=None, seed=7):
    """Inliers and shifts of every candidate of every query, re-made from the stored matches. Returns (inliers list, shifts list)."""
    import cv2
    from fine_matching.loftr_matcher import stage2_from_matches
    out_inl, out_sh = [], []
    for q in photo["queries"]:
        stored = dict(inside=q["inside"], margin=q["margin"], border_px=q["border_px"], items=q["matches"])
        cv2.setRNGSeed(seed)
        s2 = stage2_from_matches(stored, photo["level"], min_confidence, pixels=pixels, fit=fit)
        out_inl.append(s2["inliers"])
        out_sh.append(s2["shifts"])
    return out_inl, out_sh


def truncate_m(photo, m):
    """The photo as if only the first m candidates of every query (the DINO order) had gone to the fine stage (Jetson runs M = 5)."""
    out = dict(photo)
    out["queries"] = []
    for q in photo["queries"]:
        t = dict(q)
        for key in ("peaks", "z", "inliers", "shifts"):
            t[key] = q[key][:m]
        t["matches"] = q["matches"][:m]
        out["queries"].append(t)
    return out


def load_set(folder, with_matches=False):
    """All photo_<id>.npz of a dump folder -> {id: photo}."""
    from pathlib import Path
    from train.checks.rich_dump import load_rich
    files = sorted(Path(folder).glob("photo_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    return {int(p.stem.split("_")[1]): load_rich(p, with_matches) for p in files}


def spaced_agreement(photo, ids, x, y, radius_m, min_query_inliers=AGREE_QUERY_INLIERS, min_gap_deg=5.0):
    """Agreement that only counts queries whose headings are at least min_gap_deg apart (queries 2.5 degrees apart see nearly the same picture
    and fail together). Returns the largest number of agreeing queries with pairwise heading gaps >= min_gap_deg (scales do not add independence)."""
    headings = []
    for i in ids:
        c = pooled(photo, [i])
        best = int(np.argmax(c["inliers"]))
        if c["inliers"][best] >= min_query_inliers and math.hypot(c["x"][best] - x, c["y"][best] - y) <= radius_m:
            headings.append(photo["queries"][i]["heading"])
    count, last = 0, -1e9
    for h in sorted(set(headings)):
        if h - last >= min_gap_deg - 1e-9:
            count, last = count + 1, h
    return count


def decide_spaced(photo, agree_min=2, min_gap_deg=5.0, escalation_agree=2, escalation_gap=7.5, escalation_min_inliers=3, **kw):
    """Today's rule, plus: an escalated photo below bar 11 is answered when >= escalation_agree queries with headings >= escalation_gap apart agree
    on the 27-query choice (each query's own best has >= escalation_min_inliers inliers)."""
    r = decide_current(photo, **kw)
    if r["stage"] != "skip" or r["n_queries"] != len(WIDE):
        return r
    ids = query_ids(photo, WIDE)
    cand = pooled(photo, ids)
    k = choose(cand)
    if k < 0:
        return r
    x, y = cand["x"][k], cand["y"][k]
    n = spaced_agreement(photo, ids, x, y, AGREE_RADIUS_SIGMA * photo["sigma"], escalation_min_inliers, escalation_gap)
    if n >= escalation_agree:
        err = error_sigma(photo, x, y)
        r.update(answered=True, x=x, y=y, inliers=float(cand["inliers"][k]), stage="esc_agree", err_sigma=err, points=points(err), agree=n)
    return r


def headroom(photo, ids, right_sigma=1.0, wrong_sigma=8.0):
    """Best inliers at a right place (any candidate of `ids` within right_sigma of the truth) and at a wrong place (beyond wrong_sigma), and the
    best coarse rank of a right candidate. Shows whether a skipped photo is a matching problem (right place present, too few inliers) or a
    recall problem (right place absent)."""
    c = pooled(photo, ids)
    err = np.hypot(c["x"] - photo["truth"][0], c["y"] - photo["truth"][1]) / photo["sigma"]
    right, wrong = err <= right_sigma, err > wrong_sigma
    return dict(right_present=bool(right.any()), right_best=float(c["inliers"][right].max()) if right.any() else -1.0,
                wrong_best=float(c["inliers"][wrong].max()) if wrong.any() else 0.0,
                right_queries=int(len(np.unique(c["owner"][right]))), right_rank=int(c["rank"][right].min()) if right.any() else -1)


def hard_photo_ids(photos):
    """Photos the 6 standard queries do not answer (today's rule escalates them): the ones a second opinion can still change."""
    return sorted(pid for pid, ph in photos.items() if decide_current(ph)["stage"] not in ("bar", "agreement"))
