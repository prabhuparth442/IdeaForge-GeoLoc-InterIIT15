"""Decision: which candidate wins, and whether it is good enough to answer.

All candidates of all queries are put in one list. The winner has the highest  inliers + GAMMA * (DINO z-score).
Candidates with 0 inliers never win. If nobody has inliers, the first candidate is returned as a best guess (inliers 0).
The photo is answered only when the winner has at least MIN_INLIERS inliers, otherwise it is a no_match.
"""
import numpy as np

import settings


def pool_queries(results):
    """Candidates of all queries as one list. Each result has peaks (M, 5), z (M,) and stage2 (inliers (M,), shifts (M, 2)).
    Returns peaks, inliers, shifts, z and owner (which query each candidate came from)."""
    peaks = np.concatenate([np.asarray(r["peaks"], dtype=float) for r in results])
    z = np.concatenate([np.asarray(r["z"], dtype=float) for r in results])
    inliers = np.concatenate([np.asarray(r["stage2"]["inliers"], dtype=float) for r in results])
    shifts = np.concatenate([np.asarray(r["stage2"]["shifts"], dtype=float) for r in results])
    owner = np.concatenate([np.full(len(r["peaks"]), index, dtype=int) for index, r in enumerate(results)])
    return peaks, inliers, shifts, z, owner


def choose_candidate(peaks, inliers, shifts, z):
    """Returns x, y (metres), the winner's inliers, its row in the pooled list and its DINO z."""
    usable = inliers > 0
    if not usable.any():
        return peaks[0, 1], peaks[0, 2], 0.0, 0, float(z[0])
    score = np.full(len(peaks), -1e9)
    for k in range(len(peaks)):
        if usable[k]:
            score[k] = inliers[k] + settings.GAMMA * z[k]
    order = np.lexsort((-peaks[:, 0], -score))      # best score first, the coarse score breaks ties
    best = int(order[0])
    x = peaks[best, 1] + shifts[best, 0]
    y = peaks[best, 2] + shifts[best, 1]
    return x, y, float(inliers[best]), best, float(z[best])


def should_answer(inliers):
    return inliers >= settings.MIN_INLIERS
