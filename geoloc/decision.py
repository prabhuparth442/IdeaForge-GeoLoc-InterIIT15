"""Decision: which candidate wins, and whether it is good enough to answer.

All candidates of all queries are put in one list. The winner has the highest  inliers + GAMMA * (DINO z-score).
Candidates with 0 inliers never win. If nobody has inliers, the first candidate is returned as a best guess (inliers 0).
The photo is answered only when the winner has at least MIN_INLIERS inliers, otherwise it is a no_match.
Optional (off by default, see agreement.py): the first pass is also answered when several of its queries agree on the place, and the bar rises when wrong places
start to collect more inliers than on the development photos.
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


def pool_aligned(results):
    """The aligned-count z of every candidate, in the same order as pool_queries (-inf where it was not computed)."""
    parts = []
    for result in results:
        count = len(result["peaks"])
        parts.append(np.asarray(result["stage2"].get("aligned", np.full(count, -np.inf)), dtype=float))
    return np.concatenate(parts)


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


def should_answer(inliers, factor=1.0, second_pass=False):
    """factor is the inflation guard's factor for the bar (1.0 = the bar as written in settings.py).
    A photo that needed the second pass uses SECOND_PASS_MIN_INLIERS when it is set."""
    base = settings.MIN_INLIERS
    if second_pass and settings.SECOND_PASS_MIN_INLIERS is not None:
        base = settings.SECOND_PASS_MIN_INLIERS
    bar = int(round(base * factor))
    return inliers >= bar
