"""The DINO score used in the decision, not only in the order of the peaks.

second_stage checks every one of the 25 candidates on its own, and final_answer takes the one with the most inliers, so
re-ordering the 25 cannot change the answer. Here the DINO score of each candidate (z-score over the 50 shortlisted places)
is added to the inlier count when the candidate is chosen, and it can rescue a photo that has a few inliers too few.

With gamma = 0 and no rescue this gives the same answers as result_thresholding.decision.final_answer / decide
(for the photos that are answered).
"""
import numpy as np

from header.settings import ANSWER_MIN_INLIERS

MISSING = 1e8           # pick_peaks writes x = 1e9 for a peak it could not find


def shortlist_z(dense):
    """z-score of the DINO dense scores over the shortlist. NaN (no peak) becomes 0."""
    values = np.asarray(dense, dtype=float)
    result = np.zeros(len(values))
    good = np.isfinite(values)
    if good.sum() > 1:
        spread = values[good].std()
        if spread < 1e-9:
            spread = 1.0
        result[good] = (values[good] - values[good].mean()) / spread
    return result


def candidate_z(peaks, details):
    """DINO z-score of each of the 25 peaks that semantic_layer returned. details is the second output of semantic_layer."""
    shortlist = details["shortlist"]
    z = shortlist_z(details["dense"])
    lookup = {}
    for i in range(len(shortlist)):
        if shortlist[i, 1] < MISSING:
            lookup[(float(shortlist[i, 1]), float(shortlist[i, 2]))] = z[i]
    result = np.zeros(len(peaks))
    for k in range(len(peaks)):
        key = (float(peaks[k, 1]), float(peaks[k, 2]))
        if key in lookup:
            result[k] = lookup[key]
    return result


def choose_candidate(peaks, stage2, z, gamma=0.0):
    """Pick the candidate with the highest  inliers + gamma * z.  Returns x, y, inliers, index, z of the chosen one.
    Candidates with 0 inliers are never chosen. If there is none, the first peak is returned with 0 inliers."""
    assert len(peaks) == len(stage2["inliers"]) == len(z), "one inlier count and one z per peak"
    assert gamma >= 0, "gamma must be >= 0"
    inliers = stage2["inliers"]
    usable = inliers > 0
    if not usable.any():
        return peaks[0, 1], peaks[0, 2], 0.0, 0, float(z[0])
    score = np.full(len(peaks), -1e9)
    for k in range(len(peaks)):
        if usable[k]:
            score[k] = inliers[k] + gamma * z[k]
    order = np.lexsort((-peaks[:, 0], -score))
    best = int(order[0])
    x = peaks[best, 1] + stage2["shifts"][best, 0]
    y = peaks[best, 2] + stage2["shifts"][best, 1]
    return x, y, float(inliers[best]), best, float(z[best])


def decide_with_dino(inliers, z_best, rescue_inliers=None, rescue_z=None, veto_inliers=None, veto_z=None, min_inliers=None):
    """Answer or skip.
    normal: answer when inliers >= min_inliers (None = settings.ANSWER_MIN_INLIERS, 12).
    rescue: also answer with at least rescue_inliers inliers when the DINO z of the chosen place is at least rescue_z.
    veto:   skip a normal answer when it has fewer than veto_inliers inliers and the DINO z is below veto_z.
              With 0 inliers the answer is the first peak (the one DINO ranks first) without a shift correction."""
    assert (rescue_inliers is None) == (rescue_z is None), "give rescue_inliers and rescue_z together"
    assert (veto_inliers is None) == (veto_z is None), "give veto_inliers and veto_z together"
    assert inliers >= 0, "inlier count cannot be negative, got %s" % inliers
    bar = ANSWER_MIN_INLIERS if min_inliers is None else min_inliers
    if inliers >= bar:
        if veto_inliers is not None:
            if inliers < veto_inliers and z_best < veto_z:
                return False
        return True
    if rescue_inliers is not None:
        if inliers >= rescue_inliers and z_best >= rescue_z:
            return True
    return False
