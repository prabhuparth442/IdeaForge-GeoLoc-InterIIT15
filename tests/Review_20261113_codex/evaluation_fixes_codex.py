"""Opt-in evaluation fixes. Import these functions explicitly; existing files stay untouched."""
import numpy as np
from scipy.stats import beta

from evaluation.report import collect_answers as original_collect_answers


def select_certified_tau(scan, sets, alpha=0.005, delta=0.1):
    """Return None when no tested cut is certified. None MUST mean skip all upstream.

    Use an independent calibration set for a fixed, already trained predictor.
    This selector alone does not make reused or cross-fitted data a valid calibration set.
    """
    selected = scan[scan["data"].isin(sets)]
    if selected.empty or set(sets) - set(selected["data"]):
        raise ValueError("Every requested calibration set must be present")
    totals = selected.groupby("tau")[["wrong", "n_answered"]].sum()
    chosen = None
    for tau in sorted(totals.index, reverse=True):
        wrong = int(totals.loc[tau, "wrong"])
        answered = int(totals.loc[tau, "n_answered"])
        if not 0 <= wrong <= answered:
            raise ValueError("Invalid calibration counts")
        upper = 1.0
        if answered > wrong:
            upper = float(beta.ppf(1 - delta, wrong + 1, answered - wrong))
        if upper > alpha:
            break
        chosen = float(tau)
    return chosen


def available_candidate_features(table, m):
    """Compute competition features only from candidates actually fine-matched."""
    from train.checks.candidate_judge import add_context, add_features

    available = table[table["rank"] < m].copy().reset_index(drop=True)
    available = available.drop(columns=["zgap", "ligap"], errors="ignore")
    if available.empty:
        raise ValueError("No candidates: produce one skip per input photo instead")
    return add_context(add_features(available))


def decide_certified(table, probabilities, tau, **kwargs):
    """A failed certificate disables all answers, including overrides and early exits.

    Do not pass a failed calibration's None to candidate_judge.decide directly:
    that older API interprets None as the ordinary inlier-bar baseline.
    """
    import pandas as pd
    from train.checks.candidate_judge import decide

    if tau is not None:
        return decide(table, probabilities, tau, **kwargs)
    rows = []
    for (data, photo), group in table.groupby(["data", "photo"], sort=False):
        rows.append(dict(data=data, photo=photo, stage="skip", pts=0, err_sigma=np.nan,
                         inliers=0.0, p=np.nan, n_queries=0))
    return pd.DataFrame(rows)


def collect_answers(testset, disc, stage2, final, use_minima):
    """Keep empty-search photos in the denominator as skips, with infinite error.

    Run the unchanged reporting function only on rows with a real coarse peak,
    then restore every original sample position. Existing nonempty-row behavior is preserved.
    """
    from copy import copy
    valid_ids, positions = [], []
    for position, photo in enumerate(testset.sample_ids):
        peaks = disc[int(photo)]["peaks"][("zncc", 0.0)]
        if np.any(peaks[:, 1] < 1e8):
            valid_ids.append(int(photo))
            positions.append(position)
    names = ["correlation"]
    if use_minima:
        names += ["+MINIMA re-rank", "+MINIMA re-rank + refine"]
    count = len(testset.sample_ids)
    results = {}
    for name in names:
        results[name] = dict(errors=np.full(count, np.inf), points=np.zeros(count, dtype=int))
    closest = np.full(count, np.inf)
    ratios = np.zeros(count)
    inliers = np.zeros(count) if use_minima else np.array([])
    if not valid_ids:
        return results, closest, ratios, inliers
    subset = copy(testset)
    subset.sample_ids = np.asarray(valid_ids)
    partial, near, ratio, inlier = original_collect_answers(subset, disc, stage2, final, use_minima)
    for name in names:
        for field in ("errors", "points"):
            results[name][field][positions] = partial[name][field]
    closest[positions] = near
    ratios[positions] = ratio
    if use_minima:
        inliers[positions] = inlier
    return results, closest, ratios, inliers
