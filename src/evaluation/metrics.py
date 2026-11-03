import numpy as np

from header.scoring import final_score
from header.settings import SEED


def auc(values, is_right):
    assert len(values) == len(is_right) and len(values) > 0, "values and is_right must be the same non-zero length"
    values = np.asarray(values, dtype=float)
    is_right = np.asarray(is_right).astype(bool)
    n_right = is_right.sum()
    n_wrong = (~is_right).sum()
    if n_right == 0 or n_wrong == 0:
        return float("nan")
    ranks = values.argsort().argsort() + 1
    return (ranks[is_right].sum() - n_right * (n_right + 1) / 2) / (n_right * n_wrong)


def gated_curve(signal, points):
    assert len(signal) == len(points) and len(signal) > 0, "signal and points must be the same non-zero length"
    thresholds = np.quantile(signal, np.linspace(0, 0.95, 40))
    shares, scores = [], []
    for t in thresholds:
        answered = signal >= t
        shares.append(answered.mean())
        scores.append(final_score(list(np.where(answered, points, 0))))
    return np.array(shares), np.array(scores)


def best_gate(signal, points):
    assert len(signal) == len(points) and len(signal) > 0, "signal and points must be the same non-zero length"
    best = (-1.0, 0.0, 1.0)
    for t in np.quantile(signal, np.linspace(0, 0.95, 40)):
        answered = signal >= t
        s = final_score(list(np.where(answered, points, 0)))
        if s > best[0]:
            best = (s, t, answered.mean())
    return best


def honest_gate(signal, points, repeats=300):
    assert len(signal) == len(points) and len(signal) >= 4, "need at least 4 photos with one signal value and one points value each"
    assert repeats > 0, "repeats must be positive"
    rng = np.random.default_rng(SEED + 19)
    scores, shares = [], []
    for _ in range(repeats):
        order = rng.permutation(len(points))
        dev, test = order[: len(points) // 2], order[len(points) // 2:]
        _, t, _ = best_gate(signal[dev], points[dev])
        answered = signal[test] >= t
        scores.append(final_score(list(np.where(answered, points[test], 0))))
        shares.append(answered.mean())
    return np.mean(scores), np.std(scores), np.mean(shares)
