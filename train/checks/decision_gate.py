"""Block: learn the answer-or-skip gate from per-photo results (no plotting, no GPU).

points_from_error    sigma error -> points of the contest rule
loss_breakdown       where the points of a run go (skipped but right, imprecise, wrong answers)
gate_features        per-photo numbers the gate may look at
grouped_cv_gate      answer flags from a gate trained on other groups only (honest estimate)
score_of_flags       score out of 60 and number of wrong answers for given answer flags
"""
import numpy as np
from sklearn.model_selection import GroupKFold


def points_from_error(error_sigma):
    """Points per photo if it were answered, from error in sigma units (same rule as header/scoring)."""
    error_sigma = np.asarray(error_sigma, dtype=float)
    points = np.full(error_sigma.shape, -20)
    points[error_sigma <= 8] = 0
    points[error_sigma <= 4] = 3
    points[error_sigma <= 2] = 6
    points[error_sigma <= 1] = 10
    return points


def score_of_flags(points, answer_flags):
    """Score out of 60 and number of wrong answers when only the flagged photos are answered."""
    answered_points = np.where(answer_flags, points, 0)
    score = answered_points.sum() / len(points) * 6.0
    wrong = int((answer_flags & (points == -20)).sum())
    return score, wrong


def loss_breakdown(table):
    """Points lost to skipped-but-good, imprecise answers, wrong answers (table: error_sigma, answered)."""
    points = points_from_error(table.error_sigma.values)
    answered = table.answered.values.astype(bool)
    skipped_good = int(points[(~answered) & (points > 0)].sum())
    imprecise = int((10 - points[answered & (points > 0)]).sum())
    wrong = int(points[answered & (points == -20)].sum())
    return {"skipped_but_good": skipped_good, "imprecise": imprecise, "wrong_answers": wrong,
            "skipped_and_far_off": int(((~answered) & (points == -20)).sum())}


def gate_features(table, with_height=False):
    """Columns: log of inliers (capped at 200) and coarse z. Optionally flight height."""
    columns = [np.log1p(table.inliers.clip(upper=200).values), table.z_chosen.values]
    if with_height:
        columns.append(table.height.values)
    return np.column_stack(columns)


def grouped_cv_gate(table, make_model, groups, threshold, with_height=False, folds=5):
    """Answer flags where every photo is judged by a model that never saw its group."""
    points = points_from_error(table.error_sigma.values)
    labels = (points > 0).astype(int)
    features = gate_features(table, with_height)
    flags = np.zeros(len(table), dtype=bool)
    for train_ids, test_ids in GroupKFold(folds).split(features, labels, groups):
        model = make_model().fit(features[train_ids], labels[train_ids])
        flags[test_ids] = model.predict_proba(features[test_ids])[:, 1] > threshold
    return flags


def rescue_flags(table, bar=8, low_bar=4, z_min=2.0, agree_min=2):
    """Answer flags: the normal bar, plus photos with low_bar to bar-1 inliers whose coarse z and query agreement are both high.
    Needs columns inliers, z_chosen, agree_queries (see gate_features.gate_row)."""
    normal = table.inliers.values >= bar
    rescued = (table.inliers.values >= low_bar) & (table.z_chosen.values >= z_min) & (table.agree_queries.values >= agree_min)
    return normal | rescued


def paired_bootstrap_gain(points, flags_new, flags_old, draws=4000, seed=0):
    """Mean gain in marks out of 60 of the new answer flags over the old ones, with a 95% interval over photos (paired resampling)."""
    gain = (np.where(flags_new, points, 0) - np.where(flags_old, points, 0)).astype(float)
    rng = np.random.default_rng(seed)
    means = [gain[rng.integers(0, len(gain), len(gain))].mean() * 6.0 for _ in range(draws)]
    return gain.mean() * 6.0, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
