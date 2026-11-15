"""Block: inlier-scale robustness of the candidate judge (Test_Iteration_10, part 2). No GPU, no plotting, no multi-query agreement.

T9 found that if every candidate gets more inliers (a sharper camera, a more permissive matcher, a different map), right and wrong alike, the
judge answers wrong places: x1.25 gave 2-3 wrong answers on NUDT 707, mostly through the 12-inlier override. Tools here:

scale_signals(table, f)        every inlier-count signal (inliers, n15, n22, inl_*, guided15) of every candidate times f (rounded), then the
                               derived features and query-local context recomputed: the stress test of T10 (f = 0.5 ... 2)
far_levels(table, m)           per photo: the mean inliers and mean n22 of the candidates > 4 sigma from the pipeline's choice (all queries,
                               rank < m). Those are almost all wrong places, so they measure how many inliers a WRONG place gets on this camera /
                               map / matcher, without labels
running_level(levels, prior, weight)
                               per data set, in photo order: (weight * prior + sum of the levels so far, this photo included) / (weight + n),
                               i.e. a running mean that starts at the dev-data value (what a deployed system can compute as photos arrive)
add_relative(table, prior, weight, m)
                               rli = log((inliers + 1) / (running far inliers + 1)), rln22 likewise: inlier signals relative to the run's own
                               wrong-place level (scale-free when every count is multiplied by the same factor)
fit_judge(table, sets, features, increasing, factors)
                               the monotone GBM of candidate_judge trained on `sets`, optionally on copies of every set with the inlier signals
                               scaled by each factor (scale augmentation); out-of-fold P for the training rows grouped by ORIGINAL photo
                               (a photo and its scaled copies stay in one fold)
predict(model, table, features)
safety_mode(levels, low, high) per photo: True when the running far-inlier level is outside [low, high] (the range seen in dev data): the strict
                               mode (cut 0.98, no override) is then used for that photo
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from train.checks.candidate_judge import add_context, add_features, labelled

COUNT_COLS = ["inliers", "n15", "n22", "inl_s3", "inl_t2", "inl_t3", "inl_t3_15", "guided15", "inl_lo"]


def scale_signals(table, f, m=None):
    t = table.copy()
    if f != 1:
        for c in COUNT_COLS:
            if c in t.columns:
                t[c] = np.round(t[c] * f)
    t = add_features(t.drop(columns=[c for c in ("li", "ln22", "ratio", "tfrac") if c in t.columns]))
    return add_context(t, m=m)


def far_levels(table, m=None):
    t = table if m is None else table[table["rank"] < m]
    rows = []
    for (d, p), g in t.groupby(["data", "photo"], sort=True):
        u = g[g["inliers"] >= 1]
        if len(u):
            c = u.iloc[int(np.lexsort((-u["coarse"].to_numpy(), -(u["inliers"] + u["z"]).to_numpy()))[0])]
            far = np.hypot(g["x"] - c["x"], g["y"] - c["y"]) > 4 * c["sigma"]
        else:
            far = np.ones(len(g), bool)
        f = g[np.asarray(far)]
        rows.append(dict(data=d, photo=p, far_inl=float(f["inliers"].mean()) if len(f) else 0.0, far_n22=float(f["n22"].mean()) if len(f) else 0.0))
    return pd.DataFrame(rows)


def running_level(levels, prior, weight=30):
    """levels: far_levels output. prior: dict(far_inl, far_n22). Adds run_inl, run_n22 (running mean per data set in photo order)."""
    out = []
    for _, g in levels.sort_values(["data", "photo"]).groupby("data", sort=False):
        k = np.arange(1, len(g) + 1)
        out.append(g.assign(run_inl=(weight * prior["far_inl"] + g["far_inl"].cumsum().to_numpy()) / (weight + k),
                            run_n22=(weight * prior["far_n22"] + g["far_n22"].cumsum().to_numpy()) / (weight + k)))
    return pd.concat(out)


def add_relative(table, prior, weight=30, m=None):
    lv = running_level(far_levels(table, m), prior, weight)
    t = table.drop(columns=[c for c in ("run_inl", "run_n22", "rli", "rln22") if c in table.columns]).merge(
        lv[["data", "photo", "run_inl", "run_n22"]], on=["data", "photo"], how="left")
    t["rli"] = np.log((t["inliers"] + 1) / (t["run_inl"] + 1))
    t["rln22"] = np.log((t["n22"] + 1) / (t["run_n22"] + 1))
    return t


def _model(features, increasing, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    mono = [1 if f in increasing else 0 for f in features]
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, monotonic_cst=mono,
                                          early_stopping=False, random_state=seed)


def fit_judge(table, sets, features, increasing, factors=None, transform=None, folds=5):
    """table: rows of `sets` (unscaled). factors: scale augmentation (None = no augmentation). transform(t) -> t: applied to every (scaled) copy
    before training (e.g. add_relative). Returns (model fitted on all copies, P out-of-fold for the rows of every copy, the stacked copies)."""
    base = table[table["data"].isin(sets)].reset_index(drop=True)
    copies = []
    for f in (factors or [1.0]):
        c = scale_signals(base, f) if f != 1 else base.copy()
        if transform is not None:
            c = transform(c)
        copies.append(c.assign(scale_factor=f))
    t = pd.concat(copies, ignore_index=True)
    usable = (t["inliers"] >= 3).to_numpy()
    lab = labelled(t).to_numpy()
    y = (t["err_sigma"] <= 4).to_numpy()
    X = t[features].to_numpy(float)
    model = _model(features, increasing).fit(X[lab], y[lab])
    p = np.zeros(len(t))
    groups = (t["data"].astype(str) + "_" + t["photo"].astype(str)).to_numpy()
    idx = np.flatnonzero(usable)
    for a, b in GroupKFold(n_splits=folds).split(idx, groups=groups[idx]):
        fit_rows = idx[a][lab[idx[a]]]
        p[idx[b]] = _model(features, increasing).fit(X[fit_rows], y[fit_rows]).predict_proba(X[idx[b]])[:, 1]
    return model, p, t


def predict(model, table, features):
    p = np.zeros(len(table))
    u = (table["inliers"] >= 3).to_numpy()
    if u.any():
        p[u] = model.predict_proba(table.loc[u, features].to_numpy(float))[:, 1]
    return p


def safety_mode(table, low, high):
    """Per row: True when the row's photo has a running far-inlier level (column run_inl, from add_relative) outside [low, high]."""
    return ~table["run_inl"].between(low, high).to_numpy()
