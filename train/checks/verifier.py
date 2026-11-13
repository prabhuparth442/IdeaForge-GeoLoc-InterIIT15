"""Block: a learned answer-or-skip verifier on the rows of train/checks/verifier_features.decision_rows (no GPU, no plotting).

feature_matrix     the model's inputs from a rows table (no flight height or map level: the evaluation heights differ from the dev photos)
make_model         "logistic" (standardised logistic regression) or "gbm" (sklearn HistGradientBoosting, shallow)
oof_probability    out-of-fold P(wrong) with folds grouped by (data set, photo), so a photo's two rows never sit on both sides
pick_threshold     the largest P(wrong) cut that keeps the wrong answers of the given rows at <= max_wrong
apply_rule         per photo: answer the 6-query choice if its P(wrong) < t_std (or today's rule if keep_today), else the 27-query choice if
                   P(wrong) < t_wide (or >= 11 inliers if keep_today). Returns one row per photo with points, stage, n_queries.

Target: wrong = error > 8 sigma (the -20 answers). Answers between 4 and 8 sigma score 0 and are neutral.
"""
import numpy as np
import pandas as pd

from train.checks.replay import points

FEATURES = ["log_inliers", "z", "rank", "agree", "agree_any", "log_runner_far", "ratio", "fit_scale_dev", "fit_angle", "resid", "hull", "spread",
            "conf_inl", "log_pairs", "abs_heading", "scale_dev", "wide"]


def feature_matrix(rows, features=FEATURES):
    f = pd.DataFrame(index=rows.index)
    f["log_inliers"] = np.log1p(rows["inliers"])
    f["log_runner_far"] = np.log1p(rows["runner_far"])
    f["log_pairs"] = np.log1p(rows["n_pairs"])
    f["abs_heading"] = rows["q_heading"].abs()
    f["scale_dev"] = (rows["q_scale"] - 1).abs()
    f["wide"] = (rows["step"] == "wide27").astype(float)
    for c in ["z", "rank", "agree", "agree_any", "ratio", "fit_scale_dev", "fit_angle", "resid", "hull", "spread", "conf_inl"]:
        f[c] = rows[c].astype(float)
    return f[features].fillna(0.0).to_numpy()


def make_model(kind="logistic"):
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"))
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0, class_weight="balanced",
                                              random_state=0)
    raise ValueError(kind)


def labels(rows):
    return (rows["err_sigma"] > 8).to_numpy().astype(int)


def usable(rows):
    return rows["has_choice"].to_numpy().astype(bool)


def oof_probability(rows, kind="logistic", folds=5, seed=0, features=FEATURES):
    """Out-of-fold P(wrong) for every usable row (NaN for rows without a choice)."""
    from sklearn.model_selection import GroupKFold
    p = np.full(len(rows), np.nan)
    ok = usable(rows)
    X, y = feature_matrix(rows, features), labels(rows)
    groups = (rows["data"].astype(str) + "_" + rows["photo"].astype(str)).to_numpy()
    idx = np.flatnonzero(ok)
    for train, test in GroupKFold(n_splits=folds).split(idx, groups=groups[idx]):
        m = make_model(kind).fit(X[idx[train]], y[idx[train]])
        p[idx[test]] = m.predict_proba(X[idx[test]])[:, 1]
    return p


def fit_predict(train_rows, test_rows, kind="logistic", features=FEATURES):
    ok = usable(train_rows)
    m = make_model(kind).fit(feature_matrix(train_rows, features)[ok], labels(train_rows)[ok])
    p = np.full(len(test_rows), np.nan)
    ok_t = usable(test_rows)
    p[ok_t] = m.predict_proba(feature_matrix(test_rows, features)[ok_t])[:, 1]
    return p, m


def apply_rule(rows, p, t_std, t_wide, keep_today=False, bar=8, agree_min=2, escalation_bar=11, late=False):
    """rows: the decision rows (two per photo), p: P(wrong) per row. Returns one row per (data, photo).
    late=True (with keep_today): today's full rule runs first (6 queries, then the 27-query escalation); the verifier only answers photos today's
    rule skips, with the 27-query choice if P < t_wide, else the 6-query choice if P < t_std. Reason (NUDT 202, photo 377): a verifier at the
    6-query step answered a weak 6-query choice and so threw away the escalation, which would have found the right place."""
    r = rows.assign(p=p)
    out = []
    for (data, photo), g in r.groupby(["data", "photo"], sort=False):
        s = g[g.step == "std6"].iloc[0]
        w = g[g.step == "wide27"].iloc[0]
        if late:
            assert keep_today, "late mode adds to today's rule"
            if s.has_choice and (s.inliers >= bar or s.agree >= agree_min):
                out.append(dict(data=data, photo=photo, stage="std6", points=points(s.err_sigma), n_queries=6, height=s.height)); continue
            if w.has_choice and (w.inliers >= escalation_bar or p_ok(w.p, t_wide)):
                out.append(dict(data=data, photo=photo, stage="wide27", points=points(w.err_sigma), n_queries=27, height=s.height)); continue
            if s.has_choice and p_ok(s.p, t_std):
                out.append(dict(data=data, photo=photo, stage="std6_late", points=points(s.err_sigma), n_queries=27, height=s.height)); continue
            out.append(dict(data=data, photo=photo, stage="skip", points=0, n_queries=27, height=s.height)); continue
        ans_std = bool(s.has_choice) and (p_ok(s.p, t_std) or (keep_today and (s.inliers >= bar or s.agree >= agree_min)))
        if ans_std:
            out.append(dict(data=data, photo=photo, stage="std6", points=points(s.err_sigma), n_queries=6, height=s.height))
            continue
        ans_wide = bool(w.has_choice) and (p_ok(w.p, t_wide) or (keep_today and w.inliers >= escalation_bar))
        out.append(dict(data=data, photo=photo, stage="wide27" if ans_wide else "skip", points=points(w.err_sigma) if ans_wide else 0,
                        n_queries=27, height=s.height))
    return pd.DataFrame(out)


def p_ok(value, threshold):
    return bool(np.isfinite(value) and value < threshold)


def pick_threshold(rows, p, step, max_wrong=0, grid=None):
    """Largest cut t such that rows of `step` with P(wrong) < t contain at most max_wrong wrong answers."""
    sel = (rows["step"] == step).to_numpy() & np.isfinite(p)
    wrong = labels(rows)[sel]
    order = np.sort(p[sel][wrong == 1])
    if len(order) <= max_wrong:
        return 1.0
    return float(order[max_wrong])          # everything strictly below the (max_wrong+1)-th smallest wrong P


def apply_veto_rule(rows, p, t_veto, t_late, bar=8, agree_min=2, escalation_bar=11):
    """Today's rule, but an answer is dropped (veto) when its P(wrong) >= t_veto; a photo still skipped is then answered late (27-query choice
    first, then the 6-query choice) when its P(wrong) < t_late. Returns one row per photo with points and stage."""
    r = rows.assign(p=p)
    out = []
    for (data, photo), g in r.groupby(["data", "photo"], sort=False):
        s = g[g.step == "std6"].iloc[0]
        w = g[g.step == "wide27"].iloc[0]
        stage, pts = "skip", 0
        if s.has_choice and (s.inliers >= bar or s.agree >= agree_min) and not (np.isfinite(s.p) and s.p >= t_veto):
            stage, pts = "std6", points(s.err_sigma)
        elif w.has_choice and w.inliers >= escalation_bar and not (np.isfinite(w.p) and w.p >= t_veto):
            stage, pts = "wide27", points(w.err_sigma)
        elif w.has_choice and p_ok(w.p, t_late):
            stage, pts = "wide27_late", points(w.err_sigma)
        elif s.has_choice and p_ok(s.p, t_late):
            stage, pts = "std6_late", points(s.err_sigma)
        out.append(dict(data=data, photo=photo, stage=stage, points=pts, height=s.height))
    return pd.DataFrame(out)


def choose_veto_cuts(rows, p, grid=None, bar=8, agree_min=2, escalation_bar=11):
    """(t_veto, t_late) with 0 wrong answers on `rows` and the most points; ties -> the most cautious (lowest t_late, lowest t_veto)."""
    grid = grid if grid is not None else np.round(np.arange(0.05, 1.0001, 0.05), 2)
    best = None
    for tv in grid:
        for tl in grid:
            if tl > tv:
                continue
            t = apply_veto_rule(rows, p, tv, tl, bar, agree_min, escalation_bar)
            wrong = int((t.points < 0).sum())
            key = (wrong == 0, t.points.sum(), -tl, -tv)
            if best is None or key > best[0]:
                best = (key, tv, tl)
    return best[1], best[2]
