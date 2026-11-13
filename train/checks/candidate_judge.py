"""Block: a candidate-level judge for the answer-or-skip decision, trained on every fine-matched candidate (not only the chosen one), plus the
decision replays that use it. Works on the tables of train/checks/decision_signals.py. No GPU, no plotting. No multi-query agreement anywhere.

Why a candidate-level judge (and not Test_Iteration_6's verifier on the chosen candidate): a chosen-candidate verifier sees ~5 wrong answers in
1,379 photos, too few to learn from. Every photo, however, has dozens of fine-matched candidates at wrong places, many with 3-10 inliers (look-alike
buildings): 9,600 labelled wrong candidates with >= 3 inliers in the five seen sets. The judge learns from those what a chance match looks like.

Why monotone gradient boosting: the judge must not learn a quirk of one data set (for example "more inliers is worse in this inlier band").
P(right) is forced to rise with the inlier count, DINO z, inlier hull and coverage. Shallow trees, strong L2.

add_features(table)             derived columns used by the judge (log counts, ratios)
add_context(table)              query-local uniqueness: zgap, ligap against the strongest far place of the same query (FEATURES_CTX)
labelled(table)                 rows the judge learns from: >= 3 inliers and clearly right (<= 4 sigma) or clearly wrong (> 8 sigma)
make_judge(features, kind)      the model ("gbm": monotone HistGradientBoosting; "logit": standardised logistic regression)
judge_probability(table, train_sets, features, kind, folds)
                                P(right) for every row with >= 3 inliers (0 otherwise). Rows of a training set get out-of-fold values (folds grouped
                                by photo), the other sets get the full model's value, so every number is a prediction for photos the model did not see.
decide(table, p, tau, ...)      per photo: the 6 standard queries, then (if not answered) the wider queries; answer the chosen candidate when
                                p >= tau (or inliers >= override). PC: all 27 queries at once. Jetson: M = 5, extra queries one at a time, cap 10.
held_out_probabilities          the leave-one-set-out judges once, reusable by every replay
scan_tau / smallest_safe_tau / ltt_tau   the judge cut (ltt_tau: upper confidence bound of the wrong rate <= alpha, Learn-then-Test);
                                smallest_safe_tau: the judge cut: the smallest cut with 0 wrong on the training sets at it and at every larger cut, plus a margin
FROZEN, FROZEN_VARIANTS, freeze, judge_new, decide_frozen   the frozen T7 rule: fit on the seen sets, apply once to a blind set
baseline(table, bar, escalation_bar, ...)   the same replay with the plain inlier bars (no agreement): today's rule without its agreement step.
"""
import numpy as np
import pandas as pd

STANDARD = [(h, s) for s in (1.0, 1.05) for h in (-5.0, 0.0, 5.0)]
WIDE = [(h, s) for s in (0.95, 1.0, 1.05) for h in (-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0)]
SECOND_PASS = [(h, s) for s in (1.0, 1.05) for h in (-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0)]   # the submitted Jetson code's order

FEATURES = ["li", "z", "rank", "coarse", "sharp", "hull", "cover9", "spread", "conf_inl", "hc_ratio", "scale_dev", "angle", "shift_px", "resid",
            "ln22", "ratio", "tfrac"]
INCREASING = {"li", "z", "hull", "cover9"}            # P(right) may only rise with these
DECREASING = set()


def add_features(table):
    t = table.copy()
    t["li"] = np.log1p(t["inliers"])
    t["ln22"] = np.log1p(t["n22"])
    t["ratio"] = t["inliers"] / np.maximum(t["n22"], 1)
    t["tfrac"] = t["inl_t3"] / np.maximum(t["inliers"], 1)
    keys = {(round(h, 2), round(s, 2)) for h, s in STANDARD}
    t["std"] = [(round(h, 2), round(s, 2)) in keys for h, s in zip(t["heading"], t["scale"])]
    t["qkey"] = list(zip(t["heading"].round(2), t["scale"].round(2)))
    t["pts"] = points_array(t["err_sigma"].to_numpy())
    return t


def add_context(table, far_sigma=4.0):
    """Query-local uniqueness (not agreement: it can only lower P): for every candidate, the strongest OTHER candidate of the same query that sits
    more than far_sigma sigma away. zgap = own z - that place's z; ligap = log(1 + own inliers) - log(1 + its inliers). Logic: a right place stands
    out from the rest of its own query; when a far place matches about as well, the photo is ambiguous (repeated buildings) and should be skipped."""
    parts = []
    for _, g in table.groupby(["data", "photo", "query"], sort=False):
        x, y, s = g["x"].to_numpy(), g["y"].to_numpy(), float(g["sigma"].iloc[0])
        li, z = g["inliers"].to_numpy(), g["z"].to_numpy()
        far = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) > far_sigma * s
        np.fill_diagonal(far, False)
        other_li = np.where(far, li[None, :], 0).max(1)
        other_z = np.where(far, z[None, :], -5).max(1)
        parts.append(pd.DataFrame(dict(zgap=z - other_z, ligap=np.log1p(li) - np.log1p(other_li)), index=g.index))
    return table.join(pd.concat(parts))


FEATURES_CTX = FEATURES + ["zgap", "ligap"]


def points_array(err_sigma):
    e = np.asarray(err_sigma, float)
    return np.select([e <= 1, e <= 2, e <= 4, e <= 8], [10, 6, 3, 0], -20)


def labelled(table):
    return (table["inliers"] >= 3) & ((table["err_sigma"] <= 4) | (table["err_sigma"] > 8))


def make_judge(features=FEATURES, kind="gbm", seed=0):
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        mono = [1 if f in INCREASING else -1 if f in DECREASING else 0 for f in features]
        return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, monotonic_cst=mono, early_stopping=False,
                                              random_state=seed)
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
    raise ValueError(kind)


def judge_probability(table, train_sets, features=FEATURES, kind="gbm", folds=5, seed=0):
    """table: add_features output (all sets). Returns P(right) per row (0 for rows under 3 inliers) and the model fitted on all training rows."""
    from sklearn.model_selection import GroupKFold
    t = table.reset_index(drop=True)
    p = np.zeros(len(t))
    usable = (t["inliers"] >= 3).to_numpy()
    lab = labelled(t).to_numpy()
    train = t["data"].isin(train_sets).to_numpy()
    y = (t["err_sigma"] <= 4).to_numpy()
    X = t[features].to_numpy(float)
    model = make_judge(features, kind, seed).fit(X[train & lab], y[train & lab])
    other = usable & ~train
    if other.any():
        p[other] = model.predict_proba(X[other])[:, 1]
    groups = (t["data"].astype(str) + "_" + t["photo"].astype(str)).to_numpy()
    idx = np.flatnonzero(train & usable)
    if len(idx):
        for a, b in GroupKFold(n_splits=folds).split(idx, groups=groups[idx]):
            fit_rows = idx[a][lab[idx[a]]]
            m = make_judge(features, kind, seed).fit(X[fit_rows], y[fit_rows])
            p[idx[b]] = m.predict_proba(X[idx[b]])[:, 1]
    return p, model


def _pick(u, choose):
    if choose == "judge":
        return int(np.argmax(u["p"].to_numpy()))
    return int(np.lexsort((-u["coarse"].to_numpy(), -(u["inliers"] + u["z"]).to_numpy()))[0])     # the pipeline's choice


def _row(data, photo, stage, c, n_queries):
    if c is None:
        return dict(data=data, photo=photo, stage=stage, pts=0, err_sigma=np.nan, inliers=0.0, p=np.nan, n_queries=n_queries, x=np.nan, y=np.nan)
    return dict(data=data, photo=photo, stage=stage, pts=int(c["pts"]) if stage != "skip" else 0, err_sigma=float(c["err_sigma"]),
                inliers=float(c["inliers"]), p=float(c["p"]), n_queries=n_queries, x=float(c.get("x", np.nan)), y=float(c.get("y", np.nan)))


def _passes(c, tau, override, bar):
    if tau is None:
        return c["inliers"] >= bar
    return c["p"] >= tau or (override is not None and c["inliers"] >= override)


def decide(table, p, tau, choose="pipeline", override=None, jetson=False, m=None, cap=10, extra_order=SECOND_PASS, bars=(8, 11), min_choice=1, early_tau=None, wide_tau=None):
    """One row per photo: stage (std6 / wide / skip), pts, err_sigma, inliers and p of the answer (or of the last choice), n_queries.
    tau=None -> the plain inlier bars `bars` (std, wide) instead of the judge (the baseline). m: only candidates of DINO rank < m (Jetson: 5).
    jetson=True: after the 6 standard queries, the extra queries of `extra_order` one at a time up to `cap` queries in all, deciding after each.
    min_choice: candidates with fewer inliers are never chosen (the pipeline: 1). With the judge, 3: the judge has no opinion under 3 inliers
    (P = 0), and letting a 2-inlier, high-z candidate win the choice would only turn a photo the judge can answer into a skip.
    early_tau: the standard queries run one at a time (STANDARD order) and the photo is answered as soon as the choice so far has P >= early_tau
    (saves Jetson time: each query not run is ~2.2 s); None = all 6 standard queries first.
    wide_tau: the cut after the standard queries is max(tau, wide_tau) (more queries give a look-alike more chances; today's bars do the same, 8 -> 11)."""
    t = table.assign(p=p)
    if m is not None:
        t = t[t["rank"] < m]
    out = []
    std_keys = {(round(h, 2), round(s, 2)) for h, s in STANDARD}
    tau_w = tau if (tau is None or wide_tau is None) else max(tau, wide_tau)
    for (data, photo), g in t.groupby(["data", "photo"], sort=False):
        g = g[g["inliers"] >= max(min_choice, 1)]
        std = g[g["qkey"].isin(std_keys)]
        last = None
        if early_tau is not None and tau is not None:
            run, hit = set(), None
            for k, key in enumerate([(round(h, 2), round(s_, 2)) for h, s_ in STANDARD][:-1]):
                run.add(key)
                u = g[g["qkey"].isin(run)]
                if len(u):
                    c = u.iloc[_pick(u, choose)]
                    if c["p"] >= early_tau:
                        hit = (c, k + 1); break
            if hit is not None:
                out.append(_row(data, photo, "early", hit[0], hit[1])); continue
        if len(std):
            c = std.iloc[_pick(std, choose)]
            last = c
            if _passes(c, tau, override, bars[0]):
                out.append(_row(data, photo, "std6", c, 6)); continue
        if jetson:
            run = set(std_keys)
            extra = [(round(h, 2), round(s, 2)) for h, s in extra_order if (round(h, 2), round(s, 2)) not in std_keys]
            done = False
            for key in extra:
                if len(run) >= cap:
                    break
                run.add(key)
                u = g[g["qkey"].isin(run)]
                if not len(u):
                    continue
                c = u.iloc[_pick(u, choose)]
                last = c
                if _passes(c, tau_w, override, bars[1]):
                    out.append(_row(data, photo, "wide", c, len(run))); done = True; break
            if not done:
                out.append(_row(data, photo, "skip", last, len(run)))
            continue
        if len(g):
            c = g.iloc[_pick(g, choose)]
            last = c
            if _passes(c, tau_w, override, bars[1]):
                out.append(_row(data, photo, "wide", c, 27)); continue
        out.append(_row(data, photo, "skip", last, 27))
    return pd.DataFrame(out)


def baseline(table, bar=8, escalation_bar=11, **kwargs):
    t = table.assign(p=np.nan)
    return decide(t, t["p"].to_numpy(), None, bars=(bar, escalation_bar), **kwargs)


def summary(result):
    """Score /60, wrong answers, answered share and mean queries per data set."""
    g = result.groupby("data", sort=False)
    return pd.DataFrame(dict(score=g["pts"].mean() * 6, wrong=g["pts"].apply(lambda s: int((s < 0).sum())),
                             answered=g["stage"].apply(lambda s: float((s != "skip").mean())), queries=g["n_queries"].mean(),
                             n=g["pts"].size(), n_answered=g["stage"].apply(lambda s: int((s != "skip").sum())))).round(3)


def scan_tau(table, p, taus, **kwargs):
    """decide() at every cut in taus -> one row per (tau, data set) with score and wrong answers."""
    rows = []
    for tau in taus:
        s = summary(decide(table, p, tau, **kwargs)).reset_index()
        s.insert(0, "tau", tau)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def smallest_safe_tau(scan, sets, margin_steps=1):
    """The smallest cut of a scan with 0 wrong answers on every set in `sets` and on every larger cut too (so the choice does not sit on an
    isolated lucky value), moved up by margin_steps grid steps as a safety margin."""
    taus = sorted(scan["tau"].unique())
    wrong = scan[scan["data"].isin(sets)].groupby("tau")["wrong"].sum()
    safe = [t for i, t in enumerate(taus) if all(wrong.get(u, 0) == 0 for u in taus[i:])]
    if not safe:
        return taus[-1]
    return taus[min(taus.index(safe[0]) + margin_steps, len(taus) - 1)]


TAUS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99]


def ltt_tau(scan, sets, alpha=0.005, delta=0.1):
    """Learn-then-Test style cut (Angelopoulos et al.; conformal risk control): walk the cuts from the strictest down and keep going while the
    one-sided (1 - delta) Clopper-Pearson upper bound of the wrong-answer rate among answered photos (pooled over `sets`) stays <= alpha; return
    the last cut that passed (fixed-sequence testing, so no multiple-testing correction is needed). With 0 wrong in n answers the bound is about
    -ln(delta) / n, so alpha = 0.5 % needs ~460 answers at 0 wrong: the rule also refuses cuts that are only 'lucky' on little data."""
    from scipy.stats import beta
    taus = sorted(scan["tau"].unique(), reverse=True)
    sub = scan[scan["data"].isin(sets)].groupby("tau")[["wrong", "n_answered"]].sum()
    chosen = taus[0]
    for tau in taus:
        w, n = int(sub.loc[tau, "wrong"]), int(sub.loc[tau, "n_answered"])
        upper = 1.0 if n == 0 else (1.0 if w >= n else float(beta.ppf(1 - delta, w + 1, n - w)))
        if upper > alpha:
            break
        chosen = tau
    return chosen


def held_out_probabilities(table, sets, features=FEATURES_CTX, kind="gbm"):
    """{held set: P(right) per row from the judge trained on the other sets}. The judges depend only on (sets, features, kind), so one call can feed
    every leave_one_set_out run that varies only the decision (cut rule, early exit, Jetson replay)."""
    return {held: judge_probability(table, [s for s in sets if s != held], features, kind)[0] for held in sets}


def leave_one_set_out(table, sets, features=FEATURES_CTX, kind="gbm", taus=TAUS, margin_steps=1, cut_rule="zero_wrong", alpha=0.005, tau_floor=None,
                      probs=None, **decide_kwargs):
    """probs: held_out_probabilities output (skips retraining). tau_floor: the cut is never below it (a guard on the Learn-then-Test cut, which on the Jetson candidates went as low as 0.8).
    For every set: train the judge on the other sets, pick the cut on them (smallest_safe_tau with a margin), and replay the held-out set.
    Returns (one row per held-out set: tau, score, wrong, queries; the per-photo decisions of every held-out set)."""
    rows, parts = [], []
    for held in sets:
        train = [s for s in sets if s != held]
        p = probs[held] if probs is not None else judge_probability(table, train, features, kind)[0]
        scan = scan_tau(table, p, taus, **decide_kwargs)
        tau = smallest_safe_tau(scan, train, margin_steps) if cut_rule == "zero_wrong" else ltt_tau(scan, train, alpha)
        if tau_floor is not None:
            tau = max(tau, tau_floor)
        result = decide(table, p, tau, **decide_kwargs)
        result = result[result["data"] == held]
        parts.append(result.assign(tau=tau))
        rows.append(dict(held=held, tau=tau, score=round(6 * result["pts"].mean(), 2), wrong=int((result["pts"] < 0).sum()),
                         answered=round(float((result["stage"] != "skip").mean()), 3), queries=round(result["n_queries"].mean(), 2)))
    return pd.DataFrame(rows), pd.concat(parts, ignore_index=True)


def paired_gain(base, new, sets):
    """'+g [lo, hi]' per set: paired bootstrap of new over base (per-photo decisions of decide / baseline)."""
    from train.checks.replay import paired_bootstrap
    b = base.set_index(["data", "photo"])["pts"]
    n = new.set_index(["data", "photo"])["pts"]
    out = {}
    for s in sets:
        bb = b.loc[s]
        m, lo, hi = paired_bootstrap(bb.to_numpy(), n.loc[s].reindex(bb.index).to_numpy())
        out[s] = "%+.2f [%+.2f, %+.2f]" % (m, lo, hi)
    return out


# The frozen Test_Iteration_7 rule (tests/Test_Iteration_7/notebooks/It4_Decision_Rule.ipynb), set on the five seen sets before the blind set.
FROZEN = dict(features=FEATURES_CTX, kind="gbm", alpha=0.005, tau_floor=0.9, wide_tau=0.95, override=12, min_choice=3)
FROZEN_VARIANTS = {"pc": dict(), "pc_early": dict(early_tau=0.98), "jetson": dict(jetson=True, m=5), "jetson_early": dict(jetson=True, m=5, early_tau=0.98)}


def freeze(table, sets, variant, rule=FROZEN, taus=TAUS):
    """The final judge (fitted on every labelled candidate of `sets`) and its cut for one decision variant: Learn-then-Test on the out-of-fold
    P of the same sets (5 folds grouped by photo), then the floor. Returns (model, tau)."""
    kw = dict(override=rule["override"], min_choice=rule["min_choice"], wide_tau=rule["wide_tau"], **FROZEN_VARIANTS[variant])
    p, model = judge_probability(table, sets, rule["features"], rule["kind"])
    tau = max(ltt_tau(scan_tau(table, p, taus, **kw), sets, rule["alpha"]), rule["tau_floor"])
    return model, tau


def judge_new(model, table, rule=FROZEN):
    """P(right) of a table the model never saw (0 under 3 inliers), as judge_probability gives it."""
    p = np.zeros(len(table))
    usable = (table["inliers"] >= 3).to_numpy()
    if usable.any():
        p[usable] = model.predict_proba(table.loc[usable, rule["features"]].to_numpy(float))[:, 1]
    return p


def decide_frozen(table, p, tau, variant, rule=FROZEN):
    return decide(table, p, tau, override=rule["override"], min_choice=rule["min_choice"], wide_tau=rule["wide_tau"], **FROZEN_VARIANTS[variant])
