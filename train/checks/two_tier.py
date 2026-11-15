"""Block: the two-tier answer rule of Test_Iteration_8 (no learned model, three numbers set on NUDT only), plus the label-free null-calibrated cut
that was tried and dropped. Works on the T7 signal tables (train/checks/decision_signals.py + candidate_judge.add_features / add_context).
No GPU, no plotting, no multi-query agreement.

Why two tiers (the logic, from T8 It2 before any grid):
- On official almost every choice with >= 4 inliers is right (a bar of 4 makes 0 wrong answers on the try half); on NUDT the same bar makes 10-15
  wrong answers per set (look-alike city blocks). So a lower inlier bar cannot be used on its own: whether it is safe depends on the data.
- At 5-11 inliers the wrong choices look the same on both data sets: DINO z ~1.2 and a stronger DINO place elsewhere in the same query
  (zgap < 0), while right choices have z >= ~2 (official: 95 % above 1.95) and are DINO's top place of their query. z and zgap are relative to the
  photo's own candidates, so they should mean the same on a new map.
- Tier 1: inliers >= i_hi -> answer (as today's bar). Tier 2: inliers >= a_low AND z >= z0 AND zgap >= g0 -> answer.

two_tier_p(table, rule)              per candidate: 0.95 if the rule answers it, 1.0 if it also passes the early-exit level, else 0
decide_two_tier(table, rule, variant) the per-photo replay (candidate_judge.decide): "pc" (6 standard queries, then all 27) or "jetson"
                                     (M = 5, extra queries one at a time, cap 10), optional early exit after any standard query
null_pvalues(table, score)           (dropped idea) conformal p-value of each photo's choice against the strongest far candidate of the OTHER photos
                                     of the same set: a cut that adapts to how look-alike-rich a map is without labels
"""
import numpy as np
import pandas as pd

from train.checks.candidate_judge import STANDARD, decide

GRID = dict(a_low=[3, 4, 5, 6], z0=[1.8, 2.0, 2.2, 2.4], g0=[0.0, 0.25, 0.5], i_hi=[8, 10, 12], dz_wide=[0.0, 0.3], z_hi=[None, 1.3])


def two_tier_p(table, a_low, z0, g0, i_hi, dz_wide=0.0, z_hi=None, early=None):
    """dz_wide: the z floor is raised by this much after the standard queries. z_hi: tier 1 also needs z >= z_hi unless inliers >= 15 (T6's veto).
    early: inliers at which a standard-query choice is answered at once (early exit)."""
    I, z, g = table["inliers"].to_numpy(), table["z"].to_numpy(), table["zgap"].to_numpy()
    hi = I >= i_hi
    if z_hi is not None:
        hi &= (z >= z_hi) | (I >= 15)
    std_ok = hi | ((I >= a_low) & (z >= z0) & (g >= g0))
    wide_ok = hi | ((I >= a_low) & (z >= z0 + dz_wide) & (g >= g0))
    p = np.where(wide_ok, 0.97, np.where(std_ok, 0.95, 0.0))
    if early is not None:
        p = np.where(wide_ok & (I >= early), 1.0, p)
    return p


def decide_two_tier(table, rule, variant="pc"):
    """rule: dict of two_tier_p arguments. variant: "pc" or "jetson". Returns candidate_judge.decide output (one row per photo)."""
    p = two_tier_p(table, **rule)
    kw = dict(min_choice=3, override=None, wide_tau=0.96)
    if variant == "jetson":
        kw.update(jetson=True, m=5)
    if rule.get("early") is not None:
        kw.update(early_tau=0.99)
    return decide(table, p, 0.9, **kw)


def null_pvalues(table, score="inliers"):
    """For each photo and stage (std = 6 standard queries, all = 27): the pipeline choice, and as its null value the best `score` among the
    candidates > 4 sigma from the choice. p = (1 + #other photos of the set whose null value >= this choice's score) / n. Label-free."""
    std_keys = {(round(h, 2), round(s, 2)) for h, s in STANDARD}
    rows = []
    for (d, p), g in table.groupby(["data", "photo"], sort=False):
        g = g[g["inliers"] >= 3]
        for stage in ("std", "all"):
            u = g[g["qkey"].isin(std_keys)] if stage == "std" else g
            if not len(u):
                rows.append(dict(data=d, photo=p, stage=stage, s=np.nan, null=-np.inf, pts=0))
                continue
            c = u.iloc[int(np.lexsort((-u["coarse"].to_numpy(), -(u["inliers"] + u["z"]).to_numpy()))[0])]
            far = (np.hypot(u["x"] - c["x"], u["y"] - c["y"]) > 4 * c["sigma"]).to_numpy()
            rows.append(dict(data=d, photo=p, stage=stage, s=c[score], null=u[score].to_numpy()[far].max() if far.any() else -np.inf, pts=c["pts"]))
    pp = pd.DataFrame(rows)
    pp["pv"] = 1.0
    for _, g in pp.groupby(["data", "stage"]):
        nulls, s = g["null"].to_numpy(), g["s"].to_numpy()
        ge = (nulls[None, :] >= s[:, None]).sum(1) - (nulls >= s)
        pp.loc[g.index, "pv"] = np.where(np.isnan(s), 1.0, (1 + ge) / len(g))
    return pp


def null_decide(pp, eps_std, eps_all):
    """PC replay of null_pvalues: answer the standard-query choice if p <= eps_std, else the 27-query choice if p <= eps_all."""
    w = pp.pivot_table(index=["data", "photo"], columns="stage", values=["pv", "pts"], aggfunc="first")
    a_std = (w[("pv", "std")] <= eps_std).to_numpy()
    a_all = ~a_std & (w[("pv", "all")] <= eps_all).to_numpy()
    pts = np.where(a_std, w[("pts", "std")], np.where(a_all, w[("pts", "all")], 0))
    return pd.DataFrame(dict(pts=pts, stage=np.where(a_std, "std6", np.where(a_all, "wide", "skip")), n_queries=0), index=w.index).reset_index()
