"""Block: honest hold-outs for Test_Iteration_8. No GPU, no plotting.

Why: the T7 judge (and earlier rules) were built while looking at every official photo, so official was never truly held out at the design
level. Here official is cut once into a TRY half (used to check a method while building it) and a TEST half (scored once, after the method is
written down in a pre-registration file). NUDT sets are the design data.

The split is by place, not by photo: the 379 official photos are grouped into 12 spatial blocks (k-means on the true position, seed 0), and the
blocks are shared out so that both halves have the same number of photos in each height band (100-125, 125-175, 175-275, 275-351 m). Nothing
about the outcome (inliers, errors) is used. Photos of one place stay in one half, so a method cannot learn one street's look-alikes on TRY and
be rewarded for it on TEST. (Footprints of high photos are wide, so the halves still see neighbouring ground; the map itself is shared.)

official_meta(dump_dir)          photo, true x / y, height from the T6 rich dumps
official_split(meta)             adds band, block, half ("try" / "test")
load_t8_table(root, locked=True, blind=False)
                                 the T7 signal tables with official cut into official_try / official_test (the saved split); locked=True drops
                                 official_test, blind=True adds the T8 blind NUDT draw (nudt707). Only It5 (after pre-registration) unlocks.
bar_frontier / versus_frontier   plain inlier bars as the "answer more" reference: a method is compared with the bar that answers as many photos
attitude_frame / with_attitude   signed errors of answered official photos, and the points after removing a fitted attitude bias (pose_bias)
answer_change(base, new)         where a method's gain comes from: answers added (right / wrong), answers removed, precision on photos both
                                 answer, wrong answers; per data set. Answers "really improve" when the added answers carry the gain and the
                                 wrong count does not rise.
"""
import glob
import itertools
import os

import numpy as np
import pandas as pd

BANDS = [0, 125, 175, 275, 400]
BAND_LABELS = ["100-125", "125-175", "175-275", "275-351"]


def official_meta(dump_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "photo_*.npz"))):
        d = np.load(f, allow_pickle=True)
        rows.append(dict(photo=int(os.path.basename(f)[6:-4]), x=float(d["truth"][0]), y=float(d["truth"][1]), h=float(d["height"])))
    return pd.DataFrame(rows).sort_values("photo").reset_index(drop=True)


def official_split(meta, n_blocks=12, seed=0):
    from sklearn.cluster import KMeans
    m = meta.copy()
    m["band"] = pd.cut(m["h"], BANDS, labels=BAND_LABELS).astype(str)
    m["block"] = KMeans(n_blocks, random_state=seed, n_init=10).fit(m[["x", "y"]].to_numpy()).labels_
    ct = pd.crosstab(m["block"], m["band"]).reindex(columns=BAND_LABELS, fill_value=0).to_numpy()
    total = ct.sum(0)
    best = None
    for bits in itertools.product([0, 1], repeat=n_blocks - 1):
        sel = np.array((1,) + bits, bool)                      # block 0 always in TRY (the two halves are symmetric)
        cost = np.abs(ct[sel].sum(0) - total / 2).sum()
        if best is None or cost < best[0]:
            best = (cost, sel)
    m["half"] = np.where(best[1][m["block"].to_numpy()], "try", "test")
    return m


def answer_change(base, new):
    """base / new: per-photo decisions (data, photo, stage, pts). One row per data set."""
    b = base.set_index(["data", "photo"])
    n = new.set_index(["data", "photo"]).reindex(b.index)
    rows = []
    for data in b.index.get_level_values(0).unique():
        bb, nn = b.loc[data], n.loc[data]
        ba, na = bb["stage"] != "skip", nn["stage"] != "skip"
        added, removed, both = na & ~ba, ba & ~na, ba & na
        rows.append(dict(data=data, n=len(bb), gain=round(6 * (nn["pts"].mean() - bb["pts"].mean()), 2),
                         added=int(added.sum()), added_right=int((nn["pts"][added] > 0).sum()), added_wrong=int((nn["pts"][added] < 0).sum()),
                         added_pts=int(nn["pts"][added].sum()), removed=int(removed.sum()), removed_pts=int(bb["pts"][removed].sum()),
                         removed_wrong=int((bb["pts"][removed] < 0).sum()), both_pts_change=int((nn["pts"][both] - bb["pts"][both]).sum()),
                         wrong_base=int((bb["pts"] < 0).sum()), wrong_new=int((nn["pts"] < 0).sum())))
    return pd.DataFrame(rows)


def bar_frontier(table, bars=range(3, 16)):
    """Plain inlier bars (PC: bar b on the 6 standard queries, b + 3 on all 27; Jetson: bar b, second pass, cap 10, M = 5) on every set: score,
    wrong, answered count. The reference curve 'answering more' moves along."""
    from train.checks.candidate_judge import baseline, summary
    rows = []
    for b in bars:
        for var, kw, bw in (("pc", dict(min_choice=3), b + 3), ("jetson", dict(jetson=True, m=5), b)):
            s = summary(baseline(table, b, bw, **kw))
            for d in s.index:
                rows.append(dict(var=var, bar=b, data=d, score=s.loc[d, "score"], wrong=int(s.loc[d, "wrong"]), n_answered=int(s.loc[d, "n_answered"])))
    return pd.DataFrame(rows)


def versus_frontier(result, frontier, var):
    """For each set: the method's score, and the plain bar answering the closest number of photos (its score and wrong answers). A method that
    only 'answers more' sits on that curve; one that judges better sits above it."""
    from train.checks.candidate_judge import summary
    s = summary(result)
    rows = []
    for d in s.index:
        f = frontier[(frontier["var"] == var) & (frontier["data"] == d)]
        m = f.loc[(f["n_answered"] - s.loc[d, "n_answered"]).abs().idxmin()]
        rows.append(dict(data=d, score=s.loc[d, "score"], wrong=int(s.loc[d, "wrong"]), n_answered=int(s.loc[d, "n_answered"]), queries=s.loc[d, "queries"],
                         same_n_bar=int(m["bar"]), bar_score=m["score"], bar_wrong=int(m["wrong"]), above_bar=round(s.loc[d, "score"] - m["score"], 2)))
    return pd.DataFrame(rows)


def attitude_frame(result, dump_dir):
    """Answered official photos of a decide() result with the signed error (ex, ey, metres) and the attitude, for train/checks/pose_bias.py."""
    rows = []
    for _, r in result[result["stage"] != "skip"].iterrows():
        d = np.load(os.path.join(dump_dir, "photo_%d.npz" % r["photo"]), allow_pickle=True)
        rows.append(dict(data=r["data"], photo=r["photo"], ex=r["x"] - float(d["truth"][0]), ey=r["y"] - float(d["truth"][1]), err_sigma=r["err_sigma"],
                         sigma=float(d["sigma"]), h=float(d["height"]), yaw=float(d["yaw"]), pitch=float(d["pitch"]), roll=float(d["roll"])))
    return pd.DataFrame(rows)


def with_attitude(result, frame, model):
    """result with pts / err_sigma recomputed after removing the fitted attitude bias (pose_bias.apply) on the photos in `frame`."""
    from train.checks import pose_bias
    from train.checks.candidate_judge import points_array
    ex, ey = pose_bias.apply(frame, model)
    err = pd.Series(np.hypot(ex, ey) / frame["sigma"].to_numpy(), index=pd.MultiIndex.from_frame(frame[["data", "photo"]]))
    out = result.copy()
    key = pd.MultiIndex.from_frame(out[["data", "photo"]])
    hit = key.isin(err.index)
    out.loc[hit, "err_sigma"] = err.reindex(key[hit]).to_numpy()
    out.loc[hit, "pts"] = points_array(out.loc[hit, "err_sigma"].to_numpy())
    return out


NUDT_DESIGN = ["nudt101", "nudt202", "nudt404", "nudt505", "nudt606"]


def load_t8_table(root, locked=True, blind=False):
    """All signal tables with features and query-local context, official split by the saved T8 split."""
    from train.checks.candidate_judge import add_context, add_features
    root = str(root)
    sets = ["official"] + NUDT_DESIGN
    parts = [pd.read_csv(os.path.join(root, "tests/Test_Iteration_7/results/signals/%s.csv.gz" % s)) for s in sets]
    if blind:
        parts.append(pd.read_csv(os.path.join(root, "tests/Test_Iteration_8/results/signals/nudt707.csv.gz")))
    t = pd.concat(parts, ignore_index=True)
    split = pd.read_csv(os.path.join(root, "tests/Test_Iteration_8/results/official_split.csv"))
    half = dict(zip(split["photo"], split["half"]))
    off = t["data"] == "official"
    t.loc[off, "data"] = ["official_" + half[p] for p in t.loc[off, "photo"]]
    if locked:
        t = t[t["data"] != "official_test"]
    return add_context(add_features(t.reset_index(drop=True))).reset_index(drop=True)
