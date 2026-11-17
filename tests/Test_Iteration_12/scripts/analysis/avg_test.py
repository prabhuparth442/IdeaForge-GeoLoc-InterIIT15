import sys, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/e/IdeaForge/work/repo"); sys.path.insert(0, "/mnt/e/IdeaForge/work/repo/src")
from train.checks import replay
D = "/mnt/e/IdeaForge/work/mega/rich_dump/"
def best_per_query(photo, gamma=1.0):
    out = []
    for j, q in enumerate(photo["queries"]):
        inl = np.asarray(q["inliers"], float).copy(); z = np.asarray(q["z"], float); sh = np.asarray(q["shifts"], float)
        if not (inl > 0).any(): out.append(None); continue
        s = np.where(inl > 0, inl + gamma * z, -1e9); k = int(np.lexsort((-q["peaks"][:, 0], -s))[0])
        out.append((q["peaks"][k, 1] + sh[k, 0], q["peaks"][k, 2] + sh[k, 1], inl[k], s[k]))
    return out
rows = []
for name in ["official150_clean", "official150_blur", "official150_invert", "nudt_valid100_clean"]:
    photos = replay.load_set(D + name)
    for pid, photo in photos.items():
        std = [replay.query_ids(photo, [k])[0] for k in replay.STANDARD]
        allq = list(range(len(photo["queries"])))
        bq = best_per_query(photo)
        cand = [(j, bq[j]) for j in std if bq[j] is not None]
        if not cand: continue
        j0, b0 = max(cand, key=lambda c: c[1][3])
        if b0[2] < 8: continue                     # answer by the bar only (the simple case)
        x0, y0 = b0[0], b0[1]; sig = photo["sigma"]
        e0 = replay.error_sigma(photo, x0, y0)
        for label, pool in (("6 standard queries", std), ("all queries", allq)):
            pts = [(bq[j][0], bq[j][1], bq[j][2]) for j in pool if bq[j] is not None and bq[j][2] >= 4 and np.hypot(bq[j][0] - x0, bq[j][1] - y0) <= 1.0 * sig]
            w = np.array([p[2] for p in pts]); xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
            xm, ym = (xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()
            xmed, ymed = np.median(xs), np.median(ys)
            rows.append(dict(set=name, pool=label, photo=pid, n_used=len(pts), e_chosen=e0, e_mean=replay.error_sigma(photo, xm, ym), e_median=replay.error_sigma(photo, xmed, ymed)))
R = pd.DataFrame(rows)
def pts(e): return np.where(e <= 1, 10, np.where(e <= 2, 6, np.where(e <= 4, 3, np.where(e <= 8, 0, -20))))
out = []
for (s, p), g in R.groupby(["set", "pool"]):
    out.append(dict(set=s, pool=p, answered_by_bar=len(g), queries_averaged=round(g.n_used.mean(), 1), within1_chosen=round((g.e_chosen <= 1).mean(), 2), within1_mean=round((g.e_mean <= 1).mean(), 2), within1_median=round((g.e_median <= 1).mean(), 2),
        median_err_sigma_chosen=round(g.e_chosen.median(), 2), median_err_sigma_mean=round(g.e_mean.median(), 2), points_chosen=round(pts(g.e_chosen.values).mean(), 2), points_mean=round(pts(g.e_mean.values).mean(), 2)))
print(pd.DataFrame(out).to_string(index=False))
