from common import *
print("=== PART 4: wrong answers and the bar (PV-X seed 0 unless noted) ===")
cols = ["filename", "alt", "inliers", "error_m", "sigma_m", "second_pass", "fit_rms_px", "fit_dpitch", "fit_droll", "pose_fit_move_m"]
for tag, runs in (("seed0", PVX), ("seed1", PVX1)):
    for s, p in runs.items():
        d = load(p); w = d[d.p == -20]
        print("\n%s %s: %d wrong of %d answered; wrong rows (inliers, err/sigma, height, second_pass, fit_rms):" % (tag, s, len(w), (d.status == "match").sum()))
        for _, r in w.iterrows():
            print("   inl %3d  err/sigma %5.1f  h %6.0f  second_pass %s  fit_rms %.2f" % (r.inliers, r.error_m / r.sigma_m, r.alt_x if "alt_x" in r else r.alt, r.second_pass, r.fit_rms_px))
print("\n--- answered with inliers below 20 (marginal): count, wrong, mean points, per set (seed 0) ---")
for s, p in PVX.items():
    d = load(p); a = d[d.status == "match"]
    for lo, hi in ((0, 12), (12, 16), (16, 20), (20, 40), (40, 10000)):
        b = a[(a.inliers >= lo) & (a.inliers < hi)]
        if len(b): print("  %-12s inliers [%d,%d): n=%3d wrong=%d  mean points %.2f   (score-weight: %.2f of 60)" % (s, lo, hi, len(b), (b.p == -20).sum(), b.p.mean(), 6 * b.p.sum() / len(d)))
print("\n--- upper bounds on the wrong rate (answered photos), Clopper-Pearson 95% ---")
from scipy.stats import beta
def cp(k, n): return (0 if k == 0 else beta.ppf(0.025, k, n - k + 1), 1 if k == n else beta.ppf(0.975, k + 1, n - k))
tot = [0, 0]
for s, p in PVX.items():
    d = load(p); n = int((d.status == "match").sum()); k = int((d.p == -20).sum()); lo, hi = cp(k, n); print("  %-12s %d/%d  rate %.3f  95%% CI [%.3f, %.3f]" % (s, k, n, k / n, lo, hi))
d = pd.concat([load(PVX["work332"]), load(PVX["locked168"])]); n = int((d.status == "match").sum()); k = int((d.p == -20).sum()); lo, hi = cp(k, n); print("  highalt(dev+test) %d/%d rate %.4f CI [%.4f, %.4f]" % (k, n, k / n, lo, hi))
print("\n--- selection of bar: where did a bar of 12 come from? real pairs of runs (same code, bar only) ---")
S = "/tmp/claude-1000/-home-voidness-Storage-Synced-Folders-InterIIT-Tech-15/7794e0a1-ad20-440a-9aae-aa3d00586cd6/scratchpad/pv/"
for s, a, b in (("official", "pvs_official_g0b20", "pvs_official_g0b12"), ("nudt", "pvs_nudt_g0b20", "pvs_nudt_g0b12")):
    da, db = load(S + a + "/details.csv"), load(S + b + "/details.csv")
    print("  %-9s bar20 %.2f (ans %d wrong %d)   bar12 %.2f (ans %d wrong %d)" % (s, 6 * da.p.mean(), (da.status == 'match').sum(), (da.p == -20).sum(), 6 * db.p.mean(), (db.status == 'match').sum(), (db.p == -20).sum()))
print("  working332: no bar-12-vs-20 pair with the same settings exists (bar 16 vs 20 without fit: 42.31 vs 42.04; bar 24: 42.02 on a partial run of 277 photos).")
print("\n--- how much of the highalt wrong-answer count is protected by the bar? answers by inlier count bin for the 3 highalt runs ---")
a = pd.concat([load(PVX["work332"]), load(PVX["locked168"])]); a = a[a.status == "match"]
print(a.groupby(pd.cut(a.inliers, [0, 8, 12, 16, 20, 30, 50, 100, 1000])).agg(n=("p", "size"), wrong=("p", lambda v: (v == -20).sum()), mean_pts=("p", "mean")).round(2).to_string())
