from common import *
print("=== PART 6: pooled highalt (dev + locked test, 500 photos, 96 cells) and fit usage ===")
d = pd.concat([paired("work332"), paired("locked168")])
print("pooled highalt: base %.2f PV-X %.2f gain %+.2f cluster-boot95 [%+.2f,%+.2f]; PV-X level CI [%.2f,%.2f]" % (6 * d.p_a.mean(), 6 * d.p_b.mean(), 6 * d.gain.mean(), *6 * np.percentile(cluster_boot(d), [2.5, 97.5]), *6 * np.percentile(cluster_boot(d, "p_b"), [2.5, 97.5])))
print("\n--- pose-fit usage among answered photos (PV-X seed 0) and points by usage ---")
for s, p in PVX.items():
    x = load(p); a = x[x.status == "match"]
    u = a.pose_fit_used.astype(bool)
    print("  %-12s fit used %d/%d (dropped by the 3-sigma move rule or no matches: %d); mean points fitted %.2f vs not fitted %.2f" % (s, u.sum(), len(a), (~u).sum(), a.p[u].mean(), a.p[~u].mean() if (~u).any() else float('nan')))
print("\n--- sensitivity of the PV-X score to one wrong answer (points of the 60 scale) ---")
for s, n in (("work332", 332), ("official500", 500), ("nudt300", 300), ("locked168", 168)):
    print("  %-12s one wrong answer instead of a typical +9 answer = %.2f of 60; instead of a skip = %.2f" % (s, 6 * 29 / n, 6 * 20 / n))
print("\n--- hindsight bar 16 (drop answers with 12..15 inliers) on PV-X runs, in-sample: change of score ---")
for s, p in PVX.items():
    x = load(p); m = (x.status == "match") & (x.inliers >= 12) & (x.inliers < 16)
    print("  %-12s drops %d answers, %d of them wrong; score change %+.2f" % (s, m.sum(), (x.p[m] == -20).sum(), -6 * x.p[m].sum() / len(x)))
