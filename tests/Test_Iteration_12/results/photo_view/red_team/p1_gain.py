from common import *
print("=== PART 1: paired gain PV-X (seed 0) minus base, per set; CI by photo and by spatial cluster ===")
allp = []
for s in BASE:
    d = paired(s); allp.append(d)
    g6 = 6 * d.gain.mean()
    pb = 6 * np.percentile(photo_boot(d), [2.5, 97.5]); cb = 6 * np.percentile(cluster_boot(d), [2.5, 97.5])
    print("%-12s n=%3d clusters=%2d  base %.2f PVX %.2f gain %+.2f  photo-boot 95%% [%+.2f,%+.2f]  cluster-boot 95%% [%+.2f,%+.2f]  frac photos better %.2f worse %.2f same %.2f  P(gain<=0 | cluster boot) %.4f" % (
        s, len(d), d.cluster.nunique(), 6 * d.p_a.mean(), 6 * d.p_b.mean(), g6, *pb, *cb, (d.gain > 0).mean(), (d.gain < 0).mean(), (d.gain == 0).mean(), (cluster_boot(d) <= 0).mean()))
al = pd.concat(allp)
print("\n--- decomposition of gain (points per photo x6 /n), by transition ---")
for s in BASE:
    d = paired(s)
    ans_a = d.status_a == "match"; ans_b = d.status_b == "match"
    def part(m): return 6 * d.gain[m].sum() / len(d)
    print("%-12s skip->answer %+.2f (n=%d) | both answered, precision change %+.2f (n=%d) | answer->skip %+.2f (n=%d) | wrong answers base %d -> %d, points from wrong changes %+.2f" % (
        s, part(~ans_a & ans_b), (~ans_a & ans_b).sum(), part(ans_a & ans_b), (ans_a & ans_b).sum(), part(ans_a & ~ans_b), (ans_a & ~ans_b).sum(), (d.p_a == -20).sum(), (d.p_b == -20).sum(), part((d.p_a == -20) | (d.p_b == -20))))
print("\n--- concentration of gain ---")
for s in BASE:
    d = paired(s).sort_values("gain", ascending=False); pos = d.gain[d.gain > 0].sum(); tot = d.gain.sum()
    k = int(np.ceil(0.1 * len(d)))
    cs = d.gain.cumsum().values
    n50 = int(np.argmax(cs >= 0.5 * tot)) + 1
    print("%-12s positives %.0f pts total, negatives %.0f pts; top10%% photos (%d) give %.0f%% of net gain; photos needed for 50%% of net gain: %d (%.0f%% of set); largest single-cluster share of net gain %.0f%%" % (
        s, pos, d.gain[d.gain < 0].sum(), k, 100 * d.gain.iloc[:k].sum() / tot, n50, 100 * n50 / len(d), 100 * d.groupby("cluster").gain.sum().max() / tot))
print("\n--- gain by height band (points/photo x6 = contribution to score if all photos were that band) ---")
al["hb"] = band(al.alt)
print(al.groupby(["set", "hb"], observed=True).agg(n=("gain", "size"), base=("p_a", lambda v: 6 * v.mean()), pvx=("p_b", lambda v: 6 * v.mean()), gain=("gain", lambda v: 6 * v.mean()), wrong_base=("p_a", lambda v: (v == -20).sum()), wrong_pvx=("p_b", lambda v: (v == -20).sum())).round(2).to_string())
print("\n--- leave-top-cluster-out: gain on each set when the single best cluster is removed ---")
for s in BASE:
    d = paired(s); cs = d.groupby("cluster").gain.agg(["sum", "count"]); best = cs["sum"].idxmax()
    r = d[d.cluster != best]
    print("%-12s remove cluster %s (n=%d): gain %+.2f -> %+.2f" % (s, best, cs.loc[best, "count"], 6 * d.gain.mean(), 6 * r.gain.mean()))
print("\n--- seed 1 (hint jitter) for the dev sets, paired with base ---")
for s in PVX1:
    d = paired(s); d1 = paired(s, (BASE[s], PVX1[s]))
    print("%-12s seed0 gain %+.2f  seed1 gain %+.2f   per-photo SD of seed0-seed1 point difference %.2f; photos whose points differ between seeds %d/%d" % (s, 6 * d.gain.mean(), 6 * d1.gain.mean(), (d.p_b - d1.p_b).std(), ((d.p_b - d1.p_b) != 0).sum(), len(d)))
