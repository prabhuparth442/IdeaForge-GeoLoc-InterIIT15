from common import *
print("=== PART 5: locked test (unseen cells) versus dev, and a-priori prediction check ===")
d = paired("work332"); t = paired("locked168")
d["hb"] = band(d.alt); t["hb"] = band(t.alt)
print("test band counts:", t.hb.value_counts().sort_index().to_dict()); print("dev band counts:", d.hb.value_counts().sort_index().to_dict())
bm = d.groupby("hb", observed=True).agg(pvx=("p_b", "mean"), base=("p_a", "mean"))
w = t.hb.value_counts(normalize=True)
pred_pvx = 6 * sum(bm.pvx[b] * w[b] for b in w.index if b in bm.index); pred_base = 6 * sum(bm.base[b] * w[b] for b in w.index)
print("dev band-means reweighted to the test height mix: base %.2f, PV-X %.2f (gain %+.2f)  | observed test: base %.2f PV-X %.2f (gain %+.2f)" % (pred_base, pred_pvx, pred_pvx - pred_base, 6 * t.p_a.mean(), 6 * t.p_b.mean(), 6 * t.gain.mean()))
# cluster bootstrap of the observed-minus-predicted PV-X
rng = np.random.default_rng(7)
def boot_obs_pred(B=4000):
    out = []; dc = d.cluster.unique(); tc = t.cluster.unique()
    for _ in range(B):
        dd = pd.concat([d[d.cluster == c] for c in rng.choice(dc, len(dc))]); tt = pd.concat([t[t.cluster == c] for c in rng.choice(tc, len(tc))])
        bm_ = dd.groupby("hb", observed=True).p_b.mean(); w_ = tt.hb.value_counts(normalize=True)
        pred = sum(bm_.get(b, np.nan) * w_[b] for b in w_.index)
        out.append(6 * (tt.p_b.mean() - pred))
    return np.nanpercentile(out, [2.5, 50, 97.5])
print("observed test PV-X minus dev-predicted (same height mix): %.2f, cluster-boot [2.5,50,97.5] = %s" % (6 * t.p_b.mean() - pred_pvx, boot_obs_pred().round(2)))
print("Test PV-X score cluster-boot 95%% CI: [%.2f, %.2f]; base: [%.2f, %.2f]" % (*6 * np.percentile(cluster_boot(t, "p_b"), [2.5, 97.5]), *6 * np.percentile(cluster_boot(t, "p_a"), [2.5, 97.5])))
print("Dev PV-X score cluster-boot 95%% CI: [%.2f, %.2f]" % tuple(6 * np.percentile(cluster_boot(d, "p_b"), [2.5, 97.5])))
print("\n--- how independent is the test split in space? nearest DEV photo to each TEST photo, metres ---")
from scipy.spatial import cKDTree
tree = cKDTree(d[["x", "y"]].values); dist, _ = tree.query(t[["x", "y"]].values)
print("test->nearest dev: median %.0f m, 10%% %.0f m, min %.0f m; share within 100 m: %.2f within 300 m: %.2f" % (np.median(dist), np.percentile(dist, 10), dist.min(), (dist < 100).mean(), (dist < 300).mean()))
# dev->nearest dev in other cell
dd = d[["x", "y"]].values; cl = d.cluster.values; nn = []
for i in range(len(d)):
    m = cl != cl[i]; nn.append(np.min(np.hypot(*(dd[m] - dd[i]).T)))
print("dev->nearest dev photo of another cell: median %.0f m" % np.median(nn))
print("\n--- test per-band PV-X vs dev per-band PV-X (points x6) ---")
x = pd.concat([d.assign(s="dev"), t.assign(s="test")]); print(x.groupby(["hb", "s"], observed=True).agg(n=("p_b", "size"), base=("p_a", lambda v: 6 * v.mean()), pvx=("p_b", lambda v: 6 * v.mean())).round(2).unstack().to_string())
