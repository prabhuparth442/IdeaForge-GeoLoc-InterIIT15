import os, sys, glob, numpy as np, pandas as pd
B = "/home/voidness/Storage/Synced Folders/InterIIT Tech 15/IdeaForge-GeoLoc-InterIIT15/tests/Test_Iteration_12/results/"
RUNS = B + "photo_view/runs/"
sys.path.insert(0, B + "photo_view")
from score import pts
RNG = np.random.default_rng(12345)

def meta():
    hs = pd.read_csv(B + "highalt_split.csv")
    out = []
    for name, f in (("work332", "coarse_averaging/inputs/meta_highalt_dev332.csv"), ("official500", "coarse_averaging/inputs/meta_official500.csv"), ("nudt300", "coarse_averaging/inputs/meta_nudt300.csv")):
        m = pd.read_csv(B + f); m["set"] = name; out.append(m)
    t = hs[hs.split == "test"].copy(); t["set"] = "locked168"
    out.append(t[["filename", "lat_gtruth", "lon_gtruth", "alt", "set"]])
    m = pd.concat(out, ignore_index=True)[["filename", "lat_gtruth", "lon_gtruth", "alt", "set"]]
    m = m.merge(hs[["filename", "cell", "band"]], on="filename", how="left")
    # spatial clusters: highalt = the dataset's own cell (the split is by cell); official / nudt = k-means on metres, ~25 clusters
    from scipy.cluster.vq import kmeans2
    m["cluster"] = m["cell"]
    for s, k in (("official500", 25), ("nudt300", 15)):
        i = m["set"] == s
        x = (m.loc[i, "lon_gtruth"] - m.loc[i, "lon_gtruth"].mean()) * 111320 * np.cos(np.radians(m.loc[i, "lat_gtruth"].mean()))
        y = (m.loc[i, "lat_gtruth"] - m.loc[i, "lat_gtruth"].mean()) * 111320
        np.random.seed(0)
        _, lab = kmeans2(np.c_[x, y], k, minit="++", seed=0)
        m.loc[i, "cluster"] = [s[:3] + "_%02d" % l for l in lab]
    m["cluster"] = m["cluster"].astype(str)
    m["x"] = (m.lon_gtruth - 71.0) * 111320 * np.cos(np.radians(27.0)); m["y"] = (m.lat_gtruth - 27.0) * 111320
    return m
M = meta()

def load(path, name=None):
    d = pd.read_csv(path); d["p"] = d.apply(pts, axis=1)
    return d.merge(M, on="filename", how="left")

def band(h):
    return pd.cut(h, [0, 351, 700, 1000, 1300, 1600, 2100], labels=["<351", "351-700", "700-1000", "1000-1300", "1300-1600", "1600+"])

BASE = {"work332": B + "native_fine/details_xoftr800.csv", "official500": B + "native_fine/old_details_official_xoftr800.csv",
        "nudt300": B + "native_fine/old_details_nudt_xoftr800.csv", "locked168": B + "native_fine/test_details_base.csv"}
PVX = {"work332": RUNS + "pv3_dev_x0.csv", "official500": RUNS + "pv3_official_x0.csv", "nudt300": RUNS + "pv3_nudt_x0.csv", "locked168": RUNS + "pvx_test_t0.csv"}
PVX1 = {"work332": RUNS + "pvx_dev_s1.csv", "official500": RUNS + "pvx_official_s1.csv", "nudt300": RUNS + "pvx_nudt_s1.csv"}

def paired(s, other=None):
    a = load(BASE[s] if other is None else other[0]); b = load(PVX[s] if other is None else other[1])
    d = a[["filename", "p", "status", "inliers"]].merge(b[["filename", "p", "status", "inliers", "error_m", "sigma_m", "alt", "set", "cluster", "band", "x", "y"]], on="filename", suffixes=("_a", "_b"))
    d["gain"] = d.p_b - d.p_a
    return d

def cluster_boot(d, col="gain", B=10000, seed=1):
    rng = np.random.default_rng(seed)
    g = d.groupby("cluster")[col].agg(["sum", "count"]); s = g["sum"].values; n = g["count"].values
    k = len(g); idx = rng.integers(0, k, size=(B, k))
    return (s[idx].sum(1) / n[idx].sum(1))

def photo_boot(d, col="gain", B=10000, seed=2):
    rng = np.random.default_rng(seed); v = d[col].values
    return v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
