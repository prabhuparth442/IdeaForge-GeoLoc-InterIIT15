"""Scores red-team runs. usage: rt_score.py RUNS_DIR [names...]. Recomputes sigma from the ORIGINAL metadata (metadata_orig.csv next to the dataset) so height-bias arms are scored fairly.
Prints score/60, answered, wrong, mean seconds, peak GPU MB; null runs: false-answer count among photos whose truth is outside the 5.5 km disc."""
import sys, os, glob, math, re
import numpy as np, pandas as pd
runs = sys.argv[1]; names = sys.argv[2:] or sorted(os.path.basename(p)[len("out_rt_"):] for p in glob.glob(os.path.join(runs, "out_rt_*")))
def pts(e, s):
    return 10 if e <= s else 6 if e <= 2*s else 3 if e <= 4*s else 0 if e <= 8*s else -20
for n in names:
    p = os.path.join(runs, "out_rt_" + n, "details.csv")
    if not os.path.exists(p): continue
    d = pd.read_csv(p)
    op = os.path.join(sys.argv[1], "rt", "data", re.sub(r"^(base_|r2p1_|r2wide_)", "", n).replace("_prior1", "").replace("_nofit", ""), "metadata_orig.csv")
    if os.path.exists(op) and not n.startswith("alt"):
        o = pd.read_csv(op).set_index("filename"); sg = np.sqrt(8.0**2 + (3*o.alt/o.cam_fx)**2); d["sigma_m"] = d.filename.map(sg).fillna(d.sigma_m)
    m = d.status == "match"
    d["p"] = [pts(r.error_m, r.sigma_m) if r.status == "match" else 0 for r in d.itertuples()]
    print("%-18s n %3d score %5.2f answered %3d wrong %2d  s/photo %.2f gpu_peak_max %5.0f MB  fit_used %3d" % (n, len(d), 6*d.p.mean(), m.sum(), (d.p == -20).sum(), d.total_seconds.mean(), d.gpu_peak_mb.max(), d.pose_fit_used.fillna(False).astype(bool).sum()))
