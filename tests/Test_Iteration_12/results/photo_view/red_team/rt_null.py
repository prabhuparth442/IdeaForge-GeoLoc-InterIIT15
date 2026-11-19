"""Null test: true place outside the disc. usage: rt_null.py RUNS_DIR NAME JITTER_M SRC_DATASET_CSV. A photo whose hint was moved more than 5600 m from the truth cannot be found, so any 'match' is a false answer."""
import sys, os, math
import numpy as np, pandas as pd
runs, name, jit, src = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
meta = pd.read_csv(src); d = pd.read_csv(os.path.join(runs, "out_rt_" + name, "details.csv"))
dist = []
for i in range(len(meta)):  # replicates cli.move_hint (rng seeded [seed,index], seed 0)
    rng = np.random.default_rng([0, i]); dist.append(jit*math.sqrt(rng.random()))
meta["hint_d"] = dist; d = d.merge(meta[["filename", "hint_d", "alt"]], on="filename")
out = d[d.hint_d > 5600]; a = out[out.status == "match"]
print("%s: photos with truth outside disc %d, false answers %d (%.2f%%)" % (name, len(out), len(a), 100*len(a)/max(len(out), 1)))
if len(a): print(a[["filename", "inliers", "alt", "error_m", "level"]].sort_values("inliers", ascending=False).to_string(index=False))
for lo, hi in ((0, 500), (500, 1000), (1000, 1500), (1500, 2100)):
    b = out[(out.alt >= lo) & (out.alt < hi)]; print("  alt %4d-%4d: n %3d false %d" % (lo, hi, len(b), (b.status == "match").sum()))
inside = d[d.hint_d <= 5400]; print("truth inside disc: n %d answered %d" % (len(inside), (inside.status == "match").sum()))
