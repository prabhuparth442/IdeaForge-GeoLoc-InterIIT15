"""Unseen-photo check: the ODD dev photos (the arms so far used the even ones) with a new noise draw (seed 22)."""
import os, sys, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
meta = pd.read_csv(os.path.join(src, "metadata.csv")).iloc[1::2].reset_index(drop=True)
rng = np.random.default_rng(22); N = len(meta); dp, dr, dy, dh = rng.normal(size=(4, N))
def write(name, m):
    d = os.path.join(out, name); os.makedirs(os.path.join(d, "images"), exist_ok=True)
    m.to_csv(os.path.join(d, "metadata.csv"), index=False); meta.to_csv(os.path.join(d, "metadata_orig.csv"), index=False)
    for f in m.filename:
        p = os.path.join(d, "images", f)
        if not os.path.exists(p): os.link(os.path.join(src, "images", f), p)
    print(name, len(m))
write("oddcontrol", meta.copy())
for k in (2, 3):
    m = meta.copy(); m["pitch"] += k*dp; m["roll"] += k*dr; m["yaw"] += 2*k*dy; m["alt"] *= 1 + 0.02*k*dh; write("oddnoise%dx" % k, m)
m = meta.copy(); m["alt"] *= 1.10; write("oddscale_p10", m)
m = meta.copy(); m["pitch"] += 2; m["roll"] += 2; write("oddtilt2", m)
