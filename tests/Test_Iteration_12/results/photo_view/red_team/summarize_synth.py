"""Summary tables of sensitivity_synth_results.csv: median error/sigma, mean points per photo, share wrong (>8 sigma), share of fits dropped by the 3 sigma rule."""
import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sensitivity_synth_results.csv")
def pts(r): return np.where(r <= 1, 10, np.where(r <= 2, 6, np.where(r <= 4, 3, np.where(r <= 8, 0, -20))))
ests = ["median", "fit_pvx", "fit_default", "fit_x3"]
order = list(dict.fromkeys(d.scenario))
for h in sorted(d.height.unique()):
    print("\n=== height %d m (sigma %.1f m), %d trials per row" % (h, d[d.height == h].sigma.mean(), (d[(d.height == h) & (d.scenario == order[0])]).shape[0]))
    print("%-34s | %s | drop%% pvx/def" % ("scenario", " | ".join("%-18s" % (e + " pts med") for e in ests)))
    for s in order:
        x = d[(d.height == h) & (d.scenario == s)]
        cells = []
        for e in ests:
            cells.append("%5.2f %4.2f  w%4.1f%%" % (pts(x[e].values).mean(), x[e].median(), 100 * (x[e] > 8).mean()))
        print("%-34s | %s | %4.0f / %4.0f" % (s, " | ".join(cells), 100 * x.fit_pvx_dropped.mean(), 100 * x.fit_default_dropped.mean()))
print("\nraw fit (no 3-sigma drop) mean points, pvx / default / x3, and median:")
for h in sorted(d.height.unique()):
    for s in order:
        x = d[(d.height == h) & (d.scenario == s)]
        print("%5d %-34s median %.2f  pvx_raw %.2f  default_raw %.2f  x3_raw %.2f" % (h, s, pts(x["median"].values).mean(), pts(x.fit_pvx_raw.values).mean(), pts(x.fit_default_raw.values).mean(), pts(x.fit_x3_raw.values).mean()))
