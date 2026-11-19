from common import *
S = "/tmp/claude-1000/-home-voidness-Storage-Synced-Folders-InterIIT-Tech-15/7794e0a1-ad20-440a-9aae-aa3d00586cd6/scratchpad/pv/"
print("=== PART 2: the ladder of choices, step by step, with cluster-bootstrap CI of each step ===")
def L(p): return load(p)
def step(name, a, b, setname):
    da, db = L(a), L(b)
    d = da[["filename", "p", "cluster", "alt"]].merge(db[["filename", "p"]], on="filename", suffixes=("_a", "_b")); d["gain"] = d.p_b - d.p_a
    cb = 6 * np.percentile(cluster_boot(d), [2.5, 97.5]); P = (cluster_boot(d) <= 0).mean()
    print("  %-9s %-62s %6.2f -> %6.2f  step %+.2f  cluster-95%% [%+.2f,%+.2f]  P(<=0)=%.3f  n=%d wrong %d->%d" % (setname, name, 6 * d.p_a.mean(), 6 * d.p_b.mean(), 6 * d.gain.mean(), *cb, P, len(d), (da.p == -20).sum(), (db.p == -20).sum()))
    return d
print("working 332 (dev), chronological order of choices:")
step("base(XoFTR>=800) -> PV >=1000 m, bar20 (no fit)", BASE["work332"], B + "photo_view/details_pipeline_bar20.csv", "work332")
step("  + pose fit, priors 1/1/2/2% (audit noise)", B + "photo_view/details_pipeline_bar20.csv", B + "photo_view/details_pipeline_bar20_posefit.csv", "work332")
step("  gate 1000 -> 800", B + "photo_view/details_pipeline_bar20_posefit.csv", S + "pvs_dev_g800/details.csv", "work332")
step("  gate 800 -> 600", S + "pvs_dev_g800/details.csv", S + "pvs_dev_g600/details.csv", "work332")
step("  gate 600 -> 0 (every photo)", S + "pvs_dev_g600/details.csv", S + "pvs_dev_g0/details.csv", "work332")
step("  priors x1 -> x0.4 (pv2 copy, gate 0)", S + "pvs_dev_g0/details.csv", S + "pv2_dev_g0p04/details.csv", "work332")
step("  bar20/agree-on -> bar12/agree-off (LoFTR<800)", S + "pv2_dev_g0p04/details.csv", S + "pv2_dev_b12na/details.csv", "work332")
step("  bar12 agree-off -> agree-on", S + "pv2_dev_b12na/details.csv", S + "pv3_dev_b12a/details.csv", "work332")
step("  LoFTR<800 -> XoFTR for all (= PV-X)", S + "pv3_dev_b12a/details.csv", S + "pv3_dev_x0/details.csv", "work332")
print("old official 500 (dev; everything below chosen/inspected with it):")
step("base -> PV all, bar20, priors x1 (pvs g0b20)", BASE["official500"], S + "pvs_official_g0b20/details.csv", "official")
step("  bar20 -> bar12 (pvs)", S + "pvs_official_g0b20/details.csv", S + "pvs_official_g0b12/details.csv", "official")
step("  -> agree-off bar12, priors x0.4 (pv2_b12na)", S + "pvs_official_g0b12/details.csv", S + "pv2_official_b12na/details.csv", "official")
step("  -> agree-on bar12 (pv3_b12a)", S + "pv2_official_b12na/details.csv", S + "pv3_official_b12a/details.csv", "official")
step("  -> XoFTR for all (= PV-X)", S + "pv3_official_b12a/details.csv", S + "pv3_official_x0/details.csv", "official")
print("NUDT 300 (dev):")
step("base -> PV all, bar20 (pvs g0b20)", BASE["nudt300"], S + "pvs_nudt_g0b20/details.csv", "nudt")
step("  bar20 -> bar12 (pvs)", S + "pvs_nudt_g0b20/details.csv", S + "pvs_nudt_g0b12/details.csv", "nudt")
step("  bar12 -> agree-off (pv2_b12na)", S + "pvs_nudt_g0b12/details.csv", S + "pv2_nudt_b12na/details.csv", "nudt")
step("  agree-on bar12 (pv3_b12a) vs agree-off", S + "pv2_nudt_b12na/details.csv", S + "pv3_nudt_b12a/details.csv", "nudt")
step("  -> XoFTR for all (= PV-X)", S + "pv3_nudt_b12a/details.csv", S + "pv3_nudt_x0/details.csv", "nudt")
print("\nTotal PV-X gain that exists BEFORE any tuning on the set (PV at fixed 'a priori' settings: bar 20, gate>=1000 for highalt, audit priors):")
print("  working332: base 39.70 -> 46.07 (+6.37); tuned steps after: +4.62 (to 50.69).  official500: base 52.69 -> 52.08 (PV all bar20: -0.61); every point of the +7.31 came after bar/XoFTR choices made on official.  nudt300: 48.52 -> 52.28 (+3.76) then XoFTR-all -0.20.")

print("\n=== LOCO re-selection of the gate height on the working set (4 real runs, same settings otherwise) ===")
runs = {1000: B + "photo_view/details_pipeline_bar20_posefit.csv", 800: S + "pvs_dev_g800/details.csv", 600: S + "pvs_dev_g600/details.csv", 0: S + "pvs_dev_g0/details.csv"}
D = None
for g, p in runs.items():
    d = L(p)[["filename", "p", "cluster"]].rename(columns={"p": "p%d" % g}); D = d if D is None else D.merge(d[["filename", "p%d" % g]], on="filename")
cols = ["p%d" % g for g in runs]
cl = D.cluster.unique(); sel = []; held = 0; ins = D[cols].mean().max()
for c in cl:
    tr = D[D.cluster != c]; te = D[D.cluster == c]; best = tr[cols].mean().idxmax(); sel.append(best); held += te[best].sum()
print("in-sample best %.3f pts/photo (%.2f score); LOCO-selected held-out %.3f (%.2f score); selected counts %s" % (ins, 6 * ins, held / len(D), 6 * held / len(D), pd.Series(sel).value_counts().to_dict()))
print("fixed g0 = %.2f; fixed g1000 = %.2f (a-priori choice)" % (6 * D.p0.mean(), 6 * D.p1000.mean()))
