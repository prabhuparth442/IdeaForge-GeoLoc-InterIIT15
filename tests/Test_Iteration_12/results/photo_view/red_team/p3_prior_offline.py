import sys, math, pickle, os, numpy as np, pandas as pd, cv2
HEAD = "/home/voidness/Storage/Synced Folders/InterIIT Tech 15/Main_Pipeline_Headless"
sys.path.insert(0, HEAD); os.chdir(HEAD)
import settings
from geoloc import photo, photo_view, scoring
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")
S = '/tmp/claude-1000/-home-voidness-Storage-Synced-Folders-InterIIT-Tech-15/7794e0a1-ad20-440a-9aae-aa3d00586cd6/scratchpad/pv/out_tilt_full/tilt_matches_full.pkl'
saved = pickle.load(open(S, 'rb'))
RT = "/home/voidness/Storage/Synced Folders/InterIIT Tech 15/IdeaForge-GeoLoc-InterIIT15/tests/Test_Iteration_12/results/photo_view/red_team"
sys.path.insert(0, RT)
from common import M
dev = pd.read_csv("/home/voidness/Storage/Synced Folders/InterIIT Tech 15/IdeaForge-GeoLoc-InterIIT15/tests/Test_Iteration_12/results/coarse_averaging/inputs/meta_highalt_dev332.csv")
mm = M[M.set == "work332"].set_index("filename")

def ground(points, camera, pose, heading, p):
    g = settings.GEOMETRY
    cam = dict(camera); cam['k1'] = camera['k1'] + p[4]
    intrinsic = np.array([[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1]], float)
    dist = np.array([cam['k1'], cam['k2'], cam['p1'], cam['p2'], cam['k3']])
    und = cv2.undistortPointsIter(np.asarray(points, np.float64).reshape(-1, 1, 2), intrinsic, dist, None, None, (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1, 2)
    pitch = g['pitch_sign'] * pose['pitch'] + p[0]; roll = g['roll_sign'] * pose['roll'] + p[1]
    ax, ay, az = photo.camera_axes(pose['yaw'] + g['yaw_offset'] + heading + p[2], pitch, roll)
    height = (pose['height'] + g['height_offset']) * g['scale'] * math.exp(p[3])
    ray = und[:, :1] * ax + und[:, 1:] * ay + az; d = -height / ray[:, 2]
    return ray[:, 0] * d, ray[:, 1] * d

def fit(item, prior, match_px=1.5, f_scale=2.0):
    pose, camera, heading, gsd, pa = item['pose'], item['camera'], item['heading'], item['gsd'], item['pa']
    e0, n0 = ground(pa, camera, pose, heading, np.zeros(6))
    mx = item['cand'][0] + e0 + item['votes_e']; my = item['cand'][1] - (n0 + item['votes_n'])
    pr = np.array(prior + [1e-9]); w = np.ones(len(pa))
    def resid(q):
        e, n = ground(pa, camera, pose, heading, np.r_[q[2:6], 0.0, 0.0]); s = match_px * gsd
        return np.concatenate([w * (q[0] + e - mx) / s, w * (q[1] - n - my) / s, q[2:6] / pr[:4]])
    start = np.r_[np.median(mx - e0), np.median(my + n0), np.zeros(5)]
    steps = np.array([0.1, 0.1, 0.01, 0.01, 0.01, 1e-4, 1e-3])
    return photo_view.robust_least_squares(resid, start, steps, f_scale=f_scale)[:2]

def pts_of(r): return 10 if r <= 1 else 6 if r <= 2 else 3 if r <= 4 else 0 if r <= 8 else -20
base = np.array([1.0, 1.0, 2.0, 0.02])   # the audit's nominal priors (x1)
scales = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
cells = []; P = {}
for it in saved:
    fn = dev.filename[it['photo']]; cells.append(mm.loc[fn, 'cluster'])
    sig = scoring.sigma_m(it['pose']['height'], it['camera']['fx'])
    for s in scales:
        xy = fit(it, prior=list(base * s)); P[(it['photo'], s)] = pts_of(float(np.hypot(*(xy - it['truth']))) / sig)
photos = [it['photo'] for it in saved]
A = pd.DataFrame({s: [P[(k, s)] for k in photos] for s in scales}, index=photos); A['cell'] = cells
print("=== PART 3: offline prior-scale curve on the 125 photos used to choose x0.4 (points/photo, truth-scored) ===")
print(A[scales].mean().round(3).to_string())
print("per-photo SD %.2f -> s.e. of one mean %.2f; paired x0.4 vs x1.0: diff %.3f" % (A[0.4].std(), A[0.4].std() / np.sqrt(len(A)), (A[0.4] - A[1.0]).mean()))
rng = np.random.default_rng(3); cl = A.cell.unique()
bs = []
for _ in range(5000):
    pick = rng.choice(cl, len(cl)); idx = np.concatenate([np.where(A.cell == c)[0] for c in pick]); bs.append((A[0.4].values[idx] - A[1.0].values[idx]).mean())
print("x0.4 minus x1.0: %.3f pts/photo (x6 = %.2f score-scale) cluster-boot 95%% [%.3f, %.3f]" % ((A[0.4] - A[1.0]).mean(), 6 * (A[0.4] - A[1.0]).mean(), *np.percentile(bs, [2.5, 97.5])))
# how often is 0.4 the argmax under cluster resampling
wins = pd.Series(0, index=scales)
for _ in range(2000):
    pick = rng.choice(cl, len(cl)); idx = np.concatenate([np.where(A.cell == c)[0] for c in pick]); wins[A[scales].values[idx].mean(0).argmax() and scales[A[scales].values[idx].mean(0).argmax()] or scales[0]] += 1
print("cluster-bootstrap argmax frequencies of the prior scale:", (wins / 2000).round(3).to_dict())
sel = []; held = 0
for c in cl:
    tr = A[A.cell != c]; te = A[A.cell == c]; b = tr[scales].mean().idxmax(); sel.append(b); held += te[b].sum()
print("LOCO (leave one cell out): chosen scales %s; held-out %.3f vs in-sample best %.3f vs fixed 0.4 %.3f vs fixed 1.0 %.3f pts/photo" % (pd.Series(sel).value_counts().to_dict(), held / len(A), A[scales].mean().max(), A[0.4].mean(), A[1.0].mean()))
# odd / even as in the repo
for tr, te in ((1, 0), (0, 1)):
    i = [k for k in A.index if k % 2 == tr]; j = [k for k in A.index if k % 2 == te]
    b = A.loc[i, scales].mean().idxmax(); print("odd/even: choose on %s -> %.1f; other half %.3f (x1.0 %.3f)" % ('odd' if tr else 'even', b, A.loc[j, b].mean(), A.loc[j, 1.0].mean()))
A.to_csv(RT + "/offline_prior_curve_points.csv")
