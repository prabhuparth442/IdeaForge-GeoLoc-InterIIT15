import sys, math, pickle, numpy as np, pandas as pd, cv2
sys.path.insert(0, '.')
import settings
from geoloc import photo, photo_view, scoring
S = '/tmp/claude-1000/-home-voidness-Storage-Synced-Folders-InterIIT-Tech-15/7794e0a1-ad20-440a-9aae-aa3d00586cd6/scratchpad/pv/out_tilt_full'
saved = pickle.load(open(S + '/tilt_matches_full.pkl', 'rb'))

def ground(points, camera, pose, heading, p):
    """p = [dp, dr, dyaw, dlogh, dk1, dfocal_log]"""
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

def fit(item, prior, free_k1=0.0, weights=None, match_px=1.5, f_scale=2.0):
    pose, camera, heading, gsd, pa = item['pose'], item['camera'], item['heading'], item['gsd'], item['pa']
    e0, n0 = ground(pa, camera, pose, heading, np.zeros(6))
    mx = item['cand'][0] + e0 + item['votes_e']; my = item['cand'][1] - (n0 + item['votes_n'])
    pr = np.array(prior + [free_k1 if free_k1 > 0 else 1e-9])
    w = np.ones(len(pa)) if weights is None else weights
    def resid(q):
        e, n = ground(pa, camera, pose, heading, np.r_[q[2:6], q[6] if free_k1 > 0 else 0.0, 0.0])
        s = match_px * gsd
        return np.concatenate([w * (q[0] + e - mx) / s, w * (q[1] - n - my) / s, q[2:6] / pr[:4], [q[6] / pr[4]] if free_k1 > 0 else []])
    start = np.r_[np.median(mx - e0), np.median(my + n0), np.zeros(5)]
    steps = np.array([0.1, 0.1, 0.01, 0.01, 0.01, 1e-4, 1e-3])
    out = photo_view.robust_least_squares(resid, start, steps, f_scale=f_scale)
    return out[:2]

def run(label, **kw):
    rows = []
    for it in saved:
        xy = fit(it, **kw); err = float(np.hypot(*(xy - it['truth']))); sig = scoring.sigma_m(it['pose']['height'], it['camera']['fx'])
        r = err / sig; rows.append((10 if r <= 1 else 6 if r <= 2 else 3 if r <= 4 else 0 if r <= 8 else -20, r, it['photo']))
    a = np.array(rows); odd = a[a[:, 2] % 2 == 1]; even = a[a[:, 2] % 2 == 0]
    print('%-34s points/photo %.3f (odd %.2f even %.2f)  median err/sigma %.2f  within1 %.2f' % (label, a[:, 0].mean(), odd[:, 0].mean(), even[:, 0].mean(), np.median(a[:, 1]), (a[:, 1] <= 1).mean()), flush=True)


base = np.array([0.4, 0.4, 0.8, 0.008])
table = {}
for match_px in (0.75, 1.5, 3.0):
    for f_scale in (1.0, 2.0, 5.0):
        pts = {}
        for it in saved:
            xy = fit(it, prior=list(base), match_px=match_px, f_scale=f_scale); err = float(np.hypot(*(xy - it['truth']))); sig = scoring.sigma_m(it['pose']['height'], it['camera']['fx']); r = err / sig
            pts[it['photo']] = 10 if r <= 1 else 6 if r <= 2 else 3 if r <= 4 else 0 if r <= 8 else -20
        table[(match_px, f_scale)] = pts
        print('match sd %.2f px, f_scale %.1f : points/photo %.3f' % (match_px, f_scale, np.mean(list(pts.values()))), flush=True)
photos = sorted(table[(1.5, 2.0)])
for train, test in ((1, 0), (0, 1)):
    tr = [k for k in photos if k % 2 == train]; te = [k for k in photos if k % 2 == test]
    best = max(table, key=lambda sc: np.mean([table[sc][k] for k in tr]))
    print('choose on %s -> %s ; test on other half: %.3f (default 1.5/2.0: %.3f)' % ('odd' if train else 'even', best, np.mean([table[best][k] for k in te]), np.mean([table[(1.5, 2.0)][k] for k in te])))
