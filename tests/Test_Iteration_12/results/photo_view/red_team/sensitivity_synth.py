"""Red-team test of the photo-view pose fit (geoloc/photo_view.py fit_pose) on synthetic geometry. CPU only, no weights, no data, no source change.

World: a camera with the Rajasthan intrinsics (real distortion) at a known true pose looks at flat ground. The pipeline sees the METADATA pose = true pose + error.
The matcher is replaced by exact geometry: photo pixel p sees the ground point truth + G_true(p); the map drawn with the metadata pose around candidate c shows c + G_meta(q) at q,
so q is the pixel where G_meta(q) = G_true(p) + truth - c (inverse projection, as in photo.draw_canvas). Match noise N(0, 1.2 px) per coordinate (the fit's own rms on real data is 1.4-1.9 px).
The query is drawn with the heading (-5, 0, 5) and scale (1.0, 1.05) nearest to the truth, as the real winner would be, so the prior is centred on that query exactly as in the pipeline.
Estimators: median vote (what the pipeline did before the fit), fit with the PV-X prior (0.4, 0.4, 0.8 deg, 0.8 %), the settings.py default prior (1, 1, 2, 2 %) and 3 x that.
The pipeline drops the fit when it is more than 3 sigma from the median vote (then the median is the answer): replicated.
usage: python sensitivity_synth.py OUT.csv [N_TRIALS]
"""
import math
import os
import sys
from multiprocessing import Pool

HEAD = "/home/voidness/Storage/Synced Folders/InterIIT Tech 15/Main_Pipeline_Headless"
sys.path.insert(0, HEAD)
import numpy as np
import pandas as pd

import settings
from geoloc import photo, photo_view, scoring
from geoloc.levels import level_for_height, level_gsd

CAMERA = dict(fx=202.4054565, fy=202.302597, cx=163.4900665, cy=121.6884155, k1=-0.27758503, k2=0.064913362, k3=0.0, p1=0.000859941, p2=-0.000142806, width=320, height=256)
RAYS = photo_view.undistorted_rays(CAMERA)
G = settings.GEOMETRY
BASE = np.array([1.0, 1.0, 2.0, 0.02])            # the audit's assumed noise: pitch, roll, yaw (deg), height (share). Settings default prior = this.
PRIORS = {"fit_pvx": (0.4, 0.4, 0.8, 0.008), "fit_default": (1.0, 1.0, 2.0, 0.02), "fit_x3": (3.0, 3.0, 6.0, 0.06)}
MATCH_PX = 1.2


def project(east, north, pose, heading, scale_factor):
    """Pixel where the drawn map (metadata pose, query heading/scale) shows the ground point (east, north) from the nadir point. Inverse of ground_grid."""
    pitch = G["pitch_sign"] * pose["pitch"]
    roll = G["roll_sign"] * pose["roll"]
    ax, ay, az = photo.camera_axes(pose["yaw"] + G["yaw_offset"] + heading, pitch, roll)
    height = (pose["height"] + G["height_offset"]) * G["scale"] * scale_factor
    r = np.stack([east, north, np.full_like(east, -height)], 1)
    x, y, z = r @ ax, r @ ay, r @ az
    xd, yd = photo.distort(CAMERA, x / z, y / z)
    return np.stack([CAMERA["fx"] * xd + CAMERA["cx"], CAMERA["fy"] * yd + CAMERA["cy"]], 1)


def true_ground(points, cam, pose, slope_deg, slope_az_deg):
    """Horizontal (east, north) metres from the nadir point of the ground point each photo pixel sees, for the TRUE camera and pose over ground that slopes by slope_deg (0 = flat, as the pipeline assumes)."""
    intrinsic = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], float)
    dist = np.array([cam["k1"], cam["k2"], cam["p1"], cam["p2"], cam["k3"]])
    import cv2
    und = cv2.undistortPointsIter(np.asarray(points, np.float64).reshape(-1, 1, 2), intrinsic, dist, None, None, (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1, 2)
    ax, ay, az = photo.camera_axes(pose["yaw"] + G["yaw_offset"], G["pitch_sign"] * pose["pitch"], G["roll_sign"] * pose["roll"])
    ray = und[:, :1] * ax + und[:, 1:] * ay + az
    height = (pose["height"] + G["height_offset"]) * G["scale"]
    t = math.tan(math.radians(slope_deg))
    phi = math.radians(slope_az_deg)
    d = -height / (ray[:, 2] - t * (ray[:, 0] * math.cos(phi) + ray[:, 1] * math.sin(phi)))
    return ray[:, 0] * d, ray[:, 1] * d


def one_trial(args):
    height, noise_k, bias, seed, clustered, fixed_query, mismatch = args
    cam_t = dict(CAMERA, fx=CAMERA["fx"] * mismatch["fx"], fy=CAMERA["fy"] * mismatch["fx"], k1=CAMERA["k1"] + mismatch["k1"])
    rng = np.random.default_rng(seed)
    true = dict(height=height, yaw=rng.uniform(0, 360), pitch=rng.uniform(-6, 6), roll=rng.uniform(-6, 6))
    err = np.array(bias, float) + noise_k * BASE * rng.standard_normal(4)          # metadata - truth: pitch, roll, yaw, height share
    meta = dict(height=height * (1 + err[3]), yaw=true["yaw"] + err[2], pitch=true["pitch"] + err[0], roll=true["roll"] + err[1])
    level = level_for_height(meta["height"], CAMERA)
    limit = settings.MAX_SHIFT_CELLS * settings.CELL * level_gsd(level)
    ang = rng.uniform(0, 2 * math.pi)
    off = rng.uniform(0.2, 0.8) * min(limit, 60.0)
    cand = np.array([off * math.cos(ang), off * math.sin(ang)])                      # candidate minus truth, (east, south) metres; truth at (0, 0)
    # query: heading nearest to -yaw error (drawn yaw = meta + heading should equal the true yaw), scale nearest to the true/meta height ratio
    heading = 0.0 if fixed_query else min([-5.0, 0.0, 5.0], key=lambda h: abs(h + err[2]))
    scale_factor = 1.0 if fixed_query else min([1.0, 1.05], key=lambda s: abs(s - 1 / (1 + err[3])))
    n = 70
    if clustered:
        centre = np.array([rng.uniform(80, 240), rng.uniform(60, 196)])
        pa = centre + rng.standard_normal((n, 2)) * np.array([35.0, 28.0])
        pa = pa[(pa[:, 0] > 4) & (pa[:, 0] < 315) & (pa[:, 1] > 4) & (pa[:, 1] < 251)]
    else:
        pa = np.stack([rng.uniform(8, 312, n), rng.uniform(8, 248, n)], 1)
    ge, gn = true_ground(pa, cam_t, true, mismatch["slope"], rng.uniform(0, 360))   # what the photo pixel sees, relative to the TRUE nadir point (true camera, true pose, ground as given)
    want_e = ge + (0.0 - cand[0])
    want_n = gn - (0.0 - cand[1])
    pb = project(want_e, want_n, meta, heading, scale_factor) + rng.standard_normal((len(pa), 2)) * MATCH_PX
    ok = (pb[:, 0] > 2) & (pb[:, 0] < 317) & (pb[:, 1] > 2) & (pb[:, 1] < 253)
    pa, pb = pa[ok].astype(np.float32), pb[ok].astype(np.float32)
    if len(pa) < 12:
        return None
    grid = photo_view.ground_grid(RAYS, meta, CAMERA, heading, scale_factor)
    match = dict(pa=pa, pb=pb, grid=grid, heading=heading, scale_factor=scale_factor, cand=(float(cand[0]), float(cand[1])))
    de = np.median(photo_view.sample(grid[0], pb) - photo_view.sample(grid[0], pa))
    dn = np.median(photo_view.sample(grid[1], pb) - photo_view.sample(grid[1], pa))
    median_xy = cand + np.array([de, -dn])
    sigma = scoring.sigma_m(meta["height"], CAMERA["fx"])
    gsd = meta["height"] * G["scale"] / CAMERA["fx"]
    out = dict(height=height, seed=seed, n_matches=len(pa), sigma=sigma, err_tilt=math.hypot(err[0], err[1]), err_yaw=err[2], err_h=err[3], heading=heading, scale_factor=scale_factor)
    out["median"] = math.hypot(*median_xy) / sigma
    for name, prior in PRIORS.items():
        settings.PHOTO_VIEW_PRIOR_TILT_DEG, settings.PHOTO_VIEW_PRIOR_YAW_DEG, settings.PHOTO_VIEW_PRIOR_HEIGHT = prior[0], prior[2], prior[3]
        x, y, corr, rms = photo_view.fit_pose(match, meta, CAMERA, gsd)
        dropped = math.hypot(x - median_xy[0], y - median_xy[1]) > settings.PHOTO_VIEW_FIT_MAX_MOVE_SIGMA * sigma
        final = median_xy if dropped else np.array([x, y])
        out[name] = math.hypot(*final) / sigma
        out[name + "_raw"] = math.hypot(x, y) / sigma
        out[name + "_dropped"] = dropped
        if name == "fit_pvx":
            out.update(c_pitch=corr[0], c_roll=corr[1], c_yaw=corr[2], c_height=corr[3], rms_px=rms)
    return out


def points(ratio):
    return np.where(ratio <= 1, 10, np.where(ratio <= 2, 6, np.where(ratio <= 4, 3, np.where(ratio <= 8, 0, -20))))


SCENARIOS = [  # name, noise_k, bias (pitch, roll, yaw deg, height share), clustered, fixed_query
    ("noise x0 (exact metadata)", 0.0, (0, 0, 0, 0), False, False),
    ("noise x0.5", 0.5, (0, 0, 0, 0), False, False),
    ("noise x1 (audit)", 1.0, (0, 0, 0, 0), False, False),
    ("noise x2", 2.0, (0, 0, 0, 0), False, False),
    ("noise x3", 3.0, (0, 0, 0, 0), False, False),
    ("noise x1 + tilt bias 1/1 deg", 1.0, (1, 1, 0, 0), False, False),
    ("noise x1 + tilt bias 2/2 deg", 1.0, (2, 2, 0, 0), False, False),
    ("noise x1 + yaw bias 3 deg", 1.0, (0, 0, 3, 0), False, False),
    ("noise x1 + height bias +4 %", 1.0, (0, 0, 0, 0.04), False, False),
    ("noise x1 + height bias -8 %", 1.0, (0, 0, 0, -0.08), False, False),
    ("noise x2 + tilt bias 1/1", 2.0, (1, 1, 0, 0), False, False),
    ("noise x1, matches clustered", 1.0, (0, 0, 0, 0), True, False),
    ("noise x3, matches clustered", 3.0, (0, 0, 0, 0), True, False),
    ("noise x1, query fixed 0/1.0", 1.0, (0, 0, 0, 0), False, True),
    ("noise x1 + ground slope 1 deg", 1.0, (0, 0, 0, 0), False, False, dict(slope=1.0)),
    ("noise x1 + ground slope 3 deg", 1.0, (0, 0, 0, 0), False, False, dict(slope=3.0)),
    ("noise x1 + focal 1 % off", 1.0, (0, 0, 0, 0), False, False, dict(fx=1.01)),
    ("noise x1 + focal 3 % off", 1.0, (0, 0, 0, 0), False, False, dict(fx=1.03)),
    ("noise x1 + k1 off by 0.03", 1.0, (0, 0, 0, 0), False, False, dict(k1=0.03)),
    ("noise x0 + ground slope 3 deg", 0.0, (0, 0, 0, 0), False, False, dict(slope=3.0)),
    ("noise x0 + focal 3 % off", 0.0, (0, 0, 0, 0), False, False, dict(fx=1.03)),
]


def main():
    out_csv = sys.argv[1]
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    jobs, tags = [], []
    for height in (200.0, 800.0, 1600.0):
        for si, sc in enumerate(SCENARIOS):
            name, k, bias, clustered, fixed = sc[:5]
            mismatch = {**dict(slope=0.0, fx=1.0, k1=0.0), **(sc[5] if len(sc) > 5 else {})}
            for t in range(trials):
                jobs.append((height, k, bias, 1000 * si + t + int(height), clustered, fixed, mismatch))
                tags.append((height, name))
    with Pool(int(os.environ.get("NPROC", "6"))) as pool:
        results = pool.map(one_trial, jobs, chunksize=8)
    rows = []
    for (height, name), r in zip(tags, results):
        if r is not None:
            r["scenario"] = name
            rows.append(r)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
