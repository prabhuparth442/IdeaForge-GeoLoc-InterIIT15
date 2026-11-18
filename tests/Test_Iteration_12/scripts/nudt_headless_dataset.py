"""NUDT valid frames -> the headless pipeline's dataset format (images/ + metadata.csv), 320 x 256, camera scaled by 1/4.
usage: python nudt_headless_dataset.py NUDT_DIR OUT_DIR N_PHOTOS
Pitch and roll are written as 0 (the vendor's -90/180 means straight down); the yaw is the raw column (use yaw_offset 0 and scale 1.0 in the geometry)."""
import glob, os, sys
import cv2, numpy as np, pandas as pd

nudt, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
for flight_dir in sorted(glob.glob(os.path.join(nudt, "Thermal-UAV", "valid", "changsha", "*"))):
    table = pd.read_csv(os.path.join(flight_dir, "Thermal_info.csv"))
    for _, r in table.iterrows():
        rows.append((os.path.join(flight_dir, "Thermal", r["filename"]), r))
rng = np.random.default_rng(1)
pick = sorted(rng.choice(len(rows), size=min(n, len(rows)), replace=False))
os.makedirs(os.path.join(out, "images"), exist_ok=True)
scale = 4.0
fx = 12.0 / 9.83 * 1280 / scale
meta = []
for k, i in enumerate(pick):
    path, r = rows[i]
    frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    small = cv2.resize(frame, (320, 256), interpolation=cv2.INTER_AREA)
    name = "nudt_%04d.png" % k
    cv2.imwrite(os.path.join(out, "images", name), small)
    meta.append(dict(filename=name, lat_sc=r["lat"], lon_sc=r["lon"], lat_gtruth=r["lat"], lon_gtruth=r["lon"], alt=r["rel_height"], alt_takeoff_pt_dem=0,
                     yaw=r["yaw"], pitch=0.0, roll=0.0, cam_fx=fx, cam_fy=fx, cam_cx=159.5, cam_cy=127.5, cam_k1=0, cam_k2=0, cam_k3=0, cam_p1=0, cam_p2=0))
pd.DataFrame(meta).to_csv(os.path.join(out, "metadata.csv"), index=False)
print(len(meta), "photos, fx", fx)
