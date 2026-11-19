"""Component checks of the photo view that need no matcher: (a) lens model round trip for both organiser cameras and the Rajasthan corners, (b) how much the vignette removal changes what the matcher sees,
(c) the map level the photo view draws at, with the shipped store levels (3-7) and the levels the dev runs used (3-11), (d) the dev metadata cameras. usage: python component_checks.py"""
import sys, math, glob, os
HEAD = "/home/voidness/Storage/Synced Folders/InterIIT Tech 15/Main_Pipeline_Headless"
sys.path.insert(0, HEAD)
import numpy as np, pandas as pd, cv2
import settings
from geoloc import photo, photo_view, flat_field
from geoloc.levels import level_gsd, level_for_height

REPO = "/home/voidness/Storage/Synced Folders/InterIIT Tech 15/IdeaForge-GeoLoc-InterIIT15"
SETS = {"official(d1, 19N)": REPO + "/dataset/d1/images", "rajasthan(27N)": REPO + "/dataset/Dataset_Official_3/d1/images"}
print("== (d) cameras and altitudes in the dev metadata")
cams = {}
for name, folder in SETS.items():
    m = pd.read_csv(folder + "/metadata.csv")
    cams[name] = m.iloc[0]
    cols = [c for c in m.columns if c.startswith("cam_")]
    print(name, "n=%d alt %.0f-%.0f m, distinct cameras %d" % (len(m), m.alt.min(), m.alt.max(), len(m[cols].drop_duplicates())), m[cols].iloc[0].round(4).to_dict())
    print("   pitch(wrapped) mean/std %.2f/%.2f  roll(wrapped) mean/std %.2f/%.2f" % (photo.wrap_angle(m.pitch).mean(), photo.wrap_angle(m.pitch).std(), photo.wrap_angle(m.roll).mean(), photo.wrap_angle(m.roll).std()))

print("\n== (a) lens round trip: distort(undistort(pixel)) - pixel, pixels")
for name, c in cams.items():
    cam = photo.read_camera({"cam_" + k: c["cam_" + k] for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2")})
    rays = photo_view.undistorted_rays(cam)
    xd, yd = photo.distort(cam, rays[:, 0], rays[:, 1])
    px = cam["fx"] * xd + cam["cx"]; py = cam["fy"] * yd + cam["cy"]
    cols, rows = np.meshgrid(np.arange(320), np.arange(256))
    err = np.hypot(px - cols.ravel(), py - rows.ravel())
    r = np.hypot(rays[:, 0], rays[:, 1])
    print("%-20s max err %.2e px, max undistorted radius %.2f (FOV half-angle %.0f deg), corner ground distance / h = %.2f" % (name, err.max(), r.max(), math.degrees(math.atan(r.max())), r.max()))

print("\n== (b) vignette removal: correlation of the matcher input with and without it (local_normalise 15 px afterwards, as photo_view.own_photo does)")
for name, folder in SETS.items():
    m = pd.read_csv(folder + "/metadata.csv")
    rows = []
    for _, r in m.sample(25, random_state=0).iterrows():
        cam = photo.read_camera({"cam_" + k: r["cam_" + k] for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2")})
        frame = photo.load_photo(os.path.join(folder, r["filename"]))
        a = photo.normalise_frame(flat_field.local_normalise(flat_field.remove_vignette(frame, cam["cx"], cam["cy"], 3), settings.PHOTO_VIEW_LOCAL_NORM_PX))
        b = photo.normalise_frame(flat_field.local_normalise(frame, settings.PHOTO_VIEW_LOCAL_NORM_PX))
        rows.append((r.alt, np.corrcoef(a.ravel(), b.ravel())[0, 1], float(np.abs(flat_field.radial_surface(frame, cam["cx"], cam["cy"], 3) - flat_field.radial_surface(frame, cam["cx"], cam["cy"], 3).mean()).max() / (frame.std() + 1e-9))))
    t = pd.DataFrame(rows, columns=["alt", "corr_with_without", "vignette_amp_over_std"])
    print(name, "corr min %.3f median %.3f | vignette amplitude / frame std: median %.2f max %.2f" % (t.corr_with_without.min(), t.corr_with_without.median(), t.vignette_amp_over_std.median(), t.vignette_amp_over_std.max()))

print("\n== (c) map level of the photo view: photo ground pixel vs level pixel (fx 202)")
fx = 202.4
for levels in ([3, 4, 5, 6, 7], list(range(3, 12))):
    print("store levels", levels[0], "-", levels[-1])
    for h in (100, 200, 350, 600, 1000, 1400, 2000, 3000):
        gsd = h * settings.GEOMETRY["scale"] / fx
        lv = min((abs(math.log(level_gsd(l) / gsd)), l) for l in levels)[1]
        cam = dict(width=320, fx=fx)
        coarse = int(np.clip(math.ceil(2 * math.log2(320 * h / fx / (settings.TILE * settings.MAP_GSD))), min(levels), max(levels)))
        print("   h %4d m photo px %5.2f m -> view level %2d (%.2f m, ratio photo/level %.2f); coarse level %2d canvas %4.0f m vs photo footprint %4.0f m" % (h, gsd, lv, level_gsd(lv), gsd / level_gsd(lv), coarse, 224 * level_gsd(coarse), 320 * h / fx))
