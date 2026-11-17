import sys, csv, numpy as np, cv2
sys.path.insert(0, "/mnt/e/IdeaForge/work/Main_Pipeline_Headless")
import settings
from geoloc import photo, flat_field
H = "/mnt/e/IdeaForge/datasets/highalt_v2/dataset_dev"
rows = list(csv.DictReader(open(H + "/metadata.csv")))
def edges(img):
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = np.hypot(gx, gy); return float((m > 0.08).mean())   # share of strong-gradient pixels of the stretched 0..1 photo
bands = [(100, 800), (800, 1300), (1300, 1700), (1700, 2100)]
out = []
for lo, hi in bands:
    sel = [r for r in rows if lo <= float(r["alt"]) < hi][:25]
    e0, e1, e2 = [], [], []
    for r in sel:
        frame = photo.load_photo(H + "/images/" + r["filename"]); cam = photo.read_camera(r); pose = photo.read_pose(r)
        e0.append(edges(photo.normalise_frame(frame)))
        e1.append(edges(photo.normalise_frame(flat_field.remove_vignette(frame, cam["cx"], cam["cy"], 3))))
        e2.append(edges(photo.normalise_frame(flat_field.local_normalise(flat_field.remove_vignette(frame, cam["cx"], cam["cy"], 3), 15.0))))
    out.append((lo, hi, len(sel), np.mean(e0), np.mean(e1), np.mean(e2)))
print("share of strong-gradient pixels (photo after stretch + CLAHE)")
print("band | n | as now | vignette removed | + local norm 15 px")
for o in out: print("%d-%d | %d | %.3f | %.3f | %.3f" % o)
# one picture of 4 high photos: as now / vignette removed / + local norm
sel = [r for r in rows if 1300 <= float(r["alt"]) < 2100][:4]
tiles = []
for r in sel:
    frame = photo.load_photo(H + "/images/" + r["filename"]); cam = photo.read_camera(r)
    a = photo.normalise_frame(frame); b = photo.normalise_frame(flat_field.remove_vignette(frame, cam["cx"], cam["cy"], 3)); c = photo.normalise_frame(flat_field.local_normalise(flat_field.remove_vignette(frame, cam["cx"], cam["cy"], 3), 15.0))
    tiles.append(np.concatenate([a, b, c], axis=1))
cv2.imwrite("/mnt/e/IdeaForge/work/reports_interim/flat_field_look.png", (np.concatenate(tiles, axis=0) * 255).astype(np.uint8))
print("heights", [round(float(r["alt"])) for r in sel])
