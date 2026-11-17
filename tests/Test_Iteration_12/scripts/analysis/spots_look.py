import sys, csv, numpy as np, cv2
sys.path.insert(0, "/mnt/e/IdeaForge/work/Main_Pipeline_Headless")
import settings
from geoloc import photo, levels, map_store
settings.LEVELS = [3,4,5,6,7,8,9,10,11]
H = "/mnt/e/IdeaForge/datasets/highalt_v2"
store = map_store.MapStore(H + "/store")
rows = [r for r in csv.DictReader(open(H + "/dataset_dev/metadata.csv")) if 100 <= float(r["alt"]) <= 200]
# photos with many dark spots: pick by share of very dark pixels after stretch
def dark_share(r):
    f = photo.normalise_frame(photo.load_photo(H + "/dataset_dev/images/" + r["filename"])); return float((f < 0.15).mean())
rows.sort(key=dark_share, reverse=True)
tiles = []
for r in rows[:3]:
    camera = photo.read_camera(r); pose = photo.read_pose(r)
    level = levels.level_for_height(pose["height"], camera)
    ph = photo.prepare_photo(H + "/dataset_dev/images/" + r["filename"], pose, camera, level)
    canvas, valid = photo.draw_canvas(ph, pose, camera, level, 0.0, 1.0)
    x, y = store.gps_to_metres(float(r["lat_gtruth"]), float(r["lon_gtruth"]))
    mp = levels.map_piece(store.levels[level], level, x, y)
    # raw (unstretched-by-clahe) frame as stored: percentile stretch only
    raw = photo.load_photo(H + "/dataset_dev/images/" + r["filename"]).astype(np.float32); lo, hi = np.percentile(raw, [1, 99]); raw = np.clip((raw - lo) / (hi - lo), 0, 1)
    tiles.append(np.concatenate([cv2.resize(raw, (224, 224)), canvas.astype(np.float32), mp], axis=1))
    print(r["filename"][:8], "height %.0f m" % pose["height"], "level", level, "pixel %.2f m" % levels.level_gsd(level))
cv2.imwrite("/mnt/e/IdeaForge/work/reports_interim/spots_look.png", (np.concatenate(tiles, axis=0) * 255).astype(np.uint8))
