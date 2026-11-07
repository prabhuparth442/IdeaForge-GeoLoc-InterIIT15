"""Headless full-NUDT run of the unchanged pipeline, one photo at a time (same loop as Single_Image_Run.ipynb).
Writes a per-photo CSV with true_lat/true_lon (what Step1_Oracle_Tile.ipynb needs), saved every 50 photos and resumable.
usage: python run_nudt_full.py SPLIT N_EVAL [COARSE: fused|edge_zncc]   (SPLIT: test|valid|train|all; N_EVAL: 0 = every photo)"""
import os, sys, math, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(folder))
os.chdir(ROOT)
split, n_eval = sys.argv[1], int(sys.argv[2])
COARSE = sys.argv[3] if len(sys.argv) > 3 else "fused"
SEED = 958476
from train.datasets import photo_sets  # noqa: F401  (must come first, as in the notebooks)
from single_photo_pipeline import SinglePhotoPipeline, STAGES
from src.header.scoring import points_for_error, final_score

pipeline = SinglePhotoPipeline("nudt", COARSE, semantic_layer=True, semantic_decision=True, gamma=1.0, rescue_inliers=None, rescue_z=None,
                               k_peaks=50, m_fine=25, matcher="loftr_minima", min_inliers=8, ransac_pixels=None, matcher_confidence=0.3,
                               satellite="new", map_margin_m=None, loftr_coarse_threshold=None, nudt_split=split)
pipeline.setup(100000, SEED)
ts = pipeline.test_set
count = len(ts.items) if n_eval == 0 else min(n_eval, len(ts.items))
print("loaded %d photos, running %d" % (len(ts.items), count), flush=True)
out = ROOT / "tests" / "Test_Iteration_3_Thresholding" / "results" / ("nudt_%s_%s_%d.csv" % (COARSE, split, count))
rows = pd.read_csv(out).to_dict("records") if out.exists() else []
done = {r["photo"] for r in rows}
t0 = time.time()
for index in range(count):
    if index in done:
        continue
    item = ts.items[index]
    try:
        result = pipeline.run(item, ts.priors[index])
    except Exception as error:
        print("photo %d failed: %s: %s" % (index, type(error).__name__, str(error)[:120]), flush=True)
        continue
    tx, ty = ts.truths[index]
    error_m = math.hypot(result["x"] - tx, result["y"] - ty)
    earned = points_for_error(error_m, item["rel_height"], pipeline.camera.fx)
    lat, lon = pipeline.satellite_map.metres_to_gps(result["x"], result["y"])
    row = dict(photo=index, true_lat=item["lat"], true_lon=item["lon"], lat=lat, lon=lon, height=item["rel_height"], level=result["level"],
               sigma_m=ts.sigmas[index], error_m=error_m, error_sigma=error_m / ts.sigmas[index], answered=result["answered"],
               inliers=result["inliers"], chosen=result["chosen"], z_chosen=result["z_chosen"], points=earned if result["answered"] else 0)
    for label in STAGES:
        row["t_" + label] = result["seconds"][label]
    row["t_total"] = result["total"]; row["gpu_peak_mb"] = result["gpu_peak_mb"]
    rows.append(row)
    if len(rows) % 50 == 0 or index == count - 1:
        df = pd.DataFrame(rows); df.to_csv(out, index=False)
        print("%d/%d | score %.1f /60 | answered %.2f | %.1f s/photo" % (len(df), count, final_score(list(df["points"])), df["answered"].mean(), df["t_total"].mean()), flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("saved", out, "in %.0f s" % (time.time() - t0))
