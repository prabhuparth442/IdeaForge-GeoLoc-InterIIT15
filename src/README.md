# src: how the pipeline is cut into sections

`python run.py` (one level up) calls these six packages in this order. Each folder has its own README (what to run, inputs, outputs, math) and a `concepts/` folder (teaching notes written from first principles, with derivations).

| # | folder | one-line job | concept files |
|---|---|---|---|
| 1 | [`header/`](header/README.md) | constants, scoring rule, photo and map readers, test-set sampling | 3 |
| 2 | [`image_processing/`](image_processing/README.md) | thermal photo → north-up canvas; map → zoom levels → edge-direction features | 7 |
| 3 | [`coarse_search/`](coarse_search/README.md) | FFT correlation over the whole 5 km disc → 25 candidates | 4 |
| 4 | [`fine_matching/`](fine_matching/README.md) | SuperPoint+LightGlue+RANSAC check of each candidate → inliers, shifts | 3 |
| 5 | [`result_thresholding/`](result_thresholding/README.md) | pick the answer, answer-or-skip, single-photo entry point | 3 |
| 6 | [`evaluation/`](evaluation/README.md) | simulate priors, score with the contest rule, experiments | 6 |

## Whole data flow

```
photo (thermal, h, yaw, pitch, roll) ─► [2] canvas 224², valid mask, 8x56x56 edge features
map GeoTIFF ─► [2] 5 zoom levels ─► edge features per level          (built once per run)
prior (x,y) ─► [3] window of map features ─► FFT correlation (cosine, ZNCC) ─► 25 peaks
peaks ─► [4] per peak: map tile ─► keypoints ─► pairs ─► RANSAC ─► inliers, shift
[5] re-rank by inliers, apply shift, answer iff inliers ≥ 12  ─► (x, y, answer?)
[6] errors / σ ─► points (+10…−20) ─► score out of 60
```

## Suggested reading order for a newcomer

1. `header/concepts/scoring_rule.md` (what we are optimising) → `geographic_coordinates_and_map_reading.md`.
2. `image_processing/concepts/` in the table order (camera → distortion → rotations → resampling → levels → edge features).
3. `coarse_search/concepts/` (cosine vs ZNCC → FFT → peaks).
4. `fine_matching/concepts/`, then `result_thresholding/concepts/`.
5. `evaluation/concepts/` to see how each claim was tested.

## Running anything

Package imports assume `src` is on the path: `PYTHONPATH=src python -c "..."` from the repo root, or use `python run.py` which does this itself. Only `header/`, parts of `image_processing/`, `coarse_search/correlation.py`, `fine_matching/ransac.py`, `result_thresholding/decision.py` and some of `evaluation/` run without the dataset; each README marks which.

## Status of the code (honest list)

* Verified on synthetic data in a test environment: scoring, FFT correlation vs slow loop, RANSAC, decision logic, prior sampling, end-to-end smoke test of the non-neural path.
* **Not yet run on the real dataset or with the neural parts:** MINIMA stage, DINO hint, real-map store build. Treat reported numbers from the original notebook as the reference until `python run.py` has been reproduced.
* Notebook plots were not ported; the map store is rebuilt on every run.
